#include "decoder_layer.h"

#include <iostream>

#include "aes_stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>  // for std::default_random_engine

#include "func_utils.h"

#define MODULO (1LL << 32)
#define SHIFT_AMT 129

#define MULTKEY_POOL_SIZE 1024

#include "Eigen/Dense"
#include "macro.h"

namespace {
std::default_random_engine engine(0);
std::vector<uint32_t> qk_add_dec_key_buffer;   // flattened 4-D Tensor,
std::vector<uint32_t> qk_mult_dec_key_buffer;  // flattened 4-D Tensor
std::vector<int> qk_unshift_buffer;            // flattened 3-D Tensor

std::vector<uint32_t> pv_add_dec_key_buffer;   // flattened 4-D Tensor,
std::vector<uint32_t> pv_mult_dec_key_buffer;  // flattened 4-D Tensor
std::vector<int> pv_unshift_buffer;            // flattened 3-D Tensor
// notice this shared across all layers, assume no two layers happen at the same time
}  // namespace

namespace jpyo0803 {

DecoderLayer::DecoderLayer(int layer_idx, int hidden_size,
                           int intermediate_size, int max_position_embeddings,
                           int num_attention_heads, int num_key_value_heads,
                           int enc_key_pool_size, bool enable_linear_encryption,
                           bool enable_atten_encryption)
    : layer_idx_(layer_idx),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      max_position_embeddings_(max_position_embeddings),
      num_attention_heads_(num_attention_heads),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(hidden_size / num_attention_heads),
      enc_key_pool_size_(enc_key_pool_size),
      num_key_value_groups_(num_attention_heads / num_key_value_heads),
      present_token_len_(0),
      culmulative_token_len_(0),
      bsz_(0),
      is_qk_key_generated_(false),
      is_qk_dec_key_generated_(false),
      is_qk_add_buffer_generated_(false),
      is_qk_mult_buffer_generated_(false),
      is_qk_unshift_buffer_generated_(false),
      is_qk_shift_q_done_(false),
      is_qk_shift_k_done_(false),
      is_pv_key_generated_(false),
      is_pv_dec_key_generated_(false),
      is_pv_add_buffer_generated_(false),
      is_pv_mult_buffer_generated_(false),
      is_pv_unshift_buffer_generated_(false),
      is_pv_shift_p_done_(false),
      is_pv_shift_v_done_(false),
      enable_linear_encryption_(enable_linear_encryption),
      enable_atten_encryption_(enable_atten_encryption) {

#if SGX_ENABLE == 0
  std::cout << "Decoder Layer " << layer_idx_ << " is created." << std::endl;
#endif
  q_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  k_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  v_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  o_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  up_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  gate_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  down_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(intermediate_size_));

  q_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(num_attention_heads_ * head_dim_));
  k_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(num_key_value_heads_ * head_dim_));
  v_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(num_key_value_heads_ * head_dim_));
  o_dec_key_ = std::vector<std::vector<int>>(enc_key_pool_size_,
                                             std::vector<int>(hidden_size_));
  up_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(intermediate_size_));
  gate_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(intermediate_size_));
  down_dec_key_ = std::vector<std::vector<int>>(enc_key_pool_size_,
                                                std::vector<int>(hidden_size_));

  q_weight_scales_ = std::vector<float>(num_attention_heads_ * head_dim_);
  k_weight_scales_ = std::vector<float>(num_key_value_heads_ * head_dim_);
  v_weight_scales_ = std::vector<float>(num_key_value_heads_ * head_dim_);
  o_weight_scales_ = std::vector<float>(hidden_size_);

  up_weight_scales_ = std::vector<float>(intermediate_size_);
  gate_weight_scales_ = std::vector<float>(intermediate_size_);
  down_weight_scales_ = std::vector<float>(hidden_size_);

  x_mult_key_pool_.resize(MULTKEY_POOL_SIZE);
  y_mult_key_pool_.resize(MULTKEY_POOL_SIZE);

  for (int i = 0; i < MULTKEY_POOL_SIZE; ++i) {
    x_mult_key_pool_[i] = GenerateMultKey();
    y_mult_key_pool_[i] = GenerateMultKey();
  }

  precomputed_key_inv_ = std::vector<std::vector<uint32_t>>(
      MULTKEY_POOL_SIZE, std::vector<uint32_t>(MULTKEY_POOL_SIZE));

  for (int i = 0; i < MULTKEY_POOL_SIZE; ++i) {
    for (int j = 0; j < MULTKEY_POOL_SIZE; ++j) {
      uint64_t ab = (uint64_t)x_mult_key_pool_[i] * y_mult_key_pool_[j];
      precomputed_key_inv_[i][j] =
          (uint32_t)RepeatedSqr(ab, (1LL << 31) - 1, MODULO);
#if CHECK_SANITY == 1
      uint64_t test = (uint64_t)precomputed_key_inv_[i][j] * ab % MODULO;
      ASSERT_ALWAYS(test == 1, "Key inverse is not correct!");
      if (test != 1) {
#if SGX_ENABLE == 0
        std::cout << "Key inverse is not correct!" << std::endl;
        exit(-1);
#endif
      }
#endif
    }
  }

  for (int i = 0; i < head_dim_; ++i) {
    qk_permuted_index_.push_back(i);
    pv_permuted_index_.push_back(i);
  }

  std::shuffle(qk_permuted_index_.begin(), qk_permuted_index_.end(), engine);
  std::shuffle(pv_permuted_index_.begin(), pv_permuted_index_.end(), engine);

  input_layernorm_weights_.resize(hidden_size_);
  post_attention_layernorm_weights_.resize(hidden_size_);

// #if CHECK_SANITY == 1
//   ASSERT_ALWAYS(q_enc_key_pool_.size() == enc_key_pool_size_,
//                 "q_enc_key_pool_ size is not correct!");
//   ASSERT_ALWAYS(q_enc_key_pool_.at(0).size() == hidden_size_,
//                 "q_enc_key_pool_ hidden size is not correct!");
//   ASSERT_ALWAYS(k_enc_key_pool_.size() == enc_key_pool_size_,
//                 "k_enc_key_pool_ size is not correct!");
//   ASSERT_ALWAYS(k_enc_key_pool_.at(0).size() == hidden_size_,
//                 "k_enc_key_pool_ hidden size is not correct!");
//   ASSERT_ALWAYS(v_enc_key_pool_.size() == enc_key_pool_size_,
//                 "v_enc_key_pool_ size is not correct!");
//   ASSERT_ALWAYS(v_enc_key_pool_.at(0).size() == hidden_size_,
//                 "v_enc_key_pool_ hidden size is not correct!");
//   ASSERT_ALWAYS(o_enc_key_pool_.size() == enc_key_pool_size_,
//                 "o_enc_key_pool_ size is not correct!");
//   ASSERT_ALWAYS(o_enc_key_pool_.at(0).size() == hidden_size_,
//                 "o_enc_key_pool_ hidden size is not correct!");
//   ASSERT_ALWAYS(up_enc_key_pool_.size() == enc_key_pool_size_,
//                 "up_enc_key_pool_ size is not correct!");
//   ASSERT_ALWAYS(up_enc_key_pool_.at(0).size() == hidden_size_,
//                 "up_enc_key_pool_ hidden size is not correct!");
//   ASSERT_ALWAYS(gate_enc_key_pool_.size() == enc_key_pool_size_,
//                 "gate_enc_key_pool_ size is not correct!");
//   ASSERT_ALWAYS(gate_enc_key_pool_.at(0).size() == hidden_size_,
//                 "gate_enc_key_pool_ hidden size is not correct!");
//   ASSERT_ALWAYS(down_enc_key_pool_.size() == enc_key_pool_size_,
//                 "down_enc_key_pool_ size is not correct!");
//   ASSERT_ALWAYS(down_enc_key_pool_.at(0).size() == intermediate_size_,
//                 "down_enc_key_pool_ intermediate size is not correct!");

//   ASSERT_ALWAYS(q_dec_key_.size() == enc_key_pool_size_,
//                 "q_dec_key_ size is not correct!");
//   ASSERT_ALWAYS(q_dec_key_.at(0).size() == num_attention_heads_ * head_dim_,
//                 "q_dec_key_ hidden size is not correct!");
//   ASSERT_ALWAYS(k_dec_key_.size() == enc_key_pool_size_,
//                 "k_dec_key_ size is not correct!");
//   ASSERT_ALWAYS(k_dec_key_.at(0).size() == num_key_value_heads_ * head_dim_,
//                 "k_dec_key_ hidden size is not correct!");
//   ASSERT_ALWAYS(v_dec_key_.size() == enc_key_pool_size_,
//                 "v_dec_key_ size is not correct!");
//   ASSERT_ALWAYS(v_dec_key_.at(0).size() == num_key_value_heads_ * head_dim_,
//                 "v_dec_key_ hidden size is not correct!");
//   ASSERT_ALWAYS(o_dec_key_.size() == enc_key_pool_size_,
//                 "o_dec_key_ size is not correct!");
//   ASSERT_ALWAYS(o_dec_key_.at(0).size() == hidden_size_,
//                 "o_dec_key_ hidden size is not correct!");
//   ASSERT_ALWAYS(up_dec_key_.size() == enc_key_pool_size_,
//                 "up_dec_key_ size is not correct!");
//   ASSERT_ALWAYS(up_dec_key_.at(0).size() == intermediate_size_,
//                 "up_dec_key_ intermediate size is not correct!");
//   ASSERT_ALWAYS(gate_dec_key_.size() == enc_key_pool_size_,
//                 "gate_dec_key_ size is not correct!");
//   ASSERT_ALWAYS(gate_dec_key_.at(0).size() == intermediate_size_,
//                 "gate_dec_key_ intermediate size is not correct!");
//   ASSERT_ALWAYS(down_dec_key_.size() == enc_key_pool_size_,
//                 "down_dec_key_ size is not correct!");
//   ASSERT_ALWAYS(down_dec_key_.at(0).size() == hidden_size_,
//                 "down_dec_key_ hidden size is not correct!");
// #endif
}

void DecoderLayer::Reset() {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Reset() Enter"
            << std::endl;
#endif

  bsz_ = 0;
  present_token_len_ = 0;
  culmulative_token_len_ = 0;

  qk_x_row_shift_sum_.clear();
  qk_y_col_shift_sum_.clear();

  qk_x_mult_key_.clear();
  qk_x_add_key_.clear();
  qk_y_mult_key_.clear();
  qk_y_add_key_.clear();

  qk_dec_row_.clear();
  qk_dec_col_.clear();
  qk_dec_glob_.clear();

  pv_x_row_shift_sum_.clear();
  pv_y_col_shift_sum_.clear();

  pv_x_mult_key_.clear();
  pv_x_add_key_.clear();
  pv_y_mult_key_.clear();
  pv_y_add_key_.clear();

  pv_dec_row_.clear();
  pv_dec_col_.clear();
  pv_dec_glob_.clear();

  is_qk_key_generated_ = false;
  is_qk_dec_key_generated_ = false;
  is_qk_add_buffer_generated_ = false;
  is_qk_mult_buffer_generated_ = false;
  is_qk_unshift_buffer_generated_ = false;
  is_qk_shift_q_done_ = false;
  is_qk_shift_k_done_ = false;

  is_pv_key_generated_ = false;
  is_pv_dec_key_generated_ = false;
  is_pv_add_buffer_generated_ = false;
  is_pv_mult_buffer_generated_ = false;
  is_pv_unshift_buffer_generated_ = false;
  is_pv_shift_p_done_ = false;
  is_pv_shift_v_done_ = false;

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Reset() Exit" << std::endl;
#endif
}

void DecoderLayer::SetLinearWeightScales(float* weight_scales, int len,
                                         ProjectionType type) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] SetLinearWeightScales() Enter" << std::endl;
#endif
  switch (type) {
    case ProjectionType::kQ:
      q_weight_scales_.assign(weight_scales, weight_scales + len);
      break;
    case ProjectionType::kK:
      k_weight_scales_.assign(weight_scales, weight_scales + len);
      break;
    case ProjectionType::kV:
      v_weight_scales_.assign(weight_scales, weight_scales + len);
      break;
    case ProjectionType::kO:
      o_weight_scales_.assign(weight_scales, weight_scales + len);
      break;
    case ProjectionType::kGate:
      gate_weight_scales_.assign(weight_scales, weight_scales + len);
      break;
    case ProjectionType::kUp:
      up_weight_scales_.assign(weight_scales, weight_scales + len);
      break;
    case ProjectionType::kDown:
      down_weight_scales_.assign(weight_scales, weight_scales + len);
      break;
#if SGX_ENABLE == 0
    default:
      std::cout << "Invalid Projection Type!" << std::endl;
      exit(-1);
#endif
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] SetLinearWeightScales() Exit" << std::endl;
#endif
}

void DecoderLayer::SetRMSNormWeight(float* weight, float eps, int type) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] SetRMSNormWeight() Enter"
            << std::endl;
#endif

  if (type == 0) {
    input_layernorm_weights_.assign(weight, weight + hidden_size_);
    input_layernorm_eps_ = eps;
  } else if (type == 1) {
    post_attention_layernorm_weights_.assign(weight, weight + hidden_size_);
    post_attention_layernorm_eps_ = eps;
  } else {
#if SGX_ENABLE == 0
    std::cout << "Invalid RMSNorm Type!" << std::endl;
    exit(-1);
#endif
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] SetRMSNormWeight() Exit"
            << std::endl;
#endif
}

void DecoderLayer::SetEncKeyAndDecKey(
    int* src_enc_key_pool, std::vector<std::vector<int>>& dst_enc_key_pool,
    int* src_dec_key, std::vector<std::vector<int>>& dst_dec_key) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] Inner SetEncKeyAndDecKey() Enter" << std::endl;
#endif
  for (int i = 0; i < dst_enc_key_pool.size(); ++i) {
    for (int j = 0; j < dst_enc_key_pool[0].size(); ++j) {
      dst_enc_key_pool[i][j] =
          src_enc_key_pool[i * dst_enc_key_pool[0].size() + j];
    }
  }

  for (int i = 0; i < dst_dec_key.size(); ++i) {
    for (int j = 0; j < dst_dec_key[0].size(); ++j) {
      dst_dec_key[i][j] = src_dec_key[i * dst_dec_key[0].size() + j];
    }
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] Inner SetEncKeyAndDecKey() Exit" << std::endl;
#endif
}

void DecoderLayer::SetEncKeyAndDecKey(int* src_enc_key_pool, int* src_dec_key,
                                      ProjectionType type) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] Outer SetEncKeyAndDecKey() Enter" << std::endl;
#endif
  switch (type) {
    case ProjectionType::kQ:
      SetEncKeyAndDecKey(src_enc_key_pool, q_enc_key_pool_, src_dec_key,
                         q_dec_key_);
      break;
    case ProjectionType::kK:
      SetEncKeyAndDecKey(src_enc_key_pool, k_enc_key_pool_, src_dec_key,
                         k_dec_key_);
      break;
    case ProjectionType::kV:
      SetEncKeyAndDecKey(src_enc_key_pool, v_enc_key_pool_, src_dec_key,
                         v_dec_key_);
      break;
    case ProjectionType::kO:
      SetEncKeyAndDecKey(src_enc_key_pool, o_enc_key_pool_, src_dec_key,
                         o_dec_key_);
      break;
    case ProjectionType::kUp:
      SetEncKeyAndDecKey(src_enc_key_pool, up_enc_key_pool_, src_dec_key,
                         up_dec_key_);
      break;
    case ProjectionType::kGate:
      SetEncKeyAndDecKey(src_enc_key_pool, gate_enc_key_pool_, src_dec_key,
                         gate_dec_key_);
      break;
    case ProjectionType::kDown:
      SetEncKeyAndDecKey(src_enc_key_pool, down_enc_key_pool_, src_dec_key,
                         down_dec_key_);
      break;
#if SGX_ENABLE == 0
    default:
      std::cout << "Invalid Projection Type!" << std::endl;
      exit(-1);
#endif
  }
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] Outer SetEncKeyAndDecKey() Exit" << std::endl;
#endif
}

void DecoderLayer::QuantizeLinearActivation(std::shared_ptr<Tensor<int8_t>> out,
                                            std::shared_ptr<Tensor<float>> in,
                                            ProjectionType type) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] QuantizeLinearActivation() Enter" << std::endl;
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int N = shape.at(2);

  auto [quantized_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(in->data(), B, M, N);

  out->data() = std::move(quantized_act);

  switch (type) {
    case ProjectionType::kQ:
      q_act_scales_ = std::move(max_vals);
      break;
    case ProjectionType::kK:
      k_act_scales_ = std::move(max_vals);
      break;
    case ProjectionType::kV:
      v_act_scales_ = std::move(max_vals);
      break;
    case ProjectionType::kO:
      o_act_scales_ = std::move(max_vals);
      break;
    case ProjectionType::kGate:
      gate_act_scales_ = std::move(max_vals);
      break;
    case ProjectionType::kUp:
      up_act_scales_ = std::move(max_vals);
      break;
    case ProjectionType::kDown:
      down_act_scales_ = std::move(max_vals);
      break;
#if SGX_ENABLE == 0
    default:
      std::cout << "Invalid Projection Type!" << std::endl;
      exit(-1);
#endif
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] QuantizeLinearActivation() Exit" << std::endl;
#endif
}

void DecoderLayer::EncryptLinearActivation(std::shared_ptr<Tensor<int32_t>> out,
                                           std::shared_ptr<Tensor<int8_t>> in,
                                           ProjectionType type) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] EncryptLinearActivation() Enter" << std::endl;
#endif

  // For encrypted linear layers, matmul is done in int32
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int N = shape.at(2);

  std::vector<std::vector<int>>* enc_key_pool = nullptr;
  std::vector<int>* sampled_enc_key_index = nullptr;

  switch (type) {
    case ProjectionType::kQ:
      enc_key_pool = &q_enc_key_pool_;
      sampled_enc_key_index = &sampled_q_enc_key_index_;
      break;
    case ProjectionType::kK:
      enc_key_pool = &k_enc_key_pool_;
      sampled_enc_key_index = &sampled_k_enc_key_index_;
      break;
    case ProjectionType::kV:
      enc_key_pool = &v_enc_key_pool_;
      sampled_enc_key_index = &sampled_v_enc_key_index_;
      break;
    case ProjectionType::kO:
      enc_key_pool = &o_enc_key_pool_;
      sampled_enc_key_index = &sampled_o_enc_key_index_;
      break;
    case ProjectionType::kGate:
      enc_key_pool = &gate_enc_key_pool_;
      sampled_enc_key_index = &sampled_gate_enc_key_index_;
      break;
    case ProjectionType::kUp:
      enc_key_pool = &up_enc_key_pool_;
      sampled_enc_key_index = &sampled_up_enc_key_index_;
      break;
    case ProjectionType::kDown:
      enc_key_pool = &down_enc_key_pool_;
      sampled_enc_key_index = &sampled_down_enc_key_index_;
      break;
#if SGX_ENABLE == 0
    default:
      std::cout << "Invalid Projection Type!" << std::endl;
      exit(-1);
#endif
  }

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(sampled_enc_key_index->empty(),
                "sampled_enc_key_index is not empty!");
#endif
#if INTERNAL_TIME_MEASURE == 1 && SGX_ENABLE == 0
  auto start = std::chrono::steady_clock::now();
#endif

  // 3ms
  for (int i = 0; i < B * M; ++i) {
    sampled_enc_key_index->push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(sampled_enc_key_index->size() == B * M,
                "sampled_enc_key_index size is not correct!");
#endif

  // Optimization using Eigen for the innermost loop (across N)
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      int enc_key_index = sampled_enc_key_index->at(b * M + m);

      // Eigen Map for the input and output vectors across N dimension
      Eigen::Map<const Eigen::Matrix<int8_t, 1, Eigen::Dynamic>> in_vec(
          in->data().data() + b * M * N + m * N, N);
      Eigen::Map<Eigen::Matrix<int32_t, 1, Eigen::Dynamic>> out_vec(
          out->data().data() + b * M * N + m * N, N);

      // Eigen Map for the encryption key vector (reshaped as a row vector)
      Eigen::Map<const Eigen::Matrix<int, 1, Eigen::Dynamic>> enc_key_vec(
          enc_key_pool->at(enc_key_index).data(), 1, N);

      // Element-wise addition using Eigen
      out_vec = in_vec.cast<int32_t>() + enc_key_vec;

#if CHECK_SANITY == 1
      for (int n = 0; n < N; ++n) {
        int enc_test = enc_key_pool->at(enc_key_index).at(n) +
                       static_cast<int>(in->data()[b * M * N + m * N + n]);
        ASSERT_ALWAYS(out->data()[b * M * N + m * N + n] == enc_test,
                      "Encrypted value is not correct!");
      }
#endif
    }
  }

#if INTERNAL_TIME_MEASURE == 1 && SGX_ENABLE == 0
  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  std::cout << static_cast<int>(type) << ", EncryptLinearActivation Time: "
            << std::chrono::duration<double, std::milli>(diff).count() << " ms"
            << std::endl;
#endif

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] EncryptLinearActivation() Exit" << std::endl;
#endif
}

void DecoderLayer::DecryptLinearActivation(std::shared_ptr<Tensor<int32_t>> out,
                                           std::shared_ptr<Tensor<int32_t>> in,
                                           ProjectionType type) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] DecryptLinearActivation() Enter" << std::endl;
#endif

  // Note that output is int32
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int N = shape.at(2);

  std::vector<std::vector<int>>* dec_key = nullptr;
  std::vector<int>* sampled_enc_key_index = nullptr;

  switch (type) {
    case ProjectionType::kQ:
      dec_key = &q_dec_key_;
      sampled_enc_key_index = &sampled_q_enc_key_index_;
      break;
    case ProjectionType::kK:
      dec_key = &k_dec_key_;
      sampled_enc_key_index = &sampled_k_enc_key_index_;
      break;
    case ProjectionType::kV:
      dec_key = &v_dec_key_;
      sampled_enc_key_index = &sampled_v_enc_key_index_;
      break;
    case ProjectionType::kO:
      dec_key = &o_dec_key_;
      sampled_enc_key_index = &sampled_o_enc_key_index_;
      break;
    case ProjectionType::kGate:
      dec_key = &gate_dec_key_;
      sampled_enc_key_index = &sampled_gate_enc_key_index_;
      break;
    case ProjectionType::kUp:
      dec_key = &up_dec_key_;
      sampled_enc_key_index = &sampled_up_enc_key_index_;
      break;
    case ProjectionType::kDown:
      dec_key = &down_dec_key_;
      sampled_enc_key_index = &sampled_down_enc_key_index_;
      break;
#if SGX_ENABLE == 0
    default:
      std::cout << "Invalid Projection Type!" << std::endl;
      exit(-1);
#endif
  }

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(sampled_enc_key_index->size() == B * M,
                "sampled_enc_key_index size is not correct!");
#endif

#if INTERNAL_TIME_MEASURE == 1 && SGX_ENABLE == 0
  auto start = std::chrono::steady_clock::now();
#endif

  // Optimization using Eigen for the innermost loop (across N)
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      int enc_key_index = sampled_enc_key_index->at(b * M + m);

      // Map the input and output tensors for the N dimension
      Eigen::Map<const Eigen::Matrix<int32_t, 1, Eigen::Dynamic>> in_vec(
          in->data().data() + b * M * N + m * N, N);
      Eigen::Map<Eigen::Matrix<int32_t, 1, Eigen::Dynamic>> out_vec(
          out->data().data() + b * M * N + m * N, N);

      // Map the decryption key as a row vector
      Eigen::Map<const Eigen::Matrix<int, 1, Eigen::Dynamic>> dec_key_vec(
          dec_key->at(enc_key_index).data(), 1, N);

      // Perform element-wise subtraction using Eigen
      out_vec = in_vec - dec_key_vec;
    }
  }

#if INTERNAL_TIME_MEASURE == 1 && SGX_ENABLE == 0
  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  std::cout << static_cast<int>(type) << ", DecryptLinearActivation Time: "
            << std::chrono::duration<double, std::milli>(diff).count() << " ms"
            << std::endl;
#endif

  sampled_enc_key_index->clear();

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] DecryptLinearActivation() Exit" << std::endl;
#endif
}

void DecoderLayer::DequantizeLinearActivation(
    std::shared_ptr<Tensor<float>> out, std::shared_ptr<Tensor<int32_t>> in,
    ProjectionType type) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] DequantizeLinearActivation() Enter" << std::endl;
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int N = shape.at(2);

  std::vector<float>* weight_scales = nullptr;
  std::vector<float>* act_scales = nullptr;

  switch (type) {
    case ProjectionType::kQ:
      weight_scales = &q_weight_scales_;
      act_scales = &q_act_scales_;
      break;
    case ProjectionType::kK:
      weight_scales = &k_weight_scales_;
      act_scales = &k_act_scales_;
      break;
    case ProjectionType::kV:
      weight_scales = &v_weight_scales_;
      act_scales = &v_act_scales_;
      break;
    case ProjectionType::kO:
      weight_scales = &o_weight_scales_;
      act_scales = &o_act_scales_;
      break;
    case ProjectionType::kGate:
      weight_scales = &gate_weight_scales_;
      act_scales = &gate_act_scales_;
      break;
    case ProjectionType::kUp:
      weight_scales = &up_weight_scales_;
      act_scales = &up_act_scales_;
      break;
    case ProjectionType::kDown:
      weight_scales = &down_weight_scales_;
      act_scales = &down_act_scales_;
      break;
#if SGX_ENABLE == 0
    default:
      std::cout << "Invalid Projection Type!" << std::endl;
      exit(-1);
#endif
  }

  DequantizeActivationWPerChannelAPerChannel(
      out->data(), in->data(), *weight_scales, *act_scales, B * M, N);

  if (type == ProjectionType::kV) {
    out->Reshape({B, present_token_len_, num_key_value_heads_, head_dim_});
    out->Transpose(1, 2);
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] DequantizeLinearActivation() Exit" << std::endl;
#endif
}

void DecoderLayer::SetQKVOutputScales(float q_output_scale,
                                      float k_output_scale,
                                      float v_output_scale) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] SetQKVOutputScales() Enter"
            << std::endl;
#endif
  q_output_scale_ = q_output_scale;
  k_output_scale_ = k_output_scale;
  v_output_scale_ = v_output_scale;
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] SetQKVOutputScales() Exit"
            << std::endl;
#endif
}

void DecoderLayer::QuantizeQ_QK(std::shared_ptr<Tensor<int8_t>> out,
                                std::shared_ptr<Tensor<float>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] QuantizeQ_QK() Enter"
            << std::endl;
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;
  QuantizeActivationPerTensor(out->data(), in->data(), len, q_output_scale_);
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] QuantizeQ_QK() Exit"
            << std::endl;
#endif
}

void DecoderLayer::ShiftQ_QK(std::shared_ptr<Tensor<uint32_t>> out,
                             std::shared_ptr<Tensor<int8_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] ShiftQ_QK() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_shift_q_done_ == false, "ShiftQ_QK() is already done!");
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  // Initialize qk_x_row_shift_sum_ with dimensions [B][M][K]
  qk_x_row_shift_sum_ = std::vector<std::vector<std::vector<int>>>(
      B, std::vector<std::vector<int>>(M, std::vector<int>(K, 0)));

  // Iterate over B, M, K and compute the sum over N using Eigen
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        // Map the data for a single row (1xN) from in tensor as int8_t
        Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> row_in_map(
            &in->data()[b * M * K * N + m * K * N + k * N], N);

        // Sum the row after casting to int
        qk_x_row_shift_sum_[b][m][k] = row_in_map.cast<int>().sum();
      }
    }
  }

  // Apply the SHIFT_AMT to all elements using Eigen for the entire tensor
  int64_t len = static_cast<int64_t>(B) * M * K * N;

  // Create an Eigen Map for the output tensor (uint32_t) and input tensor (int8_t)
  Eigen::Map<Eigen::Array<uint32_t, Eigen::Dynamic, 1>> out_map(
      out->data().data(), len);
  Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> in_map(
      in->data().data(), len);

  // Explicitly cast the int8_t values to int before adding SHIFT_AMT, then cast to uint32_t
  out_map = (in_map.cast<int>() + SHIFT_AMT).cast<uint32_t>();

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] ShiftQ_QK() Exit"
            << std::endl;
#endif

  is_qk_shift_q_done_ = true;
}

void DecoderLayer::QuantizeK_QK(std::shared_ptr<Tensor<int8_t>> out,
                                std::shared_ptr<Tensor<float>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] QuantizeK_QK() Enter"
            << std::endl;
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;
  QuantizeActivationPerTensor(out->data(), in->data(), len, k_output_scale_);
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] QuantizeK_QK() Exit"
            << std::endl;
#endif
}

void DecoderLayer::ShiftK_QK(std::shared_ptr<Tensor<uint32_t>> out,
                             std::shared_ptr<Tensor<int8_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] ShiftK_QK() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_shift_k_done_ == false, "ShiftK_QK() is already done!");
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  if (qk_y_col_shift_sum_.empty()) {
    qk_y_col_shift_sum_ = std::vector<std::vector<std::vector<int>>>(
        B, std::vector<std::vector<int>>(M));
  }

  // Iterate over B, M, and K to compute the column sums over N using Eigen
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        // Map the data for a single row (1xN) from in tensor as int8_t
        Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> col_in_map(
            &in->data()[b * M * K * N + m * K * N + k * N], N);

        // Sum the row after casting to int
        qk_y_col_shift_sum_.at(b).at(m).push_back(col_in_map.cast<int>().sum());
      }
    }
  }

  // Apply the SHIFT_AMT to all elements using Eigen for the entire tensor
  int64_t len = static_cast<int64_t>(B) * M * K * N;

  // Create an Eigen Map for the output tensor (uint32_t) and input tensor (int8_t)
  Eigen::Map<Eigen::Array<uint32_t, Eigen::Dynamic, 1>> out_map(
      out->data().data(), len);
  Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> in_map(
      in->data().data(), len);

  // Explicitly cast the int8_t values to int before adding SHIFT_AMT, then cast to uint32_t
  out_map = (in_map.cast<int>() + SHIFT_AMT).cast<uint32_t>();

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] ShiftK_QK() Exit"
            << std::endl;
#endif

  is_qk_shift_k_done_ = true;
}

void DecoderLayer::Unshift_QK(std::shared_ptr<Tensor<int32_t>> out,
                              std::shared_ptr<Tensor<uint32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Unshift_QK() Enter"
            << std::endl;
#endif
#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_unshift_buffer_generated_ == true,
                "Unshift_QK() is not generated!");
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t total_elements = static_cast<int64_t>(B) * M * K * N;

  // Use Eigen's Map to treat the data as 1D arrays for vectorized operations
  Eigen::Map<Eigen::Array<int32_t, Eigen::Dynamic, 1>> out_map(
      out->data().data(), total_elements);
  Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> in_map(
      in->data().data(), total_elements);

  // Corrected qk_unshift_buffer to int32_t
  Eigen::Map<const Eigen::Array<int32_t, Eigen::Dynamic, 1>> unshift_buffer_map(
      qk_unshift_buffer.data(), total_elements);

  // Perform the vectorized subtraction (unshift operation)
  out_map = in_map.cast<int32_t>() - unshift_buffer_map;

  qk_unshift_buffer.clear();

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Unshift_QK() Exit"
            << std::endl;
#endif

  is_qk_unshift_buffer_generated_ = false;
  is_qk_shift_q_done_ = false;
  is_qk_shift_k_done_ = false;
}

void DecoderLayer::Dequantize_QK(std::shared_ptr<Tensor<float>> out,
                                 std::shared_ptr<Tensor<int32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Dequantize_QK() Enter"
            << std::endl;
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;

  float scale =
      q_output_scale_ * k_output_scale_ / sqrtf(head_dim_);  // Correct
  DequantizeActivationPerTensor(out->data(), in->data(), len, scale);
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Dequantize_QK() Exit"
            << std::endl;
#endif
}

void DecoderLayer::QuantizeP_PV(std::shared_ptr<Tensor<int8_t>> out,
                                std::shared_ptr<Tensor<float>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] QuantizeP_PV() Enter"
            << std::endl;
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;
  QuantizeActivationPerTensor(out->data(), in->data(), len, 1.0 / 127);
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] QuantizeP_PV() Exit"
            << std::endl;
#endif
}

void DecoderLayer::ShiftP_PV(std::shared_ptr<Tensor<uint32_t>> out,
                             std::shared_ptr<Tensor<int8_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] ShiftP_PV() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_shift_p_done_ == false, "ShiftP_PV() is already done!");
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  // Initialize pv_x_row_shift_sum_ to store row-wise sums
  pv_x_row_shift_sum_ = std::vector<std::vector<std::vector<int>>>(
      B, std::vector<std::vector<int>>(M, std::vector<int>(K, 0)));

  // Compute the row sums over N using Eigen
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        // Map the data for a single row (1xN) from in tensor as int8_t
        Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> row_in_map(
            &in->data()[b * M * K * N + m * K * N + k * N], N);

        // Sum the row after casting to int
        pv_x_row_shift_sum_[b][m][k] = row_in_map.cast<int>().sum();
      }
    }
  }

  // Apply the SHIFT_AMT to all elements using Eigen for the entire tensor
  int64_t len = static_cast<int64_t>(B) * M * K * N;

  // Create Eigen Maps for input and output
  Eigen::Map<Eigen::Array<uint32_t, Eigen::Dynamic, 1>> out_map(
      out->data().data(), len);
  Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> in_map(
      in->data().data(), len);

  // Shift the values and cast to uint32_t
  out_map = (in_map.cast<int>() + SHIFT_AMT).cast<uint32_t>();

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] ShiftP_PV() Exit"
            << std::endl;
#endif

  is_pv_shift_p_done_ = true;
}

void DecoderLayer::QuantizeV_PV(std::shared_ptr<Tensor<int8_t>> out,
                                std::shared_ptr<Tensor<float>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] QuantizeV_PV() Enter"
            << std::endl;
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;
  QuantizeActivationPerTensor(out->data(), in->data(), len, v_output_scale_);
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] QuantizeV_PV() Exit"
            << std::endl;
#endif
}

void DecoderLayer::ShiftV_PV(std::shared_ptr<Tensor<uint32_t>> out,
                             std::shared_ptr<Tensor<int8_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] ShiftV_PV() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_shift_v_done_ == false, "ShiftV_PV() is already done!");
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  // Initialize pv_y_col_shift_sum_ to store column-wise sums
  if (pv_y_col_shift_sum_.empty()) {
    pv_y_col_shift_sum_ = std::vector<std::vector<std::vector<int>>>(
        B, std::vector<std::vector<int>>(M, std::vector<int>(N, 0)));
  }

  // Compute the column sums over K using Eigen
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          // Accumulate the column sums for each element in in
          pv_y_col_shift_sum_[b][m][n] += static_cast<int>(
              in->data().at(b * M * K * N + m * K * N + k * N + n));
        }
      }
    }
  }

  // Apply the SHIFT_AMT to all elements using Eigen for the entire tensor
  int64_t len = static_cast<int64_t>(B) * M * K * N;

  // Create Eigen Maps for input and output
  Eigen::Map<Eigen::Array<uint32_t, Eigen::Dynamic, 1>> out_map(
      out->data().data(), len);
  Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> in_map(
      in->data().data(), len);

  // Shift the values and cast to uint32_t
  out_map = (in_map.cast<int>() + SHIFT_AMT).cast<uint32_t>();

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] ShiftV_PV() Exit"
            << std::endl;
#endif

  is_pv_shift_v_done_ = true;
}

void DecoderLayer::Unshift_PV(std::shared_ptr<Tensor<int32_t>> out,
                              std::shared_ptr<Tensor<uint32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Unshift_PV() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_unshift_buffer_generated_ == true,
                "Unshift_PV() is not generated!");
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t total_elements = static_cast<int64_t>(B) * M * K * N;

  // Use Eigen's Map to treat the data as 1D arrays for vectorized operations
  Eigen::Map<Eigen::Array<int32_t, Eigen::Dynamic, 1>> out_map(
      out->data().data(), total_elements);
  Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> in_map(
      in->data().data(), total_elements);

  // Corrected pv_unshift_buffer to int32_t
  Eigen::Map<const Eigen::Array<int32_t, Eigen::Dynamic, 1>> unshift_buffer_map(
      pv_unshift_buffer.data(), total_elements);

  // Perform the vectorized subtraction (unshift operation)
  out_map = in_map.cast<int32_t>() - unshift_buffer_map;

  pv_unshift_buffer.clear();

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Unshift_PV() Exit"
            << std::endl;
#endif

  is_pv_unshift_buffer_generated_ = false;
  is_pv_shift_p_done_ = false;
  is_pv_shift_v_done_ = false;
}

void DecoderLayer::Dequantize_PV(std::shared_ptr<Tensor<float>> out,
                                 std::shared_ptr<Tensor<int32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Dequantize_PV() Enter"
            << std::endl;
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;

  float scale = 1.0 / 127 * v_output_scale_;  // Correct
  DequantizeActivationPerTensor(out->data(), in->data(), len, scale);

  out->Transpose(1, 2);
  out->Reshape({B, K, M * N});
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Dequantize_PV() Exit"
            << std::endl;
#endif
}

void DecoderLayer::SetBatchSizeAndTokenLength(int bsz, int token_len) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] SetBatchSizeAndTokenLength() Enter" << std::endl;
#endif
  bsz_ = bsz;
  present_token_len_ = token_len;
  culmulative_token_len_ += token_len;
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] SetBatchSizeAndTokenLength() Exit" << std::endl;
#endif
}

void DecoderLayer::GenerateSecretKey_QK() {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateSecretKey_QK() Enter" << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_key_generated_ == false, "QK key is already generated!");
  ASSERT_ALWAYS(bsz_ > 0, "Batch size is not set!");
#endif

  // qk_x_mult_key: [bsz, num_attention_heads, q_len], and DO NOT ACCUMULATE, expected dim: [1, 32, 2048]
  // qk_y_mult_key: [bsz, num_key_value_heads, q_len], and ACCUMULATE, expected dim: [1, 8, 2048]
  // qk_x_add_key: [bsz, num_attention_heads, head_dim], and only initialized it does not change, expected dim: [1, 32, 128]
  // qk_y_add_key: [hsz, num_key_value_heads, head_dim], and only initialized it does not change, expected dim: [1, 8, 128]

  qk_x_mult_key_ =
      std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>>(
          bsz_, std::vector<std::vector<std::pair<uint32_t, int>>>(
                    num_attention_heads_));
  if (qk_y_mult_key_.empty()) {
    qk_y_mult_key_ =
        std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>>(
            bsz_, std::vector<std::vector<std::pair<uint32_t, int>>>(
                      num_key_value_heads_));

    qk_x_add_key_ = std::vector<std::vector<std::vector<uint32_t>>>(
        bsz_, std::vector<std::vector<uint32_t>>(num_attention_heads_));
    qk_y_add_key_ = std::vector<std::vector<std::vector<uint32_t>>>(
        bsz_, std::vector<std::vector<uint32_t>>(num_key_value_heads_));

    // Generate Add keys
    for (int b = 0; b < bsz_; ++b) {
      for (int m = 0; m < num_attention_heads_; ++m) {
        for (int n = 0; n < head_dim_; ++n) {
          qk_x_add_key_.at(b).at(m).push_back(GenerateAddKey());
        }
      }
      for (int m = 0; m < num_key_value_heads_; ++m) {
        for (int n = 0; n < head_dim_; ++n) {
          qk_y_add_key_.at(b).at(m).push_back(GenerateAddKey());
        }
      }
    }
  }

  // Generate Mult keys
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int n = 0; n < present_token_len_; ++n) {
        int index = GenerateCPRNG() % MULTKEY_POOL_SIZE;
        qk_x_mult_key_.at(b).at(m).emplace_back(x_mult_key_pool_[index], index);
      }
    }
    for (int m = 0; m < num_key_value_heads_; ++m) {
      for (int n = 0; n < present_token_len_; ++n) {
        int index = GenerateCPRNG() % MULTKEY_POOL_SIZE;
        qk_y_mult_key_.at(b).at(m).emplace_back(y_mult_key_pool_[index], index);
      }
    }
  }

  is_qk_key_generated_ = true;
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateSecretKey_QK() Exit" << std::endl;
#endif
}

void DecoderLayer::GenerateDecryptionKey_QK(
    std::shared_ptr<Tensor<uint32_t>> x, std::shared_ptr<Tensor<uint32_t>> y) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateDecryptionKey_QK() Enter" << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_key_generated_, "QK key is not generated!");
  ASSERT_ALWAYS(is_qk_dec_key_generated_ == false,
                "QK decryption key is already generated!");
  ASSERT_ALWAYS(bsz_ != 0, "Batch size is not set!");
#endif

  auto x_shape = x->shape();
  int X_B = x_shape.at(0);
  int X_M = x_shape.at(1);
  int X_K = x_shape.at(2);
  int X_N = x_shape.at(3);

  auto y_shape = y->shape();
  int Y_B = y_shape.at(0);
  int Y_M = y_shape.at(1);
  int Y_K = y_shape.at(2);
  int Y_N = y_shape.at(3);

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(bsz_ == X_B && bsz_ == Y_B,
                "Batch size is not matched between x and y!");
  ASSERT_ALWAYS(num_attention_heads_ == X_M && num_key_value_heads_ == Y_M,
                "num_attention_heads is not matched!");
#endif

  qk_dec_row_ = std::vector<std::vector<std::vector<uint32_t>>>(
      bsz_, std::vector<std::vector<uint32_t>>(num_attention_heads_));
  // D_COL reset at first only
  if (qk_dec_col_.empty()) {
    qk_dec_col_ = std::vector<std::vector<std::vector<uint32_t>>>(
        bsz_, std::vector<std::vector<uint32_t>>(num_attention_heads_));

    qk_dec_glob_ = std::vector<std::vector<uint32_t>>(
        bsz_, std::vector<uint32_t>(num_attention_heads_, 0));
    for (int b = 0; b < bsz_; ++b) {
      for (int m = 0; m < num_attention_heads_; ++m) {
        for (int n = 0; n < head_dim_; ++n) {
          qk_dec_glob_.at(b).at(m) +=
              qk_x_add_key_.at(b).at(m).at(n) *
              qk_y_add_key_.at(b).at(m / num_key_value_groups_).at(n);
        }
      }
    }
  }

  // X: [bsz, num_attention_heads, q_len, head_dim]
  // Y: [bsz, num_key_value_heads, q_len, head_dim]

  // d_row[b][m][k] = x_mult[b][m][k] * sum_n(y_add[b][m/4][n] * x[b][m][k][n]), expected dim: [1, 32, 2048]
  // d_col[b][m][k] = y_mult[b][m/4][k] * sum_n(x_add[b][m][n] * y[b][m/4][k][n]), notice m/4 = 0 : num_key_value_heads, expected dim: [1, 32, 2048]
  // d_glob[b][m] = sum_n(x_add[b][m][n] * y_add[b][m/4][n]), expected dim: [1, 32]

#if INTERNAL_TIME_MEASURE == 1 && SGX_ENABLE == 0
  auto start_dec_key_gen = std::chrono::steady_clock::now();
#endif

  int offset = culmulative_token_len_ - present_token_len_;
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int k = 0; k < present_token_len_; ++k) {
        uint32_t d_row_sum = 0;
        uint32_t d_col_sum = 0;
        for (int n = 0; n < head_dim_; ++n) {
          d_row_sum +=
              qk_y_add_key_.at(b).at(m / num_key_value_groups_).at(n) *
              x->data().at(b * num_attention_heads_ * present_token_len_ * head_dim_ + m * present_token_len_ * head_dim_ + k * head_dim_ + n);
          d_col_sum += qk_x_add_key_.at(b).at(m).at(n) *
                       y->data().at(b * num_key_value_heads_ * present_token_len_ * head_dim_ +
                                    (m / num_key_value_groups_) * present_token_len_ * head_dim_ +
                                    k * head_dim_ + n);
        }
        qk_dec_row_.at(b).at(m).push_back(
            d_row_sum * qk_x_mult_key_.at(b).at(m).at(k).first);
        qk_dec_col_.at(b).at(m).push_back(
            d_col_sum *
            qk_y_mult_key_.at(b).at(m / num_key_value_groups_).at(offset + k).first);
      }
    }
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateDecryptionKey_QK() Exit" << std::endl;
#endif
  is_qk_dec_key_generated_ = true;
}

void DecoderLayer::GenerateDecAddBuffer_QK() {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateDecAddBuffer_QK() Enter" << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_dec_key_generated_,
                "QK decryption key is not generated!");
#endif

  int64_t len = static_cast<int64_t>(bsz_) * num_attention_heads_ *
                present_token_len_ * culmulative_token_len_;
  qk_add_dec_key_buffer = std::vector<uint32_t>(len);

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      uint32_t qk_dec_glob_factor = qk_dec_glob_.at(b).at(m);
      for (int k = 0; k < present_token_len_; ++k) {
        uint32_t qk_dec_row_factor = qk_dec_row_.at(b).at(m).at(k);
        for (int n = 0; n < culmulative_token_len_; ++n) {
          int64_t index = b * num_attention_heads_ * present_token_len_ *
                              culmulative_token_len_ +
                          m * present_token_len_ * culmulative_token_len_ +
                          k * culmulative_token_len_ + n;
          qk_add_dec_key_buffer.at(index) = qk_dec_row_factor +
                                            qk_dec_glob_factor +
                                            qk_dec_col_.at(b).at(m).at(n);
        }
      }
    }
  }

  is_qk_add_buffer_generated_ = true;
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateDecAddBuffer_QK() Exit" << std::endl;
#endif
}

void DecoderLayer::GenerateDecMultBuffer_QK() {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateDecMultBuffer_QK() Enter" << std::endl;
#endif
#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_dec_key_generated_,
                "QK decryption key is not generated!");
#endif

  int64_t len = static_cast<int64_t>(bsz_) * num_attention_heads_ *
                present_token_len_ * culmulative_token_len_;
  qk_mult_dec_key_buffer = std::vector<uint32_t>(len);

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      int index_m_for_y = m / num_key_value_groups_;
      for (int k = 0; k < present_token_len_; ++k) {
        int qk_x_mult_key_index = qk_x_mult_key_.at(b).at(m).at(k).second;
        for (int n = 0; n < culmulative_token_len_; ++n) {
          int64_t index = b * num_attention_heads_ * present_token_len_ *
                              culmulative_token_len_ +
                          m * present_token_len_ * culmulative_token_len_ +
                          k * culmulative_token_len_ + n;

          qk_mult_dec_key_buffer.at(index) =
              precomputed_key_inv_.at(qk_x_mult_key_index)
                  .at(qk_y_mult_key_.at(b).at(index_m_for_y).at(n).second);
        }
      }
    }
  }

  is_qk_mult_buffer_generated_ = true;
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateDecMultBuffer_QK() Exit" << std::endl;
#endif
}

void DecoderLayer::GenerateUnshiftBuffer_QK() {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateUnshiftBuffer_QK() Enter" << std::endl;
#endif
#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_shift_q_done_ && is_qk_shift_k_done_,
                "QK shift is not done!");
#endif
  int64_t len = static_cast<int64_t>(bsz_) * num_attention_heads_ *
                present_token_len_ * culmulative_token_len_;

  // NOTE(jpyo0803): Filling up unshift buffer can be parallelized
  qk_unshift_buffer = std::vector<int>(len);

  int hss = head_dim_ * SHIFT_AMT * SHIFT_AMT;
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int k = 0; k < present_token_len_; ++k) {
        int row_unshift_factor = qk_x_row_shift_sum_.at(b).at(m).at(k);
        for (int n = 0; n < culmulative_token_len_; ++n) {
          int col_unshift_factor =
              qk_y_col_shift_sum_.at(b).at(m / num_key_value_groups_).at(n);
          qk_unshift_buffer.at(b * num_attention_heads_ * present_token_len_ *
                                   culmulative_token_len_ +
                               m * present_token_len_ * culmulative_token_len_ +
                               k * culmulative_token_len_ + n) =
              (row_unshift_factor + col_unshift_factor) * SHIFT_AMT + hss;
        }
      }
    }
  }

  is_qk_unshift_buffer_generated_ = true;
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateUnshiftBuffer_QK() Exit" << std::endl;
#endif
}

void DecoderLayer::EncryptX_QK(std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<uint32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] EncryptX_QK() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_key_generated_, "QK key is not generated!");
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  // #if CHECK_SANITY == 1
  //   ASSERT_ALWAYS(qk_permuted_index_.size() == N,
  //                 "Permutation size is not matched!");
  // #endif

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        uint32_t qk_x_mult_key_factor = qk_x_mult_key_.at(b).at(m).at(k).first;
        int64_t index_without_n = b * M * K * N + m * K * N + k * N;
        for (int n = 0; n < N; ++n) {
          // out->data().at(b * M * K * N + m * K * N + k * N + n) = in->data().at(b * M * K * N + m * K * N + k * N + n);
          out->data().at(index_without_n + qk_permuted_index_.at(n)) =
              in->data().at(index_without_n + n) * qk_x_mult_key_factor +
              qk_x_add_key_.at(b).at(m).at(n);
        }
      }
    }
  }
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] EncryptX_QK() Exit"
            << std::endl;
#endif
}

void DecoderLayer::EncryptY_QK(std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<uint32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] EncryptY_QK() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_key_generated_, "QK key is not generated!");
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int k_dim = qk_y_mult_key_.at(0).at(0).size();

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        uint32_t qk_y_mult_key_factor =
            qk_y_mult_key_.at(b).at(m).at(k_dim - K + k).first;
        int64_t index_without_n = b * M * K * N + m * K * N + k * N;
        for (int n = 0; n < N; ++n) {
          out->data().at(index_without_n + qk_permuted_index_.at(n)) =
              in->data().at(index_without_n + n) * qk_y_mult_key_factor +
              qk_y_add_key_.at(b).at(m).at(n);
          // out->data().at(b * M * K * N + m * K * N + k * N + n) = in->data().at(b * M * K * N + m * K * N + k * N + n);
          // NOTE(jpyo083): Dont forget that you use the valid mult key
        }
      }
    }
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] EncryptY_QK() Exit"
            << std::endl;
#endif
}

void DecoderLayer::Decrypt_QK(std::shared_ptr<Tensor<uint32_t>> out,
                              std::shared_ptr<Tensor<uint32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Decrypt_QK() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_qk_add_buffer_generated_ && is_qk_mult_buffer_generated_,
                "QK decryption buffers are not generated!");
#endif

#if INTERNAL_TIME_MEASURE == 1 && SGX_ENABLE == 0
  auto start = std::chrono::steady_clock::now();
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t total_elements = static_cast<int64_t>(B) * M * K * N;

  // Use Eigen to map the data as 1D arrays for vectorized operations
  Eigen::Map<Eigen::Array<uint32_t, Eigen::Dynamic, 1>> out_map(
      out->data().data(), total_elements);
  Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> in_map(
      in->data().data(), total_elements);
  Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> add_key_map(
      qk_add_dec_key_buffer.data(), total_elements);
  Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> mult_key_map(
      qk_mult_dec_key_buffer.data(), total_elements);

  // Perform the decryption operation using Eigen
  out_map = (in_map - add_key_map).array() * mult_key_map;
  // out_map = in_map;

  qk_add_dec_key_buffer.clear();
  qk_mult_dec_key_buffer.clear();

  // Reset key generation flags
  is_qk_key_generated_ = false;
  is_qk_dec_key_generated_ = false;
  is_qk_add_buffer_generated_ = false;
  is_qk_mult_buffer_generated_ = false;

#if INTERNAL_TIME_MEASURE == 1 && SGX_ENABLE == 0
  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  std::cout << "Decrypt_QK: "
            << std::chrono::duration<double, std::milli>(diff).count() << " ms"
            << std::endl;
#endif

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Decrypt_QK() Exit"
            << std::endl;
#endif
}

void DecoderLayer::GenerateSecretKey_PV() {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateSecretKey_PV() Enter" << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_key_generated_ == false, "PV key is already generated!");
  ASSERT_ALWAYS(bsz_ != 0, "Batch size is not set!");
#endif

  // pv_x_mult_key: [bsz, num_attention_heads, q_len], and DO NOT ACCUMULATE, expected dim: [1, 32, 2048]
  // pv_y_mult_key: [bsz, num_key_value_heads, head_dim], and DO NOT ACCUMULATE, expected dim: [1, 8, 128]
  // pv_x_add_key: [bsz, num_attention_heads, culmulative_len], and it keeps inceasing, expected dim: [1, 32, q_len]
  // pv_y_add_key: [hsz, num_key_value_heads, culmulative_len], and it keeps increasing, expected dim: [1, 8, q_len]

  pv_x_mult_key_ =
      std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>>(
          bsz_, std::vector<std::vector<std::pair<uint32_t, int>>>(
                    num_attention_heads_));

  // Generate X mult keys everytime
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int n = 0; n < present_token_len_; ++n) {
        int index = GenerateCPRNG() % MULTKEY_POOL_SIZE;
        pv_x_mult_key_.at(b).at(m).emplace_back(x_mult_key_pool_[index], index);
      }
    }
  }

  if (pv_y_mult_key_.empty()) {
    // Generate Y mult keys only once
    pv_y_mult_key_ =
        std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>>(
            bsz_, std::vector<std::vector<std::pair<uint32_t, int>>>(
                      num_key_value_heads_));

    for (int b = 0; b < bsz_; ++b) {
      for (int m = 0; m < num_key_value_heads_; ++m) {
        for (int n = 0; n < head_dim_; ++n) {
          int index = GenerateCPRNG() % MULTKEY_POOL_SIZE;
          pv_y_mult_key_.at(b).at(m).emplace_back(y_mult_key_pool_[index],
                                                  index);
        }
      }
    }

    pv_x_add_key_ = std::vector<std::vector<std::vector<uint32_t>>>(
        bsz_, std::vector<std::vector<uint32_t>>(num_attention_heads_));
    pv_y_add_key_ = std::vector<std::vector<std::vector<uint32_t>>>(
        bsz_, std::vector<std::vector<uint32_t>>(num_key_value_heads_));
  } 

  // Generate Add keys
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int n = 0; n < present_token_len_; ++n) {
        pv_x_add_key_.at(b).at(m).push_back(GenerateAddKey());
      }
    }
    for (int m = 0; m < num_key_value_heads_; ++m) {
      for (int n = 0; n < present_token_len_; ++n) {
        pv_y_add_key_.at(b).at(m).push_back(GenerateAddKey());
      }
    }
  }

  // if (pv_permuted_index_.empty()) {
  //   for (int i = 0; i < present_token_len_; ++i) {
  //     pv_permuted_index_.push_back(i);
  //   }
  // } else {
  //   pv_permuted_index_.push_back(culmulative_token_len_ - 1);
  // }

  // std::shuffle(pv_permuted_index_.begin(), pv_permuted_index_.end(), engine);

  is_pv_key_generated_ = true;

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateSecretKey_PV() Exit" << std::endl;
#endif
}

void DecoderLayer::GenerateDecryptionKey_PV(
    std::shared_ptr<Tensor<uint32_t>> x, std::shared_ptr<Tensor<uint32_t>> y) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateDecryptionKey_PV() Enter" << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(bsz_ != 0, "Batch size is not set!");
  ASSERT_ALWAYS(is_pv_key_generated_, "PV key is not generated!");
  ASSERT_ALWAYS(is_pv_dec_key_generated_ == false,
                "PV decryption key is already generated!");
#endif

  auto x_shape = x->shape();
  int X_B = x_shape.at(0);
  int X_M = x_shape.at(1);
  int X_K = x_shape.at(2);
  int X_N = x_shape.at(3);

  auto y_shape = y->shape();
  int Y_B = y_shape.at(0);
  int Y_M = y_shape.at(1);
  int Y_K = y_shape.at(2);
  int Y_N = y_shape.at(3);

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(bsz_ == X_B && bsz_ == Y_B && num_attention_heads_ == X_M &&
                    num_key_value_heads_ == Y_M,
                "Batch size or num_attention_heads is not matched!");
#endif

  pv_dec_row_ = std::vector<std::vector<std::vector<uint32_t>>>(
      bsz_, std::vector<std::vector<uint32_t>>(num_attention_heads_));

  // D_COL reset at first only
  if (pv_dec_col_.empty()) {
    pv_dec_col_ = std::vector<std::vector<std::vector<uint32_t>>>(
        bsz_, std::vector<std::vector<uint32_t>>(
                  num_attention_heads_, std::vector<uint32_t>(head_dim_, 0)));

    pv_dec_glob_ = std::vector<std::vector<uint32_t>>(
        bsz_, std::vector<uint32_t>(num_attention_heads_, 0));
  }

  int offset = culmulative_token_len_ - present_token_len_;

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      uint32_t d_glob_sum = 0;
      for (int k = 0; k < present_token_len_; ++k) {
        int corrected_k = offset + k;
        d_glob_sum +=
            pv_x_add_key_.at(b).at(m).at(corrected_k) *
            pv_y_add_key_.at(b).at(m / num_key_value_groups_).at(corrected_k);
      }
      pv_dec_glob_.at(b).at(m) += d_glob_sum;
    }
  }

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int k = 0; k < present_token_len_; ++k) {
        uint32_t d_row_sum = 0;
        for (int n = 0; n < culmulative_token_len_; ++n) {
          d_row_sum +=
              pv_y_add_key_.at(b).at(m / num_key_value_groups_).at(n) *
              x->data().at(b * num_attention_heads_ * present_token_len_ * culmulative_token_len_ 
                            + m * present_token_len_ * culmulative_token_len_ + k * culmulative_token_len_ + n);
        }
        pv_dec_row_.at(b).at(m).push_back(
            d_row_sum * pv_x_mult_key_.at(b).at(m).at(k).first);
      }

      int offset = culmulative_token_len_ - present_token_len_;
      std::vector<uint32_t> d_col_sum(head_dim_, 0);
      for (int k = 0; k < present_token_len_; ++k) {
        for (int n = 0; n < head_dim_; ++n) {
          d_col_sum.at(n) +=
              pv_x_add_key_.at(b).at(m).at(offset + k) *
              y->data().at(b * num_key_value_heads_ * present_token_len_ * head_dim_ +
                           (m / num_key_value_groups_) * present_token_len_ * head_dim_ + k * head_dim_ +
                           n);
        }
      }
      for (int n = 0; n < head_dim_; ++n) {
        pv_dec_col_.at(b).at(m).at(n) +=
            d_col_sum.at(n) *
            pv_y_mult_key_.at(b).at(m / num_key_value_groups_).at(n).first;
      }
    }
  }

  // Glob acculuates as add keys are added

  // X: [bsz, num_attention_heads, q_len, q_len]
  // Y: [bsz, num_key_value_heads, q_len, head_dim]

  // d_row[b][m][k] = x_mult[b][m][k] * sum_n(y_add[b][m/4][n] * x[b][m][k][n]), expected dim: [1, 32, 2048]
  is_pv_dec_key_generated_ = true;
  // it is generated
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_
            << "] GenerateDecryptionKey_PV() Exit" << std::endl;
#endif
}

void DecoderLayer::GenerateDecAddBuffer_PV() {
#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_dec_key_generated_,
                "PV decryption key is not generated!");
#endif

  int64_t len = static_cast<int64_t>(bsz_) * num_attention_heads_ *
                present_token_len_ * head_dim_;
  pv_add_dec_key_buffer = std::vector<uint32_t>(len);

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      uint32_t pv_dec_glob_factor = pv_dec_glob_.at(b).at(m);
      for (int k = 0; k < present_token_len_; ++k) {
        uint32_t pv_dec_row_factor = pv_dec_row_.at(b).at(m).at(k);
        for (int n = 0; n < head_dim_; ++n) {
          int64_t index =
              b * num_attention_heads_ * present_token_len_ * head_dim_ +
              m * present_token_len_ * head_dim_ + k * head_dim_ + n;
          pv_add_dec_key_buffer.at(index) = pv_dec_row_factor +
                                            pv_dec_glob_factor +
                                            pv_dec_col_.at(b).at(m).at(n);
        }
      }
    }
  }

  is_pv_add_buffer_generated_ = true;
}

void DecoderLayer::GenerateDecMultBuffer_PV() {
#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_dec_key_generated_,
                "PV decryption key is not generated!");
#endif

  int64_t len = static_cast<int64_t>(bsz_) * num_attention_heads_ *
                present_token_len_ * head_dim_;
  pv_mult_dec_key_buffer = std::vector<uint32_t>(len);

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      int index_m_for_y = m / num_key_value_groups_;
      for (int k = 0; k < present_token_len_; ++k) {
        int pv_x_mult_key_index = pv_x_mult_key_.at(b).at(m).at(k).second;
        for (int n = 0; n < head_dim_; ++n) {
          int64_t index =
              b * num_attention_heads_ * present_token_len_ * head_dim_ +
              m * present_token_len_ * head_dim_ + k * head_dim_ + n;
          pv_mult_dec_key_buffer.at(index) =
              precomputed_key_inv_.at(pv_x_mult_key_index)
                  .at(pv_y_mult_key_.at(b).at(index_m_for_y).at(n).second);
        }
      }
    }
  }

  is_pv_mult_buffer_generated_ = true;
}

void DecoderLayer::GenerateUnshiftBuffer_PV() {
#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_shift_p_done_ && is_pv_shift_v_done_,
                "PV shift is not done!");
#endif

  int64_t len = static_cast<int64_t>(bsz_) * num_attention_heads_ *
                present_token_len_ * head_dim_;

  // Filling up unshift buffer can be parallelized
  pv_unshift_buffer = std::vector<int>(len);

  int hss = culmulative_token_len_ * SHIFT_AMT * SHIFT_AMT;
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int k = 0; k < present_token_len_; ++k) {
        int row_unshift_factor = pv_x_row_shift_sum_.at(b).at(m).at(k);
        for (int n = 0; n < head_dim_; ++n) {
          int col_unshift_factor =
              pv_y_col_shift_sum_.at(b).at(m / num_key_value_groups_).at(n);
          pv_unshift_buffer.at(
              b * num_attention_heads_ * present_token_len_ * head_dim_ +
              m * present_token_len_ * head_dim_ + k * head_dim_ + n) =
              (row_unshift_factor + col_unshift_factor) * SHIFT_AMT + hss;
        }
      }
    }
  }

  is_pv_unshift_buffer_generated_ = true;
}

void DecoderLayer::EncryptX_PV(std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<uint32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] EncryptX_PV() Enter"
            << std::endl;
#endif
#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_key_generated_, "PV key is not generated!");
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        uint32_t pv_x_mult_key_factor = pv_x_mult_key_.at(b).at(m).at(k).first;
        for (int n = 0; n < N; ++n) {
          int64_t index = b * M * K * N + m * K * N + k * N + n;
          out->data().at(index) = in->data().at(index) * pv_x_mult_key_factor +
                                  pv_x_add_key_.at(b).at(m).at(n);
          // out->data().at(b * M * K * N + m * K * N + k * N + n) =
          //     in->data().at(b * M * K * N + m * K * N + k * N + n);
        }
      }
    }
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] EncryptX_PV() Exit"
            << std::endl;
#endif
}

void DecoderLayer::EncryptY_PV(std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<uint32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] EncryptY_PV() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_key_generated_, "PV key is not generated!");
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int k_dim = pv_y_add_key_.at(0).at(0).size();

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        uint32_t pv_y_add_key_factor =
            pv_y_add_key_.at(b).at(m).at(k_dim - K + k);
        int64_t index_without_n = b * M * K * N + m * K * N + k * N;
        for (int n = 0; n < N; ++n) {
          out->data().at(index_without_n + pv_permuted_index_.at(n)) =
              in->data().at(index_without_n + n) *
                  pv_y_mult_key_.at(b).at(m).at(n).first +
              pv_y_add_key_factor;
          // out->data().at(b * M * K * N + m * K * N + k * N + n) =
          //     in->data().at(b * M * K * N + m * K * N + k * N + n);
        }
      }
    }
  }

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] EncryptY_PV() Exit"
            << std::endl;
#endif
}

void DecoderLayer::Decrypt_PV(std::shared_ptr<Tensor<uint32_t>> out,
                              std::shared_ptr<Tensor<uint32_t>> in) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Decrypt_PV() Enter"
            << std::endl;
#endif

#if CHECK_SANITY == 1
  ASSERT_ALWAYS(is_pv_add_buffer_generated_ && is_pv_mult_buffer_generated_,
                "PV decryption buffers are not generated!");
#endif
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        int64_t index_without_n = b * M * K * N + m * K * N + k * N;
        for (int n = 0; n < N; ++n) {
          int64_t inorder_index = index_without_n + n;
          out->data().at(inorder_index) =
              (in->data().at(index_without_n + pv_permuted_index_.at(n)) -
               pv_add_dec_key_buffer.at(inorder_index)) *
              pv_mult_dec_key_buffer.at(inorder_index);
          // out->data().at(inorder_index) =
          //     in->data().at(inorder_index);
        }
      }
    }
  }

  pv_add_dec_key_buffer.clear();
  pv_mult_dec_key_buffer.clear();

  // Now we are done using it, actually generated means 'it has been updated' so that it is proper to use
  is_pv_dec_key_generated_ = false;
  is_pv_key_generated_ = false;
  is_pv_add_buffer_generated_ = false;
  is_pv_mult_buffer_generated_ = false;

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] Decrypt_PV() Exit"
            << std::endl;
#endif
}

void DecoderLayer::RMSNorm(std::shared_ptr<Tensor<float>> out,
                           std::shared_ptr<Tensor<float>> in, int type) {
#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] RMSNorm() Enter"
            << std::endl;
#endif

  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int N = shape.at(2);

  float eps = type == 0 ? input_layernorm_eps_ : post_attention_layernorm_eps_;
  const float* const weight = type == 0
                                  ? input_layernorm_weights_.data()
                                  : post_attention_layernorm_weights_.data();

  RMSNorm_Func(out->data().data(), in->data().data(), weight, B, M, N, eps);

#if DEBUG_PRINT == 1 && SGX_ENABLE == 0
  std::cout << "[Decoder Layer " << layer_idx_ << "] RMSNorm() Exit"
            << std::endl;
#endif
}

std::shared_ptr<Tensor<int32_t>> DecoderLayer::Matmul_CPU_QK(
    std::shared_ptr<Tensor<int8_t>> q, std::shared_ptr<Tensor<int8_t>> k) {
  // in1 and in2 are only the present token
  // update internal k cache and do matmul

  // K cache dimension after update: [bsz, num_key_value_heads, cumulative_token_len, head_dim]
  // = [bsz, num_key_value_heads, K, head_dim]

  // Update K cache

  if (k_cache_.empty()) {
    k_cache_ = std::vector<std::vector<std::vector<std::vector<int8_t>>>>(
        bsz_,
        std::vector<std::vector<std::vector<int8_t>>>(num_key_value_heads_));
  }

  const auto& k_data = k->data();

  int offset = culmulative_token_len_ - present_token_len_;
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_key_value_heads_; ++m) {
      for (int k = 0; k < present_token_len_; ++k) {
        k_cache_.at(b).at(m).push_back(std::vector<int8_t>());
        for (int n = 0; n < head_dim_; ++n) {
          k_cache_.at(b)
              .at(m)
              .at(offset + k)
              .push_back(k_data.at(
                  b * num_key_value_heads_ * present_token_len_ * head_dim_ +
                  m * present_token_len_ * head_dim_ + k * head_dim_ + n));
        }
      }
    }
  }

  // Repeat K cache
  auto repeated_k =
      RepeatKV(k_cache_, bsz_, num_key_value_heads_, culmulative_token_len_,
               head_dim_, num_key_value_groups_);

  // Matmul
  auto q_shape = q->shape();
  int B = q_shape.at(0);
  int M = q_shape.at(1);
  int K = q_shape.at(2);
  int N = q_shape.at(3);

  auto out = std::make_shared<Tensor<int32_t>>(
      std::vector<int>{B, M, present_token_len_, culmulative_token_len_});

  // Notice the dim K is the shared dim
  // a: [B, M, K]
  // b: [B, N, K]
  Matmul_Naive(out->data().data(), q->data().data(), repeated_k.data(), B * M,
               K, culmulative_token_len_, head_dim_, false);

  return out;
}

std::shared_ptr<Tensor<int32_t>> DecoderLayer::Matmul_CPU_PV(
    std::shared_ptr<Tensor<int8_t>> p, std::shared_ptr<Tensor<int8_t>> v) {

  if (v_cache_.empty()) {
    v_cache_ = std::vector<std::vector<std::vector<std::vector<int8_t>>>>(
        bsz_,
        std::vector<std::vector<std::vector<int8_t>>>(num_key_value_heads_));
  }

  // Update V cache
  const auto& v_data = v->data();
  int offset = culmulative_token_len_ - present_token_len_;
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_key_value_heads_; ++m) {
      for (int k = 0; k < present_token_len_; ++k) {
        v_cache_.at(b).at(m).push_back(std::vector<int8_t>());
        for (int n = 0; n < head_dim_; ++n) {
          v_cache_.at(b)
              .at(m)
              .at(offset + k)
              .push_back(v_data.at(
                  b * num_key_value_heads_ * present_token_len_ * head_dim_ +
                  m * present_token_len_ * head_dim_ + k * head_dim_ + n));
        }
      }
    }
  }

  // Repeat V cache
  auto repeated_v =
      RepeatKV(v_cache_, bsz_, num_key_value_heads_, culmulative_token_len_,
               head_dim_, num_key_value_groups_);

  // Matmul
  auto p_shape = p->shape();
  int B = p_shape.at(0);
  int M = p_shape.at(1);
  int K = p_shape.at(2);
  int N = p_shape.at(3);

  auto out = std::make_shared<Tensor<int32_t>>(
      std::vector<int>{B, M, present_token_len_, head_dim_});

  // Notice the dim K is the shared dim
  // a: [B, M, K]
  // b: [B, N, K]

  Matmul_Naive(out->data().data(), p->data().data(), repeated_v.data(), B * M,
               K, head_dim_, culmulative_token_len_, true);
  return out;
}

}  // namespace jpyo0803