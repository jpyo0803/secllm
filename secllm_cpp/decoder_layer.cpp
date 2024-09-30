#include "decoder_layer.h"

#include <iostream>

#include "aes_stream.h"

#include <chrono>
#include <cmath>
#include "func_utils.h"

#define MODULO (1LL << 32)
#define SHIFT_AMT 129

#define DEBUG 0
#define MULTKEY_POOL_SIZE 1024

namespace jpyo0803 {

DecoderLayer::DecoderLayer(int layer_idx, int hidden_size,
                           int intermediate_size, int max_position_embeddings,
                           int num_attention_heads, int num_key_value_heads,
                           int enc_key_pool_size)
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
      bsz_(0) {

  std::cout << "Decoder Layer " << layer_idx_ << " is created." << std::endl;

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
#if DEBUG == 1
      uint64_t test = (uint64_t)precomputed_key_inv_[i][j] * ab % MODULO;
      if (test != 1) {
        std::cout << "Key inverse is not correct!" << std::endl;
        exit(-1);
      }
#endif
    }
  }
}

void DecoderLayer::Reset() {
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
}

void DecoderLayer::SetEncKeyAndDecKey(
    int* src_enc_key_pool, std::vector<std::vector<int>>& dst_enc_key_pool,
    int* src_dec_key, std::vector<std::vector<int>>& dst_dec_key) {
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
}

void DecoderLayer::SetLinearWeightScales_Q(float* weight_scales, int len) {
  q_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_K(float* weight_scales, int len) {
  k_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_V(float* weight_scales, int len) {
  v_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_O(float* weight_scales, int len) {
  o_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_Up(float* weight_scales, int len) {
  up_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_Gate(float* weight_scales, int len) {
  gate_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_Down(float* weight_scales, int len) {
  down_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetEncKeyAndDecKey_Q(int* src_enc_key_pool,
                                        int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, q_enc_key_pool_, src_dec_key,
                     q_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_K(int* src_enc_key_pool,
                                        int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, k_enc_key_pool_, src_dec_key,
                     k_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_V(int* src_enc_key_pool,
                                        int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, v_enc_key_pool_, src_dec_key,
                     v_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_O(int* src_enc_key_pool,
                                        int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, o_enc_key_pool_, src_dec_key,
                     o_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_Up(int* src_enc_key_pool,
                                         int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, up_enc_key_pool_, src_dec_key,
                     up_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_Gate(int* src_enc_key_pool,
                                           int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, gate_enc_key_pool_, src_dec_key,
                     gate_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_Down(int* src_enc_key_pool,
                                           int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, down_enc_key_pool_, src_dec_key,
                     down_dec_key_);
}

void DecoderLayer::EncryptLinearActivation_Q(
    int* out, std::shared_ptr<Tensor<float>> q_tensor) {
  int B = q_tensor->shape()[0];
  int M = q_tensor->shape()[1];
  int N = q_tensor->shape()[2];

  // Apply Quantization
  auto [q_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(q_tensor->data(), B, M, N);
  q_act_scales_ = std::move(max_vals);

  if (!sampled_q_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_q_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)q_act[b * M * N + m * N + n] +
            q_enc_key_pool_[sampled_q_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_K(
    int* out, std::shared_ptr<Tensor<float>> k_tensor) {
  int B = k_tensor->shape()[0];
  int M = k_tensor->shape()[1];
  int N = k_tensor->shape()[2];

  auto [k_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(k_tensor->data(), B, M, N);
  k_act_scales_ = std::move(max_vals);

  if (!sampled_k_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_k_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)k_act[b * M * N + m * N + n] +
            k_enc_key_pool_[sampled_k_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_V(
    int* out, std::shared_ptr<Tensor<float>> v_tensor) {
  int B = v_tensor->shape()[0];
  int M = v_tensor->shape()[1];
  int N = v_tensor->shape()[2];

  auto [v_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(v_tensor->data(), B, M, N);
  v_act_scales_ = std::move(max_vals);

  if (!sampled_v_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_v_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)v_act[b * M * N + m * N + n] +
            v_enc_key_pool_[sampled_v_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_O(
    int* out, std::shared_ptr<Tensor<float>> o_tensor) {
  int B = o_tensor->shape()[0];
  int M = o_tensor->shape()[1];
  int N = o_tensor->shape()[2];

  auto [o_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(o_tensor->data(), B, M, N);
  o_act_scales_ = std::move(max_vals);

  if (!sampled_o_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }
  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_o_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)o_act[b * M * N + m * N + n] +
            o_enc_key_pool_[sampled_o_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_Up(
    int* out, std::shared_ptr<Tensor<float>> up_tensor) {
  int B = up_tensor->shape()[0];
  int M = up_tensor->shape()[1];
  int N = up_tensor->shape()[2];

  auto [up_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(up_tensor->data(), B, M, N);
  up_act_scales_ = std::move(max_vals);

  if (!sampled_up_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_up_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)up_act[b * M * N + m * N + n] +
            up_enc_key_pool_[sampled_up_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_Gate(
    int* out, std::shared_ptr<Tensor<float>> gate_tensor) {
  int B = gate_tensor->shape()[0];
  int M = gate_tensor->shape()[1];
  int N = gate_tensor->shape()[2];

  auto [gate_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(gate_tensor->data(), B, M, N);
  gate_act_scales_ = std::move(max_vals);

  if (!sampled_gate_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_gate_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)gate_act[b * M * N + m * N + n] +
            gate_enc_key_pool_[sampled_gate_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_Down(
    int* out, std::shared_ptr<Tensor<float>> down_tensor) {
  int B = down_tensor->shape()[0];
  int M = down_tensor->shape()[1];
  int N = down_tensor->shape()[2];

  auto [down_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(down_tensor->data(), B, M, N);
  down_act_scales_ = std::move(max_vals);

  if (!sampled_down_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_down_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)down_act[b * M * N + m * N + n] +
            down_enc_key_pool_[sampled_down_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::DecryptLinearActivation_Q(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  // Decrypt Result
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            q_dec_key_[sampled_q_enc_key_index_[b * M + m]][n];
      }
    }
  }
  sampled_q_enc_key_index_.clear();

  // Dequantize
  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, q_weight_scales_, q_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_K(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            k_dec_key_[sampled_k_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_k_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, k_weight_scales_, k_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_V(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            v_dec_key_[sampled_v_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_v_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, v_weight_scales_, v_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_O(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            o_dec_key_[sampled_o_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_o_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, o_weight_scales_, o_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_Up(
    std::shared_ptr<Tensor<float>> out, int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            up_dec_key_[sampled_up_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_up_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, up_weight_scales_, up_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_Gate(
    std::shared_ptr<Tensor<float>> out, int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            gate_dec_key_[sampled_gate_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_gate_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, gate_weight_scales_, gate_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_Down(
    std::shared_ptr<Tensor<float>> out, int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            down_dec_key_[sampled_down_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_down_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, down_weight_scales_, down_act_scales_, B * M, N);
}

void DecoderLayer::SetQKVOutputScales(float q_output_scale,
                                      float k_output_scale,
                                      float v_output_scale) {
  q_output_scale_ = q_output_scale;
  k_output_scale_ = k_output_scale;
  v_output_scale_ = v_output_scale;
}

void DecoderLayer::QuantizeAndShiftQ(std::shared_ptr<Tensor<uint32_t>> out,
                                     std::shared_ptr<Tensor<float>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;
  auto q_act = QuantizeActivationPerTensor(in->data(), len, q_output_scale_);

  qk_x_row_shift_sum_ = std::vector<std::vector<std::vector<int>>>(
      B, std::vector<std::vector<int>>(M));  // input is 4D
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        int sum = 0;
        for (int n = 0; n < N; ++n) {
          sum += (int)q_act.at(b * M * K * N + m * K * N + k * N + n);
        }
        qk_x_row_shift_sum_[b][m].push_back(sum);
      }
    }
  }

  for (int i = 0; i < len; ++i) {
    out->data().at(i) = (uint32_t)((int)q_act[i] + SHIFT_AMT);
  }
}

void DecoderLayer::QuantizeAndShiftK(std::shared_ptr<Tensor<uint32_t>> out,
                                     std::shared_ptr<Tensor<float>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;

  auto k_act = QuantizeActivationPerTensor(in->data(), len, k_output_scale_);

  if (qk_y_col_shift_sum_.empty()) {
    qk_y_col_shift_sum_ = std::vector<std::vector<std::vector<int>>>(
        B, std::vector<std::vector<int>>(M));  // input is 4D
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        int sum = 0;
        for (int n = 0; n < N; ++n) {
          sum += (int)k_act[b * M * K * N + m * K * N + k * N + n];
        }
        qk_y_col_shift_sum_[b][m].push_back(sum);
      }
    }
  }

  for (int i = 0; i < len; ++i) {
    out->data().at(i) = (uint32_t)((int)k_act[i] + SHIFT_AMT);
  }
}

void DecoderLayer::UnshiftAndDequantizeQK(
    std::shared_ptr<Tensor<float>> out, std::shared_ptr<Tensor<uint32_t>> in) {
  int B = out->shape().at(0);
  int M = out->shape().at(1);
  int K = out->shape().at(2);
  int N = out->shape().at(3);

  // std::cout << num_key_value_groups_ << std::endl;
  // Unshift
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      // std::cout << "m vs. other : " << m << " / " << m / num_key_value_groups_ << std::endl;
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          int unshift_factor =
              (qk_x_row_shift_sum_[b][m][k] +
               qk_y_col_shift_sum_[b][m / num_key_value_groups_][n]) *
                  SHIFT_AMT +
              head_dim_ * SHIFT_AMT * SHIFT_AMT;
          out->data().at(b * M * K * N + m * K * N + k * N + n) =
              (float)((int)in->data().at(b * M * K * N + m * K * N + k * N +
                                         n) -
                      unshift_factor);
        }
      }
    }
  }
  int len = static_cast<int64_t>(B) * M * K * N;

  // for (int i = 0; i < len; ++i) {
  //   out->data().at(i) = (float)((int)in->data().at(i));
  // }

  float scale =
      q_output_scale_ * k_output_scale_ / sqrtf(head_dim_);  // Correct
  DequantizeActivationPerTensor(out->data(), len, scale);
}

void DecoderLayer::QuantizeAndShiftP(std::shared_ptr<Tensor<uint32_t>> out,
                                     std::shared_ptr<Tensor<float>> in) {
  int B = in->shape().at(0);
  int M = in->shape().at(1);
  int K = in->shape().at(2);
  int N = in->shape().at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;

  auto p_act = QuantizeActivationPerTensor(in->data(), len, 1.0 / 127);

  pv_x_row_shift_sum_ = std::vector<std::vector<std::vector<int>>>(
      B, std::vector<std::vector<int>>(M));  // input is 4D

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        int sum = 0;
        for (int n = 0; n < N; ++n) {
          sum += (int)p_act[b * M * K * N + m * K * N + k * N + n];
        }
        pv_x_row_shift_sum_[b][m].push_back(sum);
      }
    }
  }

  for (int i = 0; i < len; ++i) {
    out->data().at(i) = (uint32_t)((int)p_act[i] + SHIFT_AMT);
    // out->data().at(i) = (uint32_t)((int)p_act[i]);
  }
}

void DecoderLayer::QuantizeAndShiftV(std::shared_ptr<Tensor<uint32_t>> out,
                                     std::shared_ptr<Tensor<float>> in) {
  int B = in->shape().at(0);
  int M = in->shape().at(1);
  int K = in->shape().at(2);
  int N = in->shape().at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;

  auto v_act = QuantizeActivationPerTensor(in->data(), len, v_output_scale_);

  if (pv_y_col_shift_sum_.empty()) {
    pv_y_col_shift_sum_ = std::vector<std::vector<std::vector<int>>>(
        B, std::vector<std::vector<int>>(
               M, std::vector<int>(N, 0)));  // input is 4D
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          pv_y_col_shift_sum_[b][m][n] +=
              (int)v_act[b * M * K * N + m * K * N + k * N + n];
        }
      }
    }
  }

  for (int i = 0; i < len; ++i) {
    out->data().at(i) = (uint32_t)((int)v_act[i] + SHIFT_AMT);
    // out->data().at(i) = (uint32_t)((int)v_act[i]);
  }
}

std::shared_ptr<Tensor<float>> DecoderLayer::UnshiftAndDequantizePV(
    std::shared_ptr<Tensor<uint32_t>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int64_t len = static_cast<int64_t>(B) * M * K * N;

  Tensor<float> tmp_tensor({B, M, K, N});
  // std::cout << "tmp tensor shape: ";
  // tmp_tensor.PrintShape();

  // Unshift

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      // std::cout << "m vs. other : " << m << " / " << m / num_key_value_groups_ << std::endl;
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          int unshift_factor =
              (pv_x_row_shift_sum_[b][m][k] +
               pv_y_col_shift_sum_[b][m / num_key_value_groups_][n]) *
                  SHIFT_AMT +
              culmulative_token_len_ * SHIFT_AMT * SHIFT_AMT;
          tmp_tensor.data().at(b * M * K * N + m * K * N + k * N + n) =
              (float)((int)in->data().at(b * M * K * N + m * K * N + k * N +
                                         n) -
                      unshift_factor);
        }
      }
    }
  }
  // tmp_tensor.PrintAsTorchStyle();
  // tmp_tensor.PrintCharacteristics();
  // std::cout << tmp_tensor.GetMean() << std::endl;
  // std::cout << tmp_tensor.PosDepSum() << std::endl;
  // exit(-1);
  // for (int i = 0; i < len; ++i) {
  //   tmp_tensor.data().at(i) = (float)((int)in->data().at(i));
  // }

  float scale = 1.0 / 127 * v_output_scale_;  // Correct
  DequantizeActivationPerTensor(tmp_tensor.data(), len, scale);

  auto tmp_tensor2 = tmp_tensor.Transpose(1, 2);
  // std::cout << "tmp tensor2 shape: ";
  // tmp_tensor2.PrintShape();
  auto tmp_tensor3 = tmp_tensor2.Reshape({B, K, M * N});
  // std::cout << "tmp tensor3 shape: ";
  // tmp_tensor3.PrintShape();
  return std::make_shared<Tensor<float>>(tmp_tensor3);
}

void DecoderLayer::SetBatchSizeAndTokenLength(int bsz, int token_len) {
  bsz_ = bsz;
  present_token_len_ = token_len;
  culmulative_token_len_ += token_len;
}

void DecoderLayer::GenerateSecretKey_QK() {
  if (bsz_ == 0) {
    std::cout << "Batch size is not set!" << std::endl;
    exit(-1);
  }

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

#if DEBUG == 1
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Generate Mult keys
  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int n = 0; n < present_token_len_; ++n) {
        int index = GenerateCPRNG() % MULTKEY_POOL_SIZE;
        qk_x_mult_key_.at(b).at(m).emplace_back(x_mult_key_pool_[index], index);
#if (DEBUG == 1)
        if (std::gcd(qk_x_mult_key_.at(b).at(m).at(n).first, MODULO) != 1) {
          std::cout << "Mult key is not coprime!" << std::endl;
          exit(-1);
        }
#endif
      }
    }
    for (int m = 0; m < num_key_value_heads_; ++m) {
      for (int n = 0; n < present_token_len_; ++n) {
        int index = GenerateCPRNG() % MULTKEY_POOL_SIZE;
        qk_y_mult_key_.at(b).at(m).emplace_back(y_mult_key_pool_[index], index);
#if (DEBUG == 1)
        if (std::gcd(qk_y_mult_key_.at(b).at(m).at(n).first, MODULO) != 1) {
          std::cout << "Mult key is not coprime!" << std::endl;
          exit(-1);
        }
#endif
      }
    }
  }

#if DEBUG == 1
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Mult key generation time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
#endif
  // print keys dimensions
  // std::cout << "qk_x_mult_key: " << qk_x_mult_key_.size() << " / " << qk_x_mult_key_[0].size() << " / " << qk_x_mult_key_[0][0].size() << std::endl;
  // std::cout << "qk_y_mult_key: " << qk_y_mult_key_.size() << " / " << qk_y_mult_key_[0].size() << " / " << qk_y_mult_key_[0][0].size() << std::endl;
  // std::cout << "qk_x_add_key: " << qk_x_add_key_.size() << " / " << qk_x_add_key_[0].size() << " / " << qk_x_add_key_[0][0].size() << std::endl;
  // std::cout << "qk_y_add_key: " << qk_y_add_key_.size() << " / " << qk_y_add_key_[0].size() << " / " << qk_y_add_key_[0][0].size() << std::endl;
  // exit(-1);
}

void DecoderLayer::GenerateDecryptionKey_QK(
    std::shared_ptr<Tensor<uint32_t>> x, std::shared_ptr<Tensor<uint32_t>> y) {
  if (bsz_ == 0) {
    std::cout << "Batch size is not set!" << std::endl;
    exit(-1);
  }

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

  if (bsz_ != X_B || bsz_ != Y_B || num_attention_heads_ != X_M ||
      num_key_value_heads_ != Y_M) {
    std::cout << "Batch size or num_attention_heads is not matched!"
              << std::endl;
    exit(-1);
  }

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
        for (int n = 0; n < X_N; ++n) {
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

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int k = 0; k < X_K; ++k) {
        uint32_t d_row_sum = 0;
        uint32_t d_col_sum = 0;
        for (int n = 0; n < X_N; ++n) {
          d_row_sum +=
              qk_y_add_key_.at(b).at(m / num_key_value_groups_).at(n) *
              x->data().at(b * X_M * X_K * X_N + m * X_K * X_N + k * X_N + n);
          d_col_sum += qk_x_add_key_.at(b).at(m).at(n) *
                       y->data().at(b * Y_M * Y_K * Y_N +
                                    (m / num_key_value_groups_) * Y_K * Y_N +
                                    k * Y_N + n);
        }
        qk_dec_row_.at(b).at(m).push_back(
            d_row_sum * qk_x_mult_key_.at(b).at(m).at(k).first);
        qk_dec_col_.at(b).at(m).push_back(
            d_col_sum *
            qk_y_mult_key_.at(b).at(m / num_key_value_groups_).at(k).first);
      }
    }
  }
}

void DecoderLayer::EncryptX_QK(std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<uint32_t>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          // out->data().at(b * M * K * N + m * K * N + k * N + n) = in->data().at(b * M * K * N + m * K * N + k * N + n);
          out->data().at(b * M * K * N + m * K * N + k * N + n) =
              in->data().at(b * M * K * N + m * K * N + k * N + n) *
                  qk_x_mult_key_.at(b).at(m).at(k).first +
              qk_x_add_key_.at(b).at(m).at(n);
        }
      }
    }
  }
}

void DecoderLayer::EncryptY_QK(std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<uint32_t>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int k_dim = qk_y_mult_key_.at(0).at(0).size();

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          out->data().at(b * M * K * N + m * K * N + k * N + n) =
              in->data().at(b * M * K * N + m * K * N + k * N + n) *
                  qk_y_mult_key_.at(b).at(m).at(k_dim - K + k).first +
              qk_y_add_key_.at(b).at(m).at(n);
          // out->data().at(b * M * K * N + m * K * N + k * N + n) = in->data().at(b * M * K * N + m * K * N + k * N + n);
          // NOTE(jpyo083): Dont forget that you use the valid mult key
        }
      }
    }
  }
}

void DecoderLayer::Decrypt_QK(std::shared_ptr<Tensor<uint32_t>> out,
                              std::shared_ptr<Tensor<uint32_t>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          uint32_t tmp = in->data().at(b * M * K * N + m * K * N + k * N + n) -
                         qk_dec_row_.at(b).at(m).at(k) -
                         qk_dec_col_.at(b).at(m).at(n) -
                         qk_dec_glob_.at(b).at(m);
          tmp *=
              precomputed_key_inv_.at(qk_x_mult_key_.at(b).at(m).at(k).second)
                  .at(qk_y_mult_key_.at(b)
                          .at(m / num_key_value_groups_)
                          .at(n)
                          .second);
          out->data().at(b * M * K * N + m * K * N + k * N + n) = tmp;
          // out->data().at(b * M * K * N + m * K * N + k * N + n) = in->data().at(b * M * K * N + m * K * N + k * N + n);
        }
      }
    }
  }
}

void DecoderLayer::GenerateSecretKey_PV() {
  if (bsz_ == 0) {
    std::cout << "Batch size is not set!" << std::endl;
    exit(-1);
  }

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
#if (DEBUG == 1)
        if (std::gcd(pv_x_mult_key_.at(b).at(m).at(n).first, MODULO) != 1) {
          std::cout << "Mult key is not coprime!" << std::endl;
          exit(-1);
        }
#endif
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
#if (DEBUG == 1)
          if (std::gcd(pv_y_mult_key_.at(b).at(m).at(n).first, MODULO) != 1) {
            std::cout << "Mult key is not coprime!" << std::endl;
            exit(-1);
          }
#endif
        }
      }
    }

    pv_x_add_key_ = std::vector<std::vector<std::vector<uint32_t>>>(
        bsz_, std::vector<std::vector<uint32_t>>(num_attention_heads_));
    pv_y_add_key_ = std::vector<std::vector<std::vector<uint32_t>>>(
        bsz_, std::vector<std::vector<uint32_t>>(num_key_value_heads_));

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
  } else {
    for (int b = 0; b < bsz_; ++b) {
      for (int m = 0; m < num_attention_heads_; ++m) {
        pv_x_add_key_.at(b).at(m).push_back(GenerateAddKey());
      }

      for (int m = 0; m < num_key_value_heads_; ++m) {
        pv_y_add_key_.at(b).at(m).push_back(GenerateAddKey());
      }
    }
  }

  // print keys dimensions
  // std::cout << "pv_x_mult_key: " << pv_x_mult_key_.size() << " / " << pv_x_mult_key_[0].size() << " / " << pv_x_mult_key_[0][0].size() << std::endl;
  // std::cout << "pv_y_mult_key: " << pv_y_mult_key_.size() << " / " << pv_y_mult_key_[0].size() << " / " << pv_y_mult_key_[0][0].size() << std::endl;
  // std::cout << "pv_x_add_key: " << pv_x_add_key_.size() << " / " << pv_x_add_key_[0].size() << " / " << pv_x_add_key_[0][0].size() << std::endl;
  // std::cout << "pv_y_add_key: " << pv_y_add_key_.size() << " / " << pv_y_add_key_[0].size() << " / " << pv_y_add_key_[0][0].size() << std::endl;
  // exit(-1);
}

void DecoderLayer::GenerateDecryptionKey_PV(
    std::shared_ptr<Tensor<uint32_t>> x, std::shared_ptr<Tensor<uint32_t>> y) {
  if (bsz_ == 0) {
    std::cout << "Batch size is not set!" << std::endl;
    exit(-1);
  }

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

  if (bsz_ != X_B || bsz_ != Y_B || num_attention_heads_ != X_M ||
      num_key_value_heads_ != Y_M || X_N != Y_K) {
    std::cout << "Batch size or num_attention_heads is not matched!"
              << std::endl;
    exit(-1);
  }

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

  int past_token_len = culmulative_token_len_ - present_token_len_;

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      uint32_t d_glob_sum = 0;
      for (int k = 0; k < X_K; ++k) {
        int corrected_k = past_token_len + k;
        d_glob_sum +=
            pv_x_add_key_.at(b).at(m).at(corrected_k) *
            pv_y_add_key_.at(b).at(m / num_key_value_groups_).at(corrected_k);
      }
      pv_dec_glob_.at(b).at(m) += d_glob_sum;
    }
  }

  for (int b = 0; b < bsz_; ++b) {
    for (int m = 0; m < num_attention_heads_; ++m) {
      for (int k = 0; k < X_K; ++k) {
        uint32_t d_row_sum = 0;
        for (int n = 0; n < X_N; ++n) {
          d_row_sum +=
              pv_y_add_key_.at(b).at(m / num_key_value_groups_).at(n) *
              x->data().at(b * X_M * X_K * X_N + m * X_K * X_N + k * X_N + n);
        }
        pv_dec_row_.at(b).at(m).push_back(
            d_row_sum * pv_x_mult_key_.at(b).at(m).at(k).first);
      }

      std::vector<uint32_t> d_col_sum(Y_N, 0);
      for (int k = 0; k < Y_K; ++k) {
        for (int n = 0; n < Y_N; ++n) {
          d_col_sum.at(n) +=
              pv_x_add_key_.at(b).at(m).at(k) *
              y->data().at(b * Y_M * Y_K * Y_N +
                           (m / num_key_value_groups_) * Y_K * Y_N + k * Y_N +
                           n);
        }
      }
      for (int n = 0; n < Y_N; ++n) {
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

  // Print keys
  // std::cout << "pv_dec_row: " << pv_dec_row_.size() << " / " << pv_dec_row_[0].size() << " / " << pv_dec_row_[0][0].size() << std::endl;
  // std::cout << "pv_dec_col: " << pv_dec_col_.size() << " / " << pv_dec_col_[0].size() << " / " << pv_dec_col_[0][0].size() << std::endl;
  // std::cout << "pv_dec_glob: " << pv_dec_glob_.size() << " / " << pv_dec_glob_[0].size() << std::endl;
  // exit(-1);
}

void DecoderLayer::EncryptX_PV(std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<uint32_t>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          out->data().at(b * M * K * N + m * K * N + k * N + n) =
              in->data().at(b * M * K * N + m * K * N + k * N + n) *
                  pv_x_mult_key_.at(b).at(m).at(k).first +
              pv_x_add_key_.at(b).at(m).at(n);
          // out->data().at(b * M * K * N + m * K * N + k * N + n) =
          //     in->data().at(b * M * K * N + m * K * N + k * N + n);
        }
      }
    }
  }
}

void DecoderLayer::EncryptY_PV(std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<uint32_t>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  int k_dim = pv_y_add_key_.at(0).at(0).size();

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          out->data().at(b * M * K * N + m * K * N + k * N + n) =
              in->data().at(b * M * K * N + m * K * N + k * N + n) *
                  pv_y_mult_key_.at(b).at(m).at(n).first +
              pv_y_add_key_.at(b).at(m).at(k_dim - K + k);
          // out->data().at(b * M * K * N + m * K * N + k * N + n) =
          //     in->data().at(b * M * K * N + m * K * N + k * N + n);
        }
      }
    }
  }
}

void DecoderLayer::Decrypt_PV(std::shared_ptr<Tensor<uint32_t>> out,
                              std::shared_ptr<Tensor<uint32_t>> in) {
  auto shape = in->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          uint32_t tmp = in->data().at(b * M * K * N + m * K * N + k * N + n) -
                         pv_dec_row_.at(b).at(m).at(k) -
                         pv_dec_col_.at(b).at(m).at(n) -
                         pv_dec_glob_.at(b).at(m);
          tmp *=
              precomputed_key_inv_.at(pv_x_mult_key_.at(b).at(m).at(k).second)
                  .at(pv_y_mult_key_.at(b)
                          .at(m / num_key_value_groups_)
                          .at(n)
                          .second);
          out->data().at(b * M * K * N + m * K * N + k * N + n) = tmp;
          // out->data().at(b * M * K * N + m * K * N + k * N + n) = in->data().at(b * M * K * N + m * K * N + k * N + n);
        }
      }
    }
  }
}

}  // namespace jpyo0803