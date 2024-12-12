#include "secllm_sgx.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "aes_stream.h"

#include "func_utils.h"

#include "Eigen/Dense"
#include "macro.h"

namespace {
jpyo0803::SecLLM* secllm_ptr = nullptr;

constexpr bool enable_linear_encryption = true;
constexpr bool enable_atten_encryption = true;
}  // namespace

namespace jpyo0803 {

SecLLM::SecLLM(int hidden_size, int intermediate_size,
               int max_position_embeddings, int num_attention_heads,
               int num_hidden_layers, int num_key_value_heads,
               int enc_key_pool_size)
    : num_hidden_layers_(num_hidden_layers) {
#if SGX_ENABLE == 0
  std::cout << "SecLLM is created with " << num_hidden_layers_ << " layers."
            << std::endl;
#endif
  book_keeper_float_ =
      std::make_unique<BookKeeper<Tensor<float>>>(num_hidden_layers_ * 110 * 3);

  book_keeper_int32_ = std::make_unique<BookKeeper<Tensor<int32_t>>>(
      num_hidden_layers_ * 110 * 3);

  book_keeper_uint32_ = std::make_unique<BookKeeper<Tensor<uint32_t>>>(
      num_hidden_layers_ * 110 * 3);

  book_keeper_int8_ = std::make_unique<BookKeeper<Tensor<int8_t>>>(
      num_hidden_layers_ * 110 * 3);

  /*
    We have total 91 operations [0, 90]
    An operation has at most 3 inputs
    We have 'num_layers' layers
    The number of spaces we need is (num_layers * 100 * 3) for simplicity
  */

  /*
    Book keeping rule, if a target operation is 41th operation (0-indexed) in 5th layer (0-indexed),
    its first input's location is (5 * 300 + 100 * 0 + 41)
    its second input's location is (5 * 300 + 100 * 1 + 41)
    its third input's location is (5 * 300 + 100 * 2 + 41)
    In generalization, x-th input of y-th operation in z-th layer
    is located at (z * 300 + 100 * x + y)
  */

  decoder_layers_ = std::make_unique<std::vector<DecoderLayer>>();
  for (int i = 0; i < num_hidden_layers_; ++i) {
    decoder_layers_->emplace_back(
        i, hidden_size, intermediate_size, max_position_embeddings,
        num_attention_heads, num_key_value_heads, enc_key_pool_size,
        enable_linear_encryption, enable_atten_encryption);
  }
}

void SecLLM::Reset() {
  for (int i = 0; i < num_hidden_layers_; ++i) {
    decoder_layers_->at(i).Reset();
  }
}

std::shared_ptr<Tensor<float>> SecLLM::BookKeeperLoad_Float(int loc) {
  return book_keeper_float_->Retrieve(loc);
}

void SecLLM::BookKeeperStore_Float(std::vector<int> locs,
                                   std::shared_ptr<Tensor<float>>& data_ptr) {
  book_keeper_float_->Keep(locs, data_ptr);
}

std::shared_ptr<Tensor<int32_t>> SecLLM::BookKeeperLoad_Int32(int loc) {
  return book_keeper_int32_->Retrieve(loc);
}

void SecLLM::BookKeeperStore_Int32(std::vector<int> locs,
                                   std::shared_ptr<Tensor<int32_t>>& data_ptr) {
  book_keeper_int32_->Keep(locs, data_ptr);
}

std::shared_ptr<Tensor<uint32_t>> SecLLM::BookKeeperLoad_Uint32(int loc) {
  return book_keeper_uint32_->Retrieve(loc);
}

void SecLLM::BookKeeperStore_Uint32(
    std::vector<int> locs, std::shared_ptr<Tensor<uint32_t>>& data_ptr) {
  book_keeper_uint32_->Keep(locs, data_ptr);
}

std::shared_ptr<Tensor<int8_t>> SecLLM::BookKeeperLoad_Int8(int loc) {
  return book_keeper_int8_->Retrieve(loc);
}

void SecLLM::BookKeeperStore_Int8(std::vector<int> locs,
                                  std::shared_ptr<Tensor<int8_t>>& data_ptr) {
  book_keeper_int8_->Keep(locs, data_ptr);
}

bool SecLLM::BookKeeperIsAvailable_Float(int loc) {
  return book_keeper_float_->IsAvailable(loc);
}

bool SecLLM::BookKeeperIsAvailable_Int32(int loc) {
  return book_keeper_int32_->IsAvailable(loc);
}

bool SecLLM::BookKeeperIsAvailable_Uint32(int loc) {
  return book_keeper_uint32_->IsAvailable(loc);
}

bool SecLLM::BookKeeperIsAvailable_Int8(int loc) {
  return book_keeper_int8_->IsAvailable(loc);
}

int SecLLM::BookKeeperGetShapeLength_Float(int loc) {
  return book_keeper_float_->RetrieveWithoutReset(loc)->shape().size();
}

int SecLLM::BookKeeperGetShapeLength_Int32(int loc) {
  return book_keeper_int32_->RetrieveWithoutReset(loc)->shape().size();
}

int SecLLM::BookKeeperGetShapeLength_Uint32(int loc) {
  return book_keeper_uint32_->RetrieveWithoutReset(loc)->shape().size();
}

int SecLLM::BookKeeperGetShapeLength_Int8(int loc) {
  return book_keeper_int8_->RetrieveWithoutReset(loc)->shape().size();
}

void SecLLM::BookKeeperGetShape_Float(int loc, int* out) {
  auto shape = book_keeper_float_->RetrieveWithoutReset(loc)->shape();
  for (int i = 0; i < shape.size(); ++i) {
    out[i] = shape[i];
  }
}

void SecLLM::BookKeeperGetShape_Int32(int loc, int* out) {
  auto shape = book_keeper_int32_->RetrieveWithoutReset(loc)->shape();
  for (int i = 0; i < shape.size(); ++i) {
    out[i] = shape[i];
  }
}

void SecLLM::BookKeeperGetShape_Uint32(int loc, int* out) {
  auto shape = book_keeper_uint32_->RetrieveWithoutReset(loc)->shape();
  for (int i = 0; i < shape.size(); ++i) {
    out[i] = shape[i];
  }
}

void SecLLM::BookKeeperGetShape_Int8(int loc, int* out) {
  auto shape = book_keeper_int8_->RetrieveWithoutReset(loc)->shape();
  for (int i = 0; i < shape.size(); ++i) {
    out[i] = shape[i];
  }
}

std::shared_ptr<Tensor<float>> SecLLM::BookKeeperLoadWithoutReset_Float(
    int loc) {
  return book_keeper_float_->RetrieveWithoutReset(loc);
}

std::shared_ptr<Tensor<int32_t>> SecLLM::BookKeeperLoadWithoutReset_Int32(
    int loc) {
  return book_keeper_int32_->RetrieveWithoutReset(loc);
}

std::shared_ptr<Tensor<uint32_t>> SecLLM::BookKeeperLoadWithoutReset_Uint32(
    int loc) {
  return book_keeper_uint32_->RetrieveWithoutReset(loc);
}

std::shared_ptr<Tensor<int8_t>> SecLLM::BookKeeperLoadWithoutReset_Int8(
    int loc) {
  return book_keeper_int8_->RetrieveWithoutReset(loc);
}

void SecLLM::SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool, int* dec_key,
                                ProjectionType type) {
  decoder_layers_->at(layer_idx).SetEncKeyAndDecKey(enc_key_pool, dec_key,
                                                    type);
}

void SecLLM::SetLinearWeightScales(int layer_idx, float* weight_scale, int len,
                                   ProjectionType type) {
  decoder_layers_->at(layer_idx).SetLinearWeightScales(weight_scale, len, type);
}

void SecLLM::SetRMSNormWeight(int layer_idx, float* weight, float eps,
                              int type) {
  decoder_layers_->at(layer_idx).SetRMSNormWeight(weight, eps, type);
}

void SecLLM::RMSNorm(int layer_idx, std::shared_ptr<Tensor<float>> out,
                     std::shared_ptr<Tensor<float>> in, int type) {
  decoder_layers_->at(layer_idx).RMSNorm(out, in, type);
}

void SecLLM::QuantizeLinearActivation(int layer_idx,
                                      std::shared_ptr<Tensor<int8_t>> out,
                                      std::shared_ptr<Tensor<float>> in,
                                      ProjectionType type) {
  decoder_layers_->at(layer_idx).QuantizeLinearActivation(out, in, type);
}

void SecLLM::EncryptLinearActivation(int layer_idx,
                                     std::shared_ptr<Tensor<int32_t>> out,
                                     std::shared_ptr<Tensor<int8_t>> in,
                                     ProjectionType type) {
  decoder_layers_->at(layer_idx).EncryptLinearActivation(out, in, type);
}

void SecLLM::DecryptLinearActivation(int layer_idx,
                                     std::shared_ptr<Tensor<int32_t>> out,
                                     std::shared_ptr<Tensor<int32_t>> in,
                                     ProjectionType type) {
  decoder_layers_->at(layer_idx).DecryptLinearActivation(out, in, type);
}

void SecLLM::DequantizeLinearActivation(int layer_idx,
                                        std::shared_ptr<Tensor<float>> out,
                                        std::shared_ptr<Tensor<int32_t>> in,
                                        ProjectionType type) {
  decoder_layers_->at(layer_idx).DequantizeLinearActivation(out, in, type);
}

void SecLLM::SetQKVOutputScales(int layer_idx, float q_output_scale,
                                float k_output_scale, float v_output_scale) {
  decoder_layers_->at(layer_idx).SetQKVOutputScales(
      q_output_scale, k_output_scale, v_output_scale);
}

void SecLLM::QuantizeQ_QK(int layer_idx, std::shared_ptr<Tensor<int8_t>> out,
                          std::shared_ptr<Tensor<float>> in) {
  decoder_layers_->at(layer_idx).QuantizeQ_QK(out, in);
}

void SecLLM::ShiftQ_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                       std::shared_ptr<Tensor<int8_t>> in) {
  decoder_layers_->at(layer_idx).ShiftQ_QK(out, in);
}

void SecLLM::QuantizeK_QK(int layer_idx, std::shared_ptr<Tensor<int8_t>> out,
                          std::shared_ptr<Tensor<float>> in) {
  decoder_layers_->at(layer_idx).QuantizeK_QK(out, in);
}

void SecLLM::ShiftK_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                       std::shared_ptr<Tensor<int8_t>> in) {
  decoder_layers_->at(layer_idx).ShiftK_QK(out, in);
}

void SecLLM::Unshift_QK(int layer_idx, std::shared_ptr<Tensor<int32_t>> out,
                        std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).Unshift_QK(out, in);
}

void SecLLM::Dequantize_QK(int layer_idx, std::shared_ptr<Tensor<float>> out,
                           std::shared_ptr<Tensor<int32_t>> in) {
  decoder_layers_->at(layer_idx).Dequantize_QK(out, in);
}

void SecLLM::QuantizeP_PV(int layer_idx, std::shared_ptr<Tensor<int8_t>> out,
                          std::shared_ptr<Tensor<float>> in) {
  decoder_layers_->at(layer_idx).QuantizeP_PV(out, in);
}

void SecLLM::ShiftP_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                       std::shared_ptr<Tensor<int8_t>> in) {
  decoder_layers_->at(layer_idx).ShiftP_PV(out, in);
}

void SecLLM::QuantizeV_PV(int layer_idx, std::shared_ptr<Tensor<int8_t>> out,
                          std::shared_ptr<Tensor<float>> in) {
  decoder_layers_->at(layer_idx).QuantizeV_PV(out, in);
}

void SecLLM::ShiftV_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                       std::shared_ptr<Tensor<int8_t>> in) {
  decoder_layers_->at(layer_idx).ShiftV_PV(out, in);
}

void SecLLM::Unshift_PV(int layer_idx, std::shared_ptr<Tensor<int32_t>> out,
                        std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).Unshift_PV(out, in);
}

void SecLLM::Dequantize_PV(int layer_idx, std::shared_ptr<Tensor<float>> out,
                           std::shared_ptr<Tensor<int32_t>> in) {
  decoder_layers_->at(layer_idx).Dequantize_PV(out, in);
}

std::shared_ptr<Tensor<int32_t>> SecLLM::Matmul_CPU_QK(
    int layer_idx, std::shared_ptr<Tensor<int8_t>> q,
    std::shared_ptr<Tensor<int8_t>> k) {
  return decoder_layers_->at(layer_idx).Matmul_CPU_QK(q, k);
}

std::shared_ptr<Tensor<int32_t>> SecLLM::Matmul_CPU_PV(
    int layer_idx, std::shared_ptr<Tensor<int8_t>> p,
    std::shared_ptr<Tensor<int8_t>> v) {
  return decoder_layers_->at(layer_idx).Matmul_CPU_PV(p, v);
}

void SecLLM::SetAttentionMask(float* mask, int M, int N) {

  attn_mask_ = std::vector<std::vector<float>>(M, std::vector<float>(N));
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      attn_mask_.at(i).at(j) = mask[i * N + j];
    }
  }
}

void SecLLM::SetBatchSizeAndTokenLength(int layer_idx, int bsz,
                                        int token_length) {
  decoder_layers_->at(layer_idx).SetBatchSizeAndTokenLength(bsz, token_length);
}

void SecLLM::GenerateSecretKey_QK(int layer_idx) {
  decoder_layers_->at(layer_idx).GenerateSecretKey_QK();
}

void SecLLM::GenerateDecryptionKey_QK(int layer_idx,
                                      std::shared_ptr<Tensor<uint32_t>> x,
                                      std::shared_ptr<Tensor<uint32_t>> y) {
  decoder_layers_->at(layer_idx).GenerateDecryptionKey_QK(x, y);
}

void SecLLM::GenerateDecAddBuffer_QK(int layer_idx) {
  decoder_layers_->at(layer_idx).GenerateDecAddBuffer_QK();
}

void SecLLM::GenerateDecMultBuffer_QK(int layer_idx) {
  decoder_layers_->at(layer_idx).GenerateDecMultBuffer_QK();
}

void SecLLM::GenerateUnshiftBuffer_QK(int layer_idx) {
  decoder_layers_->at(layer_idx).GenerateUnshiftBuffer_QK();
}

void SecLLM::EncryptX_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).EncryptX_QK(out, in);
}

void SecLLM::EncryptY_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).EncryptY_QK(out, in);
}

void SecLLM::Decrypt_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                        std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).Decrypt_QK(out, in);
}

void SecLLM::GenerateSecretKey_PV(int layer_idx) {
  decoder_layers_->at(layer_idx).GenerateSecretKey_PV();
}

void SecLLM::GenerateDecryptionKey_PV(int layer_idx,
                                      std::shared_ptr<Tensor<uint32_t>> x,
                                      std::shared_ptr<Tensor<uint32_t>> y) {
  decoder_layers_->at(layer_idx).GenerateDecryptionKey_PV(x, y);
}

void SecLLM::GenerateDecAddBuffer_PV(int layer_idx) {
  decoder_layers_->at(layer_idx).GenerateDecAddBuffer_PV();
}

void SecLLM::GenerateDecMultBuffer_PV(int layer_idx) {
  decoder_layers_->at(layer_idx).GenerateDecMultBuffer_PV();
}

void SecLLM::GenerateUnshiftBuffer_PV(int layer_idx) {
  decoder_layers_->at(layer_idx).GenerateUnshiftBuffer_PV();
}

void SecLLM::EncryptX_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).EncryptX_PV(out, in);
}

void SecLLM::EncryptY_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).EncryptY_PV(out, in);
}

void SecLLM::Decrypt_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                        std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).Decrypt_PV(out, in);
}

bool SecLLM::QKKeyIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsQKKeyGenerated();
}

bool SecLLM::QKDecKeyIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsQKDecKeyGenerated();
}

bool SecLLM::QKDecAddBufferIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsQKDecAddBufferGenerated();
}

bool SecLLM::QKDecMultBufferIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsQKDecMultBufferGenerated();
}

bool SecLLM::QKShiftedQIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsQKShiftQDone();
}

bool SecLLM::QKShiftedKIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsQKShiftKDone();
}

bool SecLLM::QKUnshiftBufferIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsQKUnshiftBufferGenerated();
}

bool SecLLM::PVKeyIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVKeyGenerated();
}

bool SecLLM::PVDecKeyIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVDecKeyGenerated();
}

bool SecLLM::PVDecAddBufferIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVDecAddBufferGenerated();
}

bool SecLLM::PVDecMultBufferIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVDecMultBufferGenerated();
}

bool SecLLM::PVShiftedPIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVShiftPDone();
}

bool SecLLM::PVShiftedVIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVShiftVDone();
}

bool SecLLM::PVUnshiftBufferIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVUnshiftBufferGenerated();
}

}  // namespace jpyo0803

void Internal_PrintTest(int a, int b) {
#if SGX_ENABLE == 0
  std::cout << "Hello from C++: " << a << " / " << b << std::endl;
#endif
}

void Internal_CreateSecLLM(int hidden_size, int intermediate_size,
                           int max_position_embeddings, int num_attention_heads,
                           int num_hidden_layers, int num_key_value_heads,
                           int enc_key_pool_size) {
  if (secllm_ptr == nullptr) {
    secllm_ptr = new jpyo0803::SecLLM(hidden_size, intermediate_size,
                                      max_position_embeddings,
                                      num_attention_heads, num_hidden_layers,
                                      num_key_value_heads, enc_key_pool_size);
  }
}

void Internal_Softmax_InPlace(float* x, int B, int M, int N, int K) {
  jpyo0803::Softmax_InPlace(x, B, M, N, K);
}

void Internal_Softmax(int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(from);
  auto shape = retrieved_data->shape();

  int B = shape[0];
  int M = shape[1];
  int N = shape[2];
  int K = shape[3];

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape);

  // retrieved_data->PrintAsTorchStyle();
  // exit(-1);

  jpyo0803::Softmax(out->data().data(), retrieved_data->data().data(), B, M, N,
                    K);

  auto locs = std::vector<int>(to, to + to_len);

  secllm_ptr->BookKeeperStore_Float(locs, out);
}

void Internal_SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M,
                             int N) {
  jpyo0803::SwiGLU_InPlace(gate_in, up_in, B, M, N);
}

void Internal_SwiGLU(int from1, int from2, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data1 =
      secllm_ptr->BookKeeperLoad_Float(from1);
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data2 =
      secllm_ptr->BookKeeperLoad_Float(from2);
  auto shape = retrieved_data1->shape();

  int B = shape[0];
  int M = shape[1];
  int N = shape[2];

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape);

  jpyo0803::SwiGLU(out->data().data(), retrieved_data1->data().data(),
                   retrieved_data2->data().data(), B, M, N);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Float(locs, out);
}

void Internal_RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                              int N, float eps) {
  jpyo0803::RMSNorm_InPlace(x, weight, B, M, N, eps);
}

void Internal_RMSNorm(int layer_idx, int from, int to_len, int* to,
                      int type) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(from);
  auto shape = retrieved_data->shape();

  int B = shape[0];
  int M = shape[1];
  int N = shape[2];

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape);

  secllm_ptr->RMSNorm(layer_idx, out, retrieved_data, type);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Float(locs, out);
}

void Internal_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N) {
  jpyo0803::ElementWiseAdd_InPlace(x, y, B, M, N);
}

void Internal_ElementWiseAdd(int from1, int from2, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data1 =
      secllm_ptr->BookKeeperLoad_Float(from1);
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data2 =
      secllm_ptr->BookKeeperLoad_Float(from2);
  auto shape = retrieved_data1->shape();

  int B = shape[0];
  int M = shape[1];
  int N = shape[2];

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape);

  jpyo0803::ElementWiseAdd(out->data().data(), retrieved_data1->data().data(),
                           retrieved_data2->data().data(), B, M, N);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Float(locs, out);
}

void Internal_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                                const float* const cos, const float* const sin,
                                int B, int Q_M, int K_M, int N, int K) {
  jpyo0803::ApplyRotaryPosEmb(q_tensor, k_tensor, cos, sin, B, Q_M, K_M, N, K);
}

void Internal_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                                   const float* const position_ids,
                                   int position_ids_M, float* cos, float* sin) {
  jpyo0803::LlamaRotaryEmbedding(inv_freq, inv_freq_M, position_ids,
                                 position_ids_M, cos, sin);
}

uint32_t Internal_GenerateCPRNG() {
  return jpyo0803::GenerateCPRNG();
}

uint32_t Internal_GenerateMultKey() {
  return jpyo0803::GenerateMultKey();
}

uint32_t Internal_GenerateAddKey() {
  return jpyo0803::GenerateAddKey();
}

void Internal_Reset() {
  secllm_ptr->Reset();
}

// TODO(jpyo0803): Where is its declaration?
void Internal_BookKeeperStore_Float(int loc, float* data, int shape_len,
                                    int* shape) {
  // Convert shape array to std::vector for shape management
  std::vector<int> shape_vec(shape, shape + shape_len);

  // Compute the total number of elements based on shape
  int num_elements = std::accumulate(shape_vec.begin(), shape_vec.end(), 1,
                                     std::multiplies<int>());

  // Create an empty std::vector to hold the data
  std::vector<float> input_vec(num_elements);

  // Use Eigen::Map to vectorize the copy operation
  Eigen::Map<Eigen::VectorXf> data_map(data, num_elements);
  Eigen::Map<Eigen::VectorXf>(input_vec.data(), num_elements) = data_map;

  // Now input_vec contains the data copied from `data` in a vectorized way
  jpyo0803::Tensor<float> tensor(shape_vec, std::move(input_vec));

  // Store the tensor in the BookKeeper
  std::shared_ptr<jpyo0803::Tensor<float>> data_ptr =
      std::make_shared<jpyo0803::Tensor<float>>(tensor);

  secllm_ptr->BookKeeperStore_Float({loc}, data_ptr);
}

void Internal_BookKeeperStore_Int32(int loc, int32_t* data, int shape_len,
                                    int* shape) {
  // Convert shape array to std::vector for shape management
  std::vector<int> shape_vec(shape, shape + shape_len);

  // Compute the total number of elements based on shape
  int num_elements = std::accumulate(shape_vec.begin(), shape_vec.end(), 1,
                                     std::multiplies<int>());

  // Create an empty std::vector to hold the data
  std::vector<int32_t> input_vec(num_elements);

  // Use Eigen::Map to vectorize the copy operation
  Eigen::Map<Eigen::VectorXi> data_map(data, num_elements);
  Eigen::Map<Eigen::VectorXi>(input_vec.data(), num_elements) = data_map;

  // Create the tensor with the data in the input_vec
  jpyo0803::Tensor<int32_t> tensor(shape_vec, std::move(input_vec));

  // Store the tensor in the BookKeeper
  std::shared_ptr<jpyo0803::Tensor<int32_t>> data_ptr =
      std::make_shared<jpyo0803::Tensor<int32_t>>(tensor);

  secllm_ptr->BookKeeperStore_Int32({loc}, data_ptr);
}

void Internal_BookKeeperStore_Uint32(int loc, uint32_t* data, int shape_len,
                                     int* shape) {
  // Convert shape array to std::vector for shape management
  std::vector<int> shape_vec(shape, shape + shape_len);

  // Compute the total number of elements based on shape
  int num_elements = std::accumulate(shape_vec.begin(), shape_vec.end(), 1,
                                     std::multiplies<int>());

  // Create an empty std::vector to hold the data
  std::vector<uint32_t> input_vec(num_elements);

  // Use Eigen::Map to vectorize the copy operation
  Eigen::Map<Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>> data_map(data,
                                                                  num_elements);
  Eigen::Map<Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>>(
      input_vec.data(), num_elements) = data_map;

  // Create the tensor with the data in the input_vec
  jpyo0803::Tensor<uint32_t> tensor(shape_vec, std::move(input_vec));

  // Store the tensor in the BookKeeper
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> data_ptr =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(tensor);

  secllm_ptr->BookKeeperStore_Uint32({loc}, data_ptr);
}

void Internal_BookKeeperStore_Int8(int loc, int8_t* data, int shape_len,
                                   int* shape) {
  // Convert shape array to std::vector for shape management
  std::vector<int> shape_vec(shape, shape + shape_len);

  // Compute the total number of elements based on shape
  int num_elements = std::accumulate(shape_vec.begin(), shape_vec.end(), 1,
                                     std::multiplies<int>());

  // Create an empty std::vector to hold the data
  std::vector<int8_t> input_vec(num_elements);

  // Use Eigen::Map to vectorize the copy operation
  Eigen::Map<Eigen::Matrix<int8_t, Eigen::Dynamic, 1>> data_map(data,
                                                                num_elements);
  Eigen::Map<Eigen::Matrix<int8_t, Eigen::Dynamic, 1>>(input_vec.data(),
                                                       num_elements) = data_map;

  // Create the tensor with the data in the input_vec
  jpyo0803::Tensor<int8_t> tensor(shape_vec, std::move(input_vec));

  // Store the tensor in the BookKeeper
  std::shared_ptr<jpyo0803::Tensor<int8_t>> data_ptr =
      std::make_shared<jpyo0803::Tensor<int8_t>>(tensor);

  secllm_ptr->BookKeeperStore_Int8({loc}, data_ptr);
}

void Internal_BookKeeperLoad_Float(int loc, float* out, int shape_len,
                                   int* shape) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(loc);

  if (retrieved_data == nullptr) {
    throw std::runtime_error("No object at the location: " +
                             std::to_string(loc));
  }

  // Check for shape mismatch
  for (int i = 0; i < shape_len; ++i) {
    if (retrieved_data->shape()[i] != shape[i]) {
      throw std::runtime_error("Shape mismatch at the location: " +
                               std::to_string(loc));
    }
  }

  // Get the total number of elements
  int num_elements =
      std::accumulate(shape, shape + shape_len, 1, std::multiplies<int>());

  // Use Eigen's Map to vectorize the data copy
  Eigen::Map<Eigen::VectorXf>(out, num_elements) =
      Eigen::Map<const Eigen::VectorXf>(retrieved_data->data().data(),
                                        num_elements);
}

void Internal_BookKeeperLoad_Int32(int loc, int32_t* out, int shape_len,
                                   int* shape) {
  std::shared_ptr<jpyo0803::Tensor<int32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int32(loc);

  if (retrieved_data == nullptr) {
    throw std::runtime_error("No object at the location: " +
                             std::to_string(loc));
  }

  // Check for shape mismatch
  for (int i = 0; i < shape_len; ++i) {
    if (retrieved_data->shape()[i] != shape[i]) {
      throw std::runtime_error("Shape mismatch at the location: " +
                               std::to_string(loc));
    }
  }

  // Get the total number of elements
  int num_elements =
      std::accumulate(shape, shape + shape_len, 1, std::multiplies<int>());

  // Use Eigen's Map to vectorize the data copy
  Eigen::Map<Eigen::VectorXi>(out, num_elements) =
      Eigen::Map<const Eigen::VectorXi>(retrieved_data->data().data(),
                                        num_elements);
}

void Internal_BookKeeperLoad_Uint32(int loc, uint32_t* out, int shape_len,
                                    int* shape) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(loc);

  if (retrieved_data == nullptr) {
    throw std::runtime_error("No object at the location: " +
                             std::to_string(loc));
  }

  // Check for shape mismatch
  for (int i = 0; i < shape_len; ++i) {
    if (retrieved_data->shape()[i] != shape[i]) {
      throw std::runtime_error("Shape mismatch at the location: " +
                               std::to_string(loc));
    }
  }

  // Get the total number of elements
  int num_elements =
      std::accumulate(shape, shape + shape_len, 1, std::multiplies<int>());

  // Use Eigen's Map to vectorize the data copy
  Eigen::Map<Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>>(out, num_elements) =
      Eigen::Map<const Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>>(
          retrieved_data->data().data(), num_elements);
}

void Internal_BookKeeperLoad_Int8(int loc, int8_t* out, int shape_len,
                                  int* shape) {
  std::shared_ptr<jpyo0803::Tensor<int8_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int8(loc);

  if (retrieved_data == nullptr) {
    throw std::runtime_error("No object at the location: " +
                             std::to_string(loc));
  }

  // Check for shape mismatch
  for (int i = 0; i < shape_len; ++i) {
    if (retrieved_data->shape()[i] != shape[i]) {
      throw std::runtime_error("Shape mismatch at the location: " +
                               std::to_string(loc));
    }
  }

  // Get the total number of elements
  int num_elements =
      std::accumulate(shape, shape + shape_len, 1, std::multiplies<int>());

  // Use Eigen's Map to vectorize the data copy
  Eigen::Map<Eigen::Matrix<int8_t, Eigen::Dynamic, 1>>(out, num_elements) =
      Eigen::Map<const Eigen::Matrix<int8_t, Eigen::Dynamic, 1>>(
          retrieved_data->data().data(), num_elements);
}

void Internal_ReplicateTensor_Float(int from, int* to, int to_len) {
  std::vector<int> locs(to, to + to_len);

  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(from);
  // Notice, this removes a tensor in the book keeper
  secllm_ptr->BookKeeperStore_Float(locs, retrieved_data);
}

void Internal_ReplicateTensor_Int32(int from, int* to, int to_len) {
  std::vector<int> locs(to, to + to_len);

  std::shared_ptr<jpyo0803::Tensor<int32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int32(from);
  // Notice, this removes a tensor in the book keeper
  secllm_ptr->BookKeeperStore_Int32(locs, retrieved_data);
}

void Internal_ReplicateTensor_Uint32(int from, int* to, int to_len) {
  std::vector<int> locs(to, to + to_len);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);
  // Notice, this removes a tensor in the book keeper
  secllm_ptr->BookKeeperStore_Uint32(locs, retrieved_data);
}

void Internal_ReplicateTensor_Int8(int from, int* to, int to_len) {
  std::vector<int> locs(to, to + to_len);

  std::shared_ptr<jpyo0803::Tensor<int8_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int8(from);
  // Notice, this removes a tensor in the book keeper
  secllm_ptr->BookKeeperStore_Int8(locs, retrieved_data);
}

void Internal_GetCprngTensor(int* out, int shape_len, int* shape) {
  int num_elements =
      std::accumulate(shape, shape + shape_len, 1, std::multiplies<int>());

  for (int i = 0; i < num_elements; ++i) {
    out[i] = jpyo0803::GenerateCPRNG();
  }
}

void Internal_SetEncKeyAndDecKey(int layer_idx, int* src_enc_key_pool,
                                 int* src_dec_key,
                                 jpyo0803::ProjectionType type) {
  secllm_ptr->SetEncKeyAndDecKey(layer_idx, src_enc_key_pool, src_dec_key,
                                 type);
}

void Internal_SetLinearWeightScales(int layer_idx, float* scales, int len,
                                    jpyo0803::ProjectionType type) {
  // weight scales's dim == 1
  secllm_ptr->SetLinearWeightScales(layer_idx, scales, len, type);
}

void Internal_SetRMSNormWeight(int layer_idx, float* weight, float eps,
                               int type) {
  secllm_ptr->SetRMSNormWeight(layer_idx, weight, eps, type);
}

void Internal_QuantizeLinearActivation(int layer_idx, int from,
                                       int to_len, int* to,
                                       jpyo0803::ProjectionType type) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(from);

  std::shared_ptr<jpyo0803::Tensor<int8_t>> out =
      std::make_shared<jpyo0803::Tensor<int8_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeLinearActivation(layer_idx, out, retrieved_data, type);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int8(locs, out);
}

void Internal_EncryptLinearActivation(int layer_idx, int from,
                                      int to_len, int* to,
                                      jpyo0803::ProjectionType type) {
  std::shared_ptr<jpyo0803::Tensor<int8_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int8(from);

  std::shared_ptr<jpyo0803::Tensor<int32_t>> out =
      std::make_shared<jpyo0803::Tensor<int32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptLinearActivation(layer_idx, out, retrieved_data, type);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int32(locs, out);
}

void Internal_DecryptLinearActivation(int layer_idx, int from,
                                      int to_len, int* to,
                                      jpyo0803::ProjectionType type) {
  std::shared_ptr<jpyo0803::Tensor<int32_t>> enc_tensor =
      secllm_ptr->BookKeeperLoad_Int32(from);

  std::shared_ptr<jpyo0803::Tensor<int32_t>> out =
      std::make_shared<jpyo0803::Tensor<int32_t>>(enc_tensor->shape());

  secllm_ptr->DecryptLinearActivation(layer_idx, out, enc_tensor, type);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int32(locs, out);
}

void Internal_DequantizeLinearActivation(int layer_idx, int from,
                                         int to_len, int* to,
                                         jpyo0803::ProjectionType type) {
  std::shared_ptr<jpyo0803::Tensor<int32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int32(from);

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(retrieved_data->shape());

  secllm_ptr->DequantizeLinearActivation(layer_idx, out, retrieved_data, type);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Float(locs, out);
}

void Internal_SetQKVOutputScales(int layer_idx, float q_output_scale,
                                 float k_output_scale, float v_output_scale) {
  secllm_ptr->SetQKVOutputScales(layer_idx, q_output_scale, k_output_scale,
                                 v_output_scale);
}

void Internal_QuantizeQ_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(from);

  std::shared_ptr<jpyo0803::Tensor<int8_t>> out =
      std::make_shared<jpyo0803::Tensor<int8_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeQ_QK(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int8(locs, out);
}

void Internal_ShiftQ_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<int8_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int8(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->ShiftQ_QK(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_QuantizeK_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(from);

  std::shared_ptr<jpyo0803::Tensor<int8_t>> out =
      std::make_shared<jpyo0803::Tensor<int8_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeK_QK(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int8(locs, out);
}

void Internal_ShiftK_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<int8_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int8(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->ShiftK_QK(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_Unshift_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<int32_t>> out =
      std::make_shared<jpyo0803::Tensor<int32_t>>(retrieved_data->shape());

  secllm_ptr->Unshift_QK(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int32(locs, out);
}

void Internal_Dequantize_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<int32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int32(from);

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(retrieved_data->shape());

  secllm_ptr->Dequantize_QK(layer_idx, out, retrieved_data);

  // Need masking

  auto shape = out->shape();
  int B = shape.at(0);
  int M = shape.at(1);
  int K = shape.at(2);
  int N = shape.at(3);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          out->data()[b * M * K * N + m * K * N + k * N + n] +=
              secllm_ptr->attn_mask_.at(k).at(n);
        }
      }
    }
  }

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Float(locs, out);
}

void Internal_QuantizeP_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(from);

  std::shared_ptr<jpyo0803::Tensor<int8_t>> out =
      std::make_shared<jpyo0803::Tensor<int8_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeP_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int8(locs, out);
}

void Internal_ShiftP_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<int8_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int8(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->ShiftP_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_QuantizeV_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Float(from);

  std::shared_ptr<jpyo0803::Tensor<int8_t>> out =
      std::make_shared<jpyo0803::Tensor<int8_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeV_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int8(locs, out);
}

void Internal_ShiftV_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<int8_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int8(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->ShiftV_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_Unshift_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<int32_t>> out =
      std::make_shared<jpyo0803::Tensor<int32_t>>(retrieved_data->shape());

  secllm_ptr->Unshift_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int32(locs, out);
}

void Internal_Dequantize_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<int32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Int32(from);

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(retrieved_data->shape());

  secllm_ptr->Dequantize_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Float(locs, out);
}

void Internal_SetAttentionMask(float* mask, int M, int N) {
  secllm_ptr->SetAttentionMask(mask, M, N);
}

void Internal_SetBatchSizeAndTokenLength(int layer_idx, int bsz,
                                         int token_length) {
  secllm_ptr->SetBatchSizeAndTokenLength(layer_idx, bsz, token_length);
}

void Internal_GenerateSecretKey_QK(int layer_idx) {
  secllm_ptr->GenerateSecretKey_QK(layer_idx);
}

void Internal_GenerateDecryptionKey_QK(int layer_idx, int from_x, int from_y) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> x =
      secllm_ptr->BookKeeperLoad_Uint32(from_x);
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> y =
      secllm_ptr->BookKeeperLoad_Uint32(from_y);

  secllm_ptr->GenerateDecryptionKey_QK(layer_idx, x, y);
}

void Internal_GenerateDecAddBuffer_QK(int layer_idx) {
  secllm_ptr->GenerateDecAddBuffer_QK(layer_idx);
}

void Internal_GenerateDecMultBuffer_QK(int layer_idx) {
  secllm_ptr->GenerateDecMultBuffer_QK(layer_idx);
}

void Internal_GenerateUnshiftBuffer_QK(int layer_idx) {
  secllm_ptr->GenerateUnshiftBuffer_QK(layer_idx);
}

void Internal_EncryptX_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptX_QK(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_EncryptY_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptY_QK(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_Decrypt_QK(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->Decrypt_QK(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_GenerateSecretKey_PV(int layer_idx) {
  secllm_ptr->GenerateSecretKey_PV(layer_idx);
}

void Internal_GenerateDecryptionKey_PV(int layer_idx, int from_x, int from_y) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> x =
      secllm_ptr->BookKeeperLoad_Uint32(from_x);
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> y =
      secllm_ptr->BookKeeperLoad_Uint32(from_y);

  secllm_ptr->GenerateDecryptionKey_PV(layer_idx, x, y);
}

void Internal_GenerateDecAddBuffer_PV(int layer_idx) {
  secllm_ptr->GenerateDecAddBuffer_PV(layer_idx);
}

void Internal_GenerateDecMultBuffer_PV(int layer_idx) {
  secllm_ptr->GenerateDecMultBuffer_PV(layer_idx);
}

void Internal_GenerateUnshiftBuffer_PV(int layer_idx) {
  secllm_ptr->GenerateUnshiftBuffer_PV(layer_idx);
}

void Internal_EncryptX_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptX_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_EncryptY_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptY_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_Decrypt_PV(int layer_idx, int from, int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->Decrypt_PV(layer_idx, out, retrieved_data);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_BookKeeperIsAvailable_Float(int loc, bool* ret) {
  *ret = secllm_ptr->BookKeeperIsAvailable_Float(loc);
}

void Internal_BookKeeperIsAvailable_Int32(int loc, bool* ret) {
  *ret = secllm_ptr->BookKeeperIsAvailable_Int32(loc);
}

void Internal_BookKeeperIsAvailable_Uint32(int loc, bool* ret) {
  *ret = secllm_ptr->BookKeeperIsAvailable_Uint32(loc);
}

void Internal_BookKeeperIsAvailable_Int8(int loc, bool* ret) {
  *ret = secllm_ptr->BookKeeperIsAvailable_Int8(loc);
}

void Internal_BookKeeperGetShapeLength_Float(int loc, int* ret) {
  *ret = secllm_ptr->BookKeeperGetShapeLength_Float(loc);
}

void Internal_BookKeeperGetShapeLength_Int32(int loc, int* ret) {
  *ret = secllm_ptr->BookKeeperGetShapeLength_Int32(loc);
}

void Internal_BookKeeperGetShapeLength_Uint32(int loc, int* ret) {
  *ret = secllm_ptr->BookKeeperGetShapeLength_Uint32(loc);
}

void Internal_BookKeeperGetShapeLength_Int8(int loc, int* ret) {
  *ret = secllm_ptr->BookKeeperGetShapeLength_Int8(loc);
}

void Internal_BookKeeperGetShape_Float(int loc, int* ret) {
  secllm_ptr->BookKeeperGetShape_Float(loc, ret);
}

void Internal_BookKeeperGetShape_Int32(int loc, int* ret) {
  secllm_ptr->BookKeeperGetShape_Int32(loc, ret);
}

void Internal_BookKeeperGetShape_Uint32(int loc, int* ret) {
  secllm_ptr->BookKeeperGetShape_Uint32(loc, ret);
}

void Internal_BookKeeperGetShape_Int8(int loc, int* ret) {
  secllm_ptr->BookKeeperGetShape_Int8(loc, ret);
}

void Internal_QKKeyIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKKeyIsAvailable(layer_idx);
}

void Internal_QKDecKeyIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKDecKeyIsAvailable(layer_idx);
}

void Internal_QKDecAddBufferIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKDecAddBufferIsAvailable(layer_idx);
}

void Internal_QKDecMultBufferIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKDecMultBufferIsAvailable(layer_idx);
}

void Internal_QKUnshiftBufferIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKUnshiftBufferIsAvailable(layer_idx);
}

void Internal_QKShiftedQIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKShiftedQIsAvailable(layer_idx);
}

void Internal_QKShiftedKIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKShiftedKIsAvailable(layer_idx);
}

void Internal_PVKeyIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVKeyIsAvailable(layer_idx);
}

void Internal_PVDecKeyIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVDecKeyIsAvailable(layer_idx);
}

void Internal_PVDecAddBufferIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVDecAddBufferIsAvailable(layer_idx);
}

void Internal_PVDecMultBufferIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVDecMultBufferIsAvailable(layer_idx);
}

void Internal_PVUnshiftBufferIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVUnshiftBufferIsAvailable(layer_idx);
}

void Internal_PVShiftedPIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVShiftedPIsAvailable(layer_idx);
}

void Internal_PVShiftedVIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVShiftedVIsAvailable(layer_idx);
}

void Internal_Matmul_CPU_QK(int layer_idx, int q_from, int k_from,
                            int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<int8_t>> q_tensor =
      secllm_ptr->BookKeeperLoad_Int8(q_from);
  std::shared_ptr<jpyo0803::Tensor<int8_t>> k_tensor =
      secllm_ptr->BookKeeperLoad_Int8(k_from);

  std::shared_ptr<jpyo0803::Tensor<int32_t>> out =
      secllm_ptr->Matmul_CPU_QK(layer_idx, q_tensor, k_tensor);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int32(locs, out);
}

void Internal_Matmul_CPU_PV(int layer_idx, int p_from, int v_from,
                            int to_len, int* to) {
  std::shared_ptr<jpyo0803::Tensor<int8_t>> p_tensor =
      secllm_ptr->BookKeeperLoad_Int8(p_from);
  std::shared_ptr<jpyo0803::Tensor<int8_t>> v_tensor =
      secllm_ptr->BookKeeperLoad_Int8(v_from);

  std::shared_ptr<jpyo0803::Tensor<int32_t>> out =
      secllm_ptr->Matmul_CPU_PV(layer_idx, p_tensor, v_tensor);

  auto locs = std::vector<int>(to, to + to_len);
  secllm_ptr->BookKeeperStore_Int32(locs, out);
}

void Internal_BookKeeperLoadWithoutReset_Float(int loc, float* out) {
  auto retrieved_data = secllm_ptr->BookKeeperLoadWithoutReset_Float(loc);
  std::copy(retrieved_data->data().begin(), retrieved_data->data().end(), out);
}

void Internal_BookKeeperLoadWithoutReset_Int32(int loc, int32_t* out) {
  auto retrieved_data = secllm_ptr->BookKeeperLoadWithoutReset_Int32(loc);
  std::copy(retrieved_data->data().begin(), retrieved_data->data().end(), out);
}

void Internal_BookKeeperLoadWithoutReset_Uint32(int loc, uint32_t* out) {
  auto retrieved_data = secllm_ptr->BookKeeperLoadWithoutReset_Uint32(loc);
  std::copy(retrieved_data->data().begin(), retrieved_data->data().end(), out);
}

void Internal_BookKeeperLoadWithoutReset_Int8(int loc, int8_t* out) {
  auto retrieved_data = secllm_ptr->BookKeeperLoadWithoutReset_Int8(loc);
  std::copy(retrieved_data->data().begin(), retrieved_data->data().end(), out);
}

extern "C" {

void ecall_Internal_PrintTest(int a, int b) {
  Internal_PrintTest(a, b);
}

void ecall_Internal_CreateSecLLM(int hidden_size, int intermediate_size,
                           int max_position_embeddings, int num_attention_heads,
                           int num_hidden_layers, int num_key_value_heads,
                           int enc_key_pool_size) {
  Internal_CreateSecLLM(hidden_size, intermediate_size,
                             max_position_embeddings, num_attention_heads,
                             num_hidden_layers, num_key_value_heads,
                             enc_key_pool_size);
}

void ecall_Internal_Softmax_InPlace(float* x, int B, int M, int N, int K) {
  Internal_Softmax_InPlace(x, B, M, N, K);
}

void ecall_Internal_Softmax(int from, int to_len, int* to) {
  Internal_Softmax(from, to_len, to);
}

void ecall_Internal_RMSNorm_InPlace(float* x, const float* const weight, int B, int M, int N, float eps) {
  Internal_RMSNorm_InPlace(x, weight, B, M, N, eps);
}

void ecall_Internal_RMSNorm(int layer_idx, int from, int to_len, int* to, int type) {
  Internal_RMSNorm(layer_idx, from, to_len, to, type);
}

void ecall_Internal_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N) {
  Internal_ElementWiseAdd_InPlace(x, y, B, M, N);
}

void ecall_Internal_ElementWiseAdd(int from1, int from2, int to_len, int* to) {
  Internal_ElementWiseAdd(from1, from2, to_len, to);
}

void ecall_Internal_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor, const float* const cos, const float* const sin, int B, int Q_M, int K_M, int N, int K) {
  Internal_ApplyRotaryPosEmb(q_tensor, k_tensor, cos, sin, B, Q_M, K_M, N, K);
}

void ecall_Internal_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                                   const float* const position_ids,
                                   int position_ids_M, float* cos, float* sin) {
  Internal_LlamaRotaryEmbedding(inv_freq, inv_freq_M, position_ids,
                                     position_ids_M, cos, sin);
                                   }

// Deprecated
uint32_t ecall_Internal_GenerateCPRNG() {
  return 0;
}

// Deprecated
uint32_t ecall_Internal_GenerateMultKey() {
  return 0;
}

// Deprecated
uint32_t ecall_Internal_GenerateAddKey() {
  return 0;
}

void ecall_Internal_Reset() {
  Internal_Reset();
}

void ecall_Internal_ReplicateTensor_Float(int from, int* to, int to_len) {
  Internal_ReplicateTensor_Float(from, to, to_len);
}

void ecall_Internal_ReplicateTensor_Int32(int from, int* to, int to_len) {
  Internal_ReplicateTensor_Int32(from, to, to_len);
}

void ecall_Internal_ReplicateTensor_Uint32(int from, int* to, int to_len) {
  Internal_ReplicateTensor_Uint32(from, to, to_len);
}

void ecall_Internal_ReplicateTensor_Int8(int from, int* to, int to_len) {
  Internal_ReplicateTensor_Int8(from, to, to_len);
}

void ecall_Internal_SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool, int* dec_key, jpyo0803::ProjectionType type) {
  Internal_SetEncKeyAndDecKey(layer_idx, enc_key_pool, dec_key, type);
}

void ecall_Internal_SetLinearWeightScales(int layer_idx, float* scales, int len, jpyo0803::ProjectionType type) {
  Internal_SetLinearWeightScales(layer_idx, scales, len, type);
}

void ecall_Internal_SetRMSNormWeight(int layer_idx, float* weight, float eps, int type) {
  Internal_SetRMSNormWeight(layer_idx, weight, eps, type);
}

void ecall_Internal_QuantizeLinearActivation(int layer_idx, int from, int to_len, int* to, jpyo0803::ProjectionType type) {
  Internal_QuantizeLinearActivation(layer_idx, from, to_len, to, type);
}

void ecall_Internal_EncryptLinearActivation(int layer_idx, int from, int to_len, int* to, jpyo0803::ProjectionType type) {
  Internal_EncryptLinearActivation(layer_idx, from, to_len, to, type);
}

void ecall_Internal_DecryptLinearActivation(int layer_idx, int from, int to_len, int* to, jpyo0803::ProjectionType type) {
  Internal_DecryptLinearActivation(layer_idx, from, to_len, to, type);
}

void ecall_Internal_DequantizeLinearActivation(int layer_idx, int from, int to_len, int* to, jpyo0803::ProjectionType type) {
  Internal_DequantizeLinearActivation(layer_idx, from, to_len, to, type);
}

void ecall_Internal_SetQKVOutputScales(int layer_idx, float q_output_scale, float k_output_scale, float v_output_scale) {
  Internal_SetQKVOutputScales(layer_idx, q_output_scale, k_output_scale, v_output_scale);
}

void ecall_Internal_QuantizeAndShiftQ(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeQ_QK(layer_idx, from, to_len, to);
}

void ecall_Internal_QuantizeAndShiftK(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeK_QK(layer_idx, from, to_len, to);
}

void ecall_Internal_SetAttentionMask(float* mask, int M, int N) {
  Internal_SetAttentionMask(mask, M, N);
}

void ecall_Internal_SetBatchSizeAndTokenLength(int layer_idx, int bsz, int token_length) {
  Internal_SetBatchSizeAndTokenLength(layer_idx, bsz, token_length);
}

void ecall_Internal_GenerateSecretKey_QK(int layer_idx) {
  Internal_GenerateSecretKey_QK(layer_idx);
}
void ecall_Internal_GenerateDecryptionKey_QK(int layer_idx, int from_x, int from_y) {
  Internal_GenerateDecryptionKey_QK(layer_idx, from_x, from_y);
}
void ecall_Internal_GenerateDecAddBuffer_QK(int layer_idx) {
  Internal_GenerateDecAddBuffer_QK(layer_idx);
}
void ecall_Internal_GenerateDecMultBuffer_QK(int layer_idx) {
  Internal_GenerateDecMultBuffer_QK(layer_idx);
}
void ecall_Internal_GenerateUnshiftBuffer_QK(int layer_idx) {
  Internal_GenerateUnshiftBuffer_QK(layer_idx);
}

void ecall_Internal_QuantizeQ_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeQ_QK(layer_idx, from, to_len, to);
}
void ecall_Internal_ShiftQ_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_ShiftQ_QK(layer_idx, from, to_len, to);
}
void ecall_Internal_QuantizeK_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeK_QK(layer_idx, from, to_len, to);
}
void ecall_Internal_ShiftK_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_ShiftK_QK(layer_idx, from, to_len, to);
}
void ecall_Internal_EncryptX_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_EncryptX_QK(layer_idx, from, to_len, to);
}
void ecall_Internal_EncryptY_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_EncryptY_QK(layer_idx, from, to_len, to);
}
void ecall_Internal_Decrypt_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_Decrypt_QK(layer_idx, from, to_len, to);
}
void ecall_Internal_Unshift_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_Unshift_QK(layer_idx, from, to_len, to);
}
void ecall_Internal_Dequantize_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_Dequantize_QK(layer_idx, from, to_len, to);
}

void ecall_Internal_GenerateSecretKey_PV(int layer_idx) {
  Internal_GenerateSecretKey_PV(layer_idx);
}
void ecall_Internal_GenerateDecryptionKey_PV(int layer_idx, int from_x, int from_y) {
  Internal_GenerateDecryptionKey_PV(layer_idx, from_x, from_y);
}
void ecall_Internal_GenerateDecAddBuffer_PV(int layer_idx) {
  Internal_GenerateDecAddBuffer_PV(layer_idx);
}
void ecall_Internal_GenerateDecMultBuffer_PV(int layer_idx) {
  Internal_GenerateDecMultBuffer_PV(layer_idx);
}
void ecall_Internal_GenerateUnshiftBuffer_PV(int layer_idx) {
  Internal_GenerateUnshiftBuffer_PV(layer_idx);
}

void ecall_Internal_QuantizeP_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeP_PV(layer_idx, from, to_len, to);
}
void ecall_Internal_ShiftP_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_ShiftP_PV(layer_idx, from, to_len, to);
}
void ecall_Internal_QuantizeV_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeV_PV(layer_idx, from, to_len, to);
}
void ecall_Internal_ShiftV_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_ShiftV_PV(layer_idx, from, to_len, to);
}
void ecall_Internal_EncryptX_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_EncryptX_PV(layer_idx, from, to_len, to);
}
void ecall_Internal_EncryptY_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_EncryptY_PV(layer_idx, from, to_len, to);
}
void ecall_Internal_Decrypt_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_Decrypt_PV(layer_idx, from, to_len, to);
}
void ecall_Internal_Unshift_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_Unshift_PV(layer_idx, from, to_len, to);
}
void ecall_Internal_Dequantize_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_Dequantize_PV(layer_idx, from, to_len, to);
}

void ecall_Internal_BookKeeperStore_Float(int loc, float* data, int shape_len,
                                    int* shape) {
  Internal_BookKeeperStore_Float(loc, data, shape_len, shape);
                                    }
void ecall_Internal_BookKeeperStore_Int32(int loc, int32_t* data, int shape_len,
                                    int* shape) {
  Internal_BookKeeperStore_Int32(loc, data, shape_len, shape);
                                    }
void ecall_Internal_BookKeeperStore_Uint32(int loc, uint32_t* data, int shape_len,
                                     int* shape) {
  Internal_BookKeeperStore_Uint32(loc, data, shape_len, shape);
                                     }
void ecall_Internal_BookKeeperStore_Int8(int loc, int8_t* data, int shape_len,
                                      int* shape) {
  Internal_BookKeeperStore_Int8(loc, data, shape_len, shape);
                                      }

void ecall_Internal_BookKeeperLoad_Float(int loc, float* out, int shape_len,
                                      int* shape) {
  Internal_BookKeeperLoad_Float(loc, out, shape_len, shape);
                                      }
void ecall_Internal_BookKeeperLoad_Int32(int loc, int32_t* out, int shape_len,
                                        int* shape) {
  Internal_BookKeeperLoad_Int32(loc, out, shape_len, shape);
                                        }
void ecall_Internal_BookKeeperLoad_Uint32(int loc, uint32_t* out, int shape_len,
                                            int* shape) {
  Internal_BookKeeperLoad_Uint32(loc, out, shape_len, shape);
                                            }
void ecall_Internal_BookKeeperLoad_Int8(int loc, int8_t* out, int shape_len,
                                        int* shape) {
  Internal_BookKeeperLoad_Int8(loc, out, shape_len, shape);
                                        }

void ecall_Internal_BookKeeperIsAvailable_Float(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Float(loc, ret);
}
void ecall_Internal_BookKeeperIsAvailable_Int32(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Int32(loc, ret);
}
void ecall_Internal_BookKeeperIsAvailable_Uint32(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Uint32(loc, ret);
}
void ecall_Internal_BookKeeperIsAvailable_Int8(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Int8(loc, ret);
}

void ecall_Internal_BookKeeperGetShapeLength_Float(int loc, int* ret) {
  Internal_BookKeeperGetShapeLength_Float(loc, ret);
}
void ecall_Internal_BookKeeperGetShapeLength_Int32(int loc, int* ret) {
  Internal_BookKeeperGetShapeLength_Int32(loc, ret);
}
void ecall_Internal_BookKeeperGetShapeLength_Uint32(int loc, int* ret) {
  Internal_BookKeeperGetShapeLength_Uint32(loc, ret);
}
void ecall_Internal_BookKeeperGetShapeLength_Int8(int loc, int* ret) {
  Internal_BookKeeperGetShapeLength_Int8(loc, ret);
}

void ecall_Internal_BookKeeperGetShape_Float(int loc, int* out) {
  Internal_BookKeeperGetShape_Float(loc, out);
}
void ecall_Internal_BookKeeperGetShape_Int32(int loc, int* out) {
  Internal_BookKeeperGetShape_Int32(loc, out);
}
void ecall_Internal_BookKeeperGetShape_Uint32(int loc, int* out) {
  Internal_BookKeeperGetShape_Uint32(loc, out);
}
void ecall_Internal_BookKeeperGetShape_Int8(int loc, int* out) {
  Internal_BookKeeperGetShape_Int8(loc, out);
}

void ecall_Internal_QKKeyIsAvailable(int layer_idx, bool* ret) {
  Internal_QKKeyIsAvailable(layer_idx, ret);
}
void ecall_Internal_QKDecKeyIsAvailable(int layer_idx, bool* ret) {
  Internal_QKDecKeyIsAvailable(layer_idx, ret);
}
void ecall_Internal_QKDecAddBufferIsAvailable(int layer_idx, bool* ret) {
  Internal_QKDecAddBufferIsAvailable(layer_idx, ret);
}
void ecall_Internal_QKDecMultBufferIsAvailable(int layer_idx, bool* ret) {
  Internal_QKDecMultBufferIsAvailable(layer_idx, ret);
}

void ecall_Internal_QKShiftedQIsAvailable(int layer_idx, bool* ret) {
  Internal_QKShiftedQIsAvailable(layer_idx, ret);
}
void ecall_Internal_QKShiftedKIsAvailable(int layer_idx, bool* ret) {
  Internal_QKShiftedKIsAvailable(layer_idx, ret);
}

void ecall_Internal_QKUnshiftBufferIsAvailable(int layer_idx, bool* ret) {
  Internal_QKUnshiftBufferIsAvailable(layer_idx, ret);
}

void ecall_Internal_PVKeyIsAvailable(int layer_idx, bool* ret) {
  Internal_PVKeyIsAvailable(layer_idx, ret);
}
void ecall_Internal_PVDecKeyIsAvailable(int layer_idx, bool* ret) {
  Internal_PVDecKeyIsAvailable(layer_idx, ret);
}
void ecall_Internal_PVDecAddBufferIsAvailable(int layer_idx, bool* ret) {
  Internal_PVDecAddBufferIsAvailable(layer_idx, ret);
}
void ecall_Internal_PVDecMultBufferIsAvailable(int layer_idx, bool* ret) {
  Internal_PVDecMultBufferIsAvailable(layer_idx, ret);
}

void ecall_Internal_PVShiftedPIsAvailable(int layer_idx, bool* ret) {
  Internal_PVShiftedPIsAvailable(layer_idx, ret);
}
void ecall_Internal_PVShiftedVIsAvailable(int layer_idx, bool* ret) {
  Internal_PVShiftedVIsAvailable(layer_idx, ret);
}

void ecall_Internal_PVUnshiftBufferIsAvailable(int layer_idx, bool* ret) {
  Internal_PVUnshiftBufferIsAvailable(layer_idx, ret);
}

void ecall_Internal_Matmul_CPU_QK(int layer_idx, int q_from, int k_from,
                            int to_len, int* to) {
  Internal_Matmul_CPU_QK(layer_idx, q_from, k_from, to_len, to);
                            }
void ecall_Internal_Matmul_CPU_PV(int layer_idx, int p_from, int v_from,  
                            int to_len, int* to) {
  Internal_Matmul_CPU_PV(layer_idx, p_from, v_from, to_len, to);
                            }

void ecall_Internal_BookKeeperLoadWithoutReset_Float(int loc, float* out) {
  Internal_BookKeeperLoadWithoutReset_Float(loc, out);
}
void ecall_Internal_BookKeeperLoadWithoutReset_Int32(int loc, int32_t* out) {
  Internal_BookKeeperLoadWithoutReset_Int32(loc, out);
}
void ecall_Internal_BookKeeperLoadWithoutReset_Uint32(int loc, uint32_t* out) {
  Internal_BookKeeperLoadWithoutReset_Uint32(loc, out);
}
void ecall_Internal_BookKeeperLoadWithoutReset_Int8(int loc, int8_t* out) {
  Internal_BookKeeperLoadWithoutReset_Int8(loc, out);
}

}