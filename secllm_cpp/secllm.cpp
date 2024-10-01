#include "secllm.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "aes_stream.h"

#include "func_utils.h"
#include "secllm.h"

namespace {
jpyo0803::SecLLM* secllm_ptr = nullptr;
}  // namespace

namespace jpyo0803 {

SecLLM::SecLLM(int hidden_size, int intermediate_size,
               int max_position_embeddings, int num_attention_heads,
               int num_hidden_layers, int num_key_value_heads,
               int enc_key_pool_size)
    : num_hidden_layers_(num_hidden_layers) {
  std::cout << "SecLLM is created with " << num_hidden_layers_ << " layers."
            << std::endl;
  book_keeper_ =
      std::make_unique<BookKeeper<Tensor<float>>>(num_hidden_layers_ * 100 * 3);

  book_keeper_uint32_ = std::make_unique<BookKeeper<Tensor<uint32_t>>>(
      num_hidden_layers_ * 100 * 3);

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
    decoder_layers_->emplace_back(i, hidden_size, intermediate_size,
                                  max_position_embeddings, num_attention_heads,
                                  num_key_value_heads, enc_key_pool_size);
  }
}

void SecLLM::Reset() {
  for (int i = 0; i < num_hidden_layers_; ++i) {
    decoder_layers_->at(i).Reset();
  }
}

void SecLLM::BookKeeperStore(std::vector<int> locs,
                             std::shared_ptr<Tensor<float>>& data_ptr) {
  book_keeper_->Keep(locs, data_ptr);
}

std::shared_ptr<Tensor<float>> SecLLM::BookKeeperLoad(int loc) {
  return book_keeper_->Retrieve(loc);
}

void SecLLM::BookKeeperStore_Uint32(
    std::vector<int> locs, std::shared_ptr<Tensor<uint32_t>>& data_ptr) {
  book_keeper_uint32_->Keep(locs, data_ptr);
}

std::shared_ptr<Tensor<uint32_t>> SecLLM::BookKeeperLoad_Uint32(int loc) {
  return book_keeper_uint32_->Retrieve(loc);
}

bool SecLLM::BookKeeperIsAvailable(int loc) {
  return book_keeper_->IsAvailable(loc);
}

bool SecLLM::BookKeeperIsAvailable_Uint32(int loc) {
  return book_keeper_uint32_->IsAvailable(loc);
}

void SecLLM::SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool, int* dec_key,
                                int type) {
  switch (type) {
    case 0:
      decoder_layers_->at(layer_idx).SetEncKeyAndDecKey_Q(enc_key_pool,
                                                          dec_key);
      break;
    case 1:
      decoder_layers_->at(layer_idx).SetEncKeyAndDecKey_K(enc_key_pool,
                                                          dec_key);
      break;
    case 2:
      decoder_layers_->at(layer_idx).SetEncKeyAndDecKey_V(enc_key_pool,
                                                          dec_key);
      break;
    case 3:
      decoder_layers_->at(layer_idx).SetEncKeyAndDecKey_O(enc_key_pool,
                                                          dec_key);
      break;
    case 4:
      decoder_layers_->at(layer_idx).SetEncKeyAndDecKey_Up(enc_key_pool,
                                                           dec_key);
      break;
    case 5:
      decoder_layers_->at(layer_idx).SetEncKeyAndDecKey_Gate(enc_key_pool,
                                                             dec_key);
      break;
    case 6:
      decoder_layers_->at(layer_idx).SetEncKeyAndDecKey_Down(enc_key_pool,
                                                             dec_key);
      break;
    default:
      break;
  }
}

void SecLLM::SetLinearWeightScales(int layer_idx, float* weight_scale, int len,
                                   int type) {
  switch (type) {
    case 0:
      decoder_layers_->at(layer_idx).SetLinearWeightScales_Q(weight_scale, len);
      break;
    case 1:
      decoder_layers_->at(layer_idx).SetLinearWeightScales_K(weight_scale, len);
      break;
    case 2:
      decoder_layers_->at(layer_idx).SetLinearWeightScales_V(weight_scale, len);
      break;
    case 3:
      decoder_layers_->at(layer_idx).SetLinearWeightScales_O(weight_scale, len);
      break;
    case 4:
      decoder_layers_->at(layer_idx).SetLinearWeightScales_Up(weight_scale,
                                                              len);
      break;
    case 5:
      decoder_layers_->at(layer_idx).SetLinearWeightScales_Gate(weight_scale,
                                                                len);
      break;
    case 6:
      decoder_layers_->at(layer_idx).SetLinearWeightScales_Down(weight_scale,
                                                                len);
      break;
    default:
      break;
  }
}

void SecLLM::EncryptLinearActivation(int layer_idx,
                                     std::shared_ptr<Tensor<uint32_t>> out,
                                     std::shared_ptr<Tensor<float>> in,
                                     int type) {
  switch (type) {
    case 0:
      decoder_layers_->at(layer_idx).EncryptLinearActivation_Q(out, in);
      break;
    case 1:
      decoder_layers_->at(layer_idx).EncryptLinearActivation_K(out, in);
      break;
    case 2:
      decoder_layers_->at(layer_idx).EncryptLinearActivation_V(out, in);
      break;
    case 3:
      decoder_layers_->at(layer_idx).EncryptLinearActivation_O(out, in);
      break;
    case 4:
      decoder_layers_->at(layer_idx).EncryptLinearActivation_Up(out, in);
      break;
    case 5:
      decoder_layers_->at(layer_idx).EncryptLinearActivation_Gate(out, in);
      break;
    case 6:
      decoder_layers_->at(layer_idx).EncryptLinearActivation_Down(out, in);
      break;
    default:
      break;
  }
}

void SecLLM::DecryptLinearActivation(int layer_idx,
                                     std::shared_ptr<Tensor<float>> out,
                                     int* in, int type) {
  switch (type) {
    case 0:
      decoder_layers_->at(layer_idx).DecryptLinearActivation_Q(out, in);
      break;
    case 1:
      decoder_layers_->at(layer_idx).DecryptLinearActivation_K(out, in);
      break;
    case 2:
      decoder_layers_->at(layer_idx).DecryptLinearActivation_V(out, in);
      break;
    case 3:
      decoder_layers_->at(layer_idx).DecryptLinearActivation_O(out, in);
      break;
    case 4:
      decoder_layers_->at(layer_idx).DecryptLinearActivation_Up(out, in);
      break;
    case 5:
      decoder_layers_->at(layer_idx).DecryptLinearActivation_Gate(out, in);
      break;
    case 6:
      decoder_layers_->at(layer_idx).DecryptLinearActivation_Down(out, in);
      break;
    default:
      break;
  }
}

void SecLLM::SetQKVOutputScales(int layer_idx, float q_output_scale,
                                float k_output_scale, float v_output_scale) {
  decoder_layers_->at(layer_idx).SetQKVOutputScales(
      q_output_scale, k_output_scale, v_output_scale);
}

void SecLLM::QuantizeAndShiftQ(int layer_idx,
                               std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<float>> in) {
  decoder_layers_->at(layer_idx).QuantizeAndShiftQ(out, in);
}

void SecLLM::QuantizeAndShiftK(int layer_idx,
                               std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<float>> in) {
  decoder_layers_->at(layer_idx).QuantizeAndShiftK(out, in);
}

void SecLLM::UnshiftAndDequantizeQK(int layer_idx,
                                    std::shared_ptr<Tensor<float>> out,
                                    std::shared_ptr<Tensor<uint32_t>> in) {
  decoder_layers_->at(layer_idx).UnshiftAndDequantizeQK(out, in);
}

void SecLLM::QuantizeAndShiftP(int layer_idx,
                               std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<float>> in) {
  decoder_layers_->at(layer_idx).QuantizeAndShiftP(out, in);
}

void SecLLM::QuantizeAndShiftV(int layer_idx,
                               std::shared_ptr<Tensor<uint32_t>> out,
                               std::shared_ptr<Tensor<float>> in) {
  decoder_layers_->at(layer_idx).QuantizeAndShiftV(out, in);
}

std::shared_ptr<Tensor<float>> SecLLM::UnshiftAndDequantizePV(
    int layer_idx, std::shared_ptr<Tensor<uint32_t>> in) {
  return decoder_layers_->at(layer_idx).UnshiftAndDequantizePV(in);
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

bool SecLLM::PVKeyIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVKeyGenerated();
}

bool SecLLM::PVDecKeyIsAvailable(int layer_idx) {
  return decoder_layers_->at(layer_idx).IsPVDecKeyGenerated();
}

}  // namespace jpyo0803

void Internal_PrintTest(int a, int b) {
  std::cout << "Hello from C++: " << a << " / " << b << std::endl;
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

void Internal_Softmax(int from, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);
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

  secllm_ptr->BookKeeperStore(locs, out);
}

void Internal_SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M,
                             int N) {
  jpyo0803::SwiGLU_InPlace(gate_in, up_in, B, M, N);
}

void Internal_SwiGLU(int from1, int from2, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data1 =
      secllm_ptr->BookKeeperLoad(from1);
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data2 =
      secllm_ptr->BookKeeperLoad(from2);
  auto shape = retrieved_data1->shape();

  int B = shape[0];
  int M = shape[1];
  int N = shape[2];

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape);

  jpyo0803::SwiGLU(out->data().data(), retrieved_data1->data().data(),
                   retrieved_data2->data().data(), B, M, N);

  secllm_ptr->BookKeeperStore(locs, out);
}

void Internal_RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                              int N, float eps) {
  jpyo0803::RMSNorm_InPlace(x, weight, B, M, N, eps);
}

void Internal_RMSNorm(int from, int to_len, int* to, const float* const weight,
                      float eps) {
  std::vector<int> locs(to, to + to_len);

  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);
  auto shape = retrieved_data->shape();

  int B = shape[0];
  int M = shape[1];
  int N = shape[2];

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape);

  jpyo0803::RMSNorm(out->data().data(), retrieved_data->data().data(), weight,
                    B, M, N, eps);

  secllm_ptr->BookKeeperStore(locs, out);
}

void Internal_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N) {
  jpyo0803::ElementWiseAdd_InPlace(x, y, B, M, N);
}

void Internal_ElementWiseAdd(int from1, int from2, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data1 =
      secllm_ptr->BookKeeperLoad(from1);
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data2 =
      secllm_ptr->BookKeeperLoad(from2);
  auto shape = retrieved_data1->shape();

  int B = shape[0];
  int M = shape[1];
  int N = shape[2];

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape);

  jpyo0803::ElementWiseAdd(out->data().data(), retrieved_data1->data().data(),
                           retrieved_data2->data().data(), B, M, N);

  secllm_ptr->BookKeeperStore(locs, out);
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
void Internal_BookKeeperStore(int loc, float* data, int shape_len, int* shape) {
  std::vector<int> shape_vec(shape, shape + shape_len);

  int num_elements = std::accumulate(shape_vec.begin(), shape_vec.end(), 1,
                                     std::multiplies<int>());

  std::vector<float> input_vec(data, data + num_elements);

  jpyo0803::Tensor<float> tensor(
      shape_vec,
      input_vec);  // this involves copy, so it may include some overhead

  std::shared_ptr<jpyo0803::Tensor<float>> data_ptr =
      std::make_shared<jpyo0803::Tensor<float>>(tensor);

  secllm_ptr->BookKeeperStore({loc}, data_ptr);
}

void Internal_BookKeeperStore_Uint32(int loc, int* data, int shape_len,
                                     int* shape) {
  std::vector<int> shape_vec(shape, shape + shape_len);

  int num_elements = std::accumulate(shape_vec.begin(), shape_vec.end(), 1,
                                     std::multiplies<int>());

  std::vector<uint32_t> input_vec(data, data + num_elements);

  jpyo0803::Tensor<uint32_t> tensor(shape_vec, input_vec);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> data_ptr =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(tensor);

  secllm_ptr->BookKeeperStore_Uint32({loc}, data_ptr);
}

void Internal_BookKeeperLoad(int loc, float* out, int shape_len, int* shape) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(loc);

  if (retrieved_data == nullptr) {
    throw std::runtime_error("No object at the location: " +
                             std::to_string(loc));
  }
  for (int i = 0; i < shape_len; ++i) {
    if (retrieved_data->shape()[i] != shape[i]) {
      throw std::runtime_error("Shape mismatch at the location: " +
                               std::to_string(loc));
    }
  }

  std::copy(retrieved_data->data().begin(), retrieved_data->data().end(), out);
}

void Internal_BookKeeperLoad_Uint32(int loc, int* out, int shape_len,
                                    int* shape) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(loc);

  if (retrieved_data == nullptr) {
    throw std::runtime_error("No object at the location: " +
                             std::to_string(loc));
  }
  for (int i = 0; i < shape_len; ++i) {
    if (retrieved_data->shape()[i] != shape[i]) {
      throw std::runtime_error("Shape mismatch at the location: " +
                               std::to_string(loc));
    }
  }

  std::copy(retrieved_data->data().begin(), retrieved_data->data().end(), out);
}

void Internal_ReplicateTensor(int from, int* to, int to_len) {
  std::vector<int> locs(to, to + to_len);

  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);
  // Notice, this removes a tensor in the book keeper
  secllm_ptr->BookKeeperStore(locs, retrieved_data);
}

void Internal_ReplicateTensor_Uint32(int from, int* to, int to_len) {
  std::vector<int> locs(to, to + to_len);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);
  // Notice, this removes a tensor in the book keeper
  secllm_ptr->BookKeeperStore_Uint32(locs, retrieved_data);
}

void Internal_GetCprngTensor(int* out, int shape_len, int* shape) {
  int num_elements =
      std::accumulate(shape, shape + shape_len, 1, std::multiplies<int>());

  for (int i = 0; i < num_elements; ++i) {
    out[i] = jpyo0803::GenerateCPRNG();
  }
}

void Internal_SetEncKeyAndDecKey(int layer_idx, int* src_enc_key_pool,
                                 int* src_dec_key, int type) {
  secllm_ptr->SetEncKeyAndDecKey(layer_idx, src_enc_key_pool, src_dec_key,
                                 type);
}

void Internal_SetLinearWeightScales(int layer_idx, float* scales, int len,
                                    int type) {
  // weight scales's dim == 1
  secllm_ptr->SetLinearWeightScales(layer_idx, scales, len, type);
}

void Internal_EncryptLinearActivation(int layer_idx, int from,
                                      std::vector<int> locs, int type) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptLinearActivation(layer_idx, out, retrieved_data, type);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_DecryptLinearActivation(int layer_idx, int to_len, int* to,
                                      int* enc_tensor, int shape_len,
                                      int* shape, int type) {
  std::vector<int> locs{to, to + to_len};

  std::vector<int> shape_vec(shape, shape + shape_len);

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape_vec);

  secllm_ptr->DecryptLinearActivation(layer_idx, out, enc_tensor, type);

  secllm_ptr->BookKeeperStore(locs, out);
}

void Internal_SetQKVOutputScales(int layer_idx, float q_output_scale,
                                 float k_output_scale, float v_output_scale) {
  secllm_ptr->SetQKVOutputScales(layer_idx, q_output_scale, k_output_scale,
                                 v_output_scale);
}

void Internal_QuantizeAndShiftQ(int layer_idx, int from,
                                std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeAndShiftQ(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_QuantizeAndShiftK(int layer_idx, int from,
                                std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeAndShiftK(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_UnshiftAndDequantizeQK(int layer_idx, int from,
                                     std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(retrieved_data->shape());

  secllm_ptr->UnshiftAndDequantizeQK(layer_idx, out, retrieved_data);

  // out->PrintAsTorchStyle();
  // out->PrintCharacteristics();
  // std::cout << out->GetMean() << std::endl;
  // std::cout << out->PosDepSum() << std::endl;
  // exit(-1);

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

  secllm_ptr->BookKeeperStore(locs, out);
}

void Internal_QuantizeAndShiftP(int layer_idx, int from,
                                std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeAndShiftP(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_QuantizeAndShiftV(int layer_idx, int from,
                                std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->QuantizeAndShiftV(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_UnshiftAndDequantizePV(int layer_idx, int from,
                                     std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  auto out = secllm_ptr->UnshiftAndDequantizePV(layer_idx, retrieved_data);
  secllm_ptr->BookKeeperStore(locs, out);
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

void Internal_EncryptX_QK(int layer_idx, int from, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptX_QK(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_EncryptY_QK(int layer_idx, int from, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptY_QK(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_Decrypt_QK(int layer_idx, int from, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->Decrypt_QK(layer_idx, out, retrieved_data);

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

void Internal_EncryptX_PV(int layer_idx, int from, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptX_PV(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_EncryptY_PV(int layer_idx, int from, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->EncryptY_PV(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_Decrypt_PV(int layer_idx, int from, std::vector<int> locs) {
  std::shared_ptr<jpyo0803::Tensor<uint32_t>> retrieved_data =
      secllm_ptr->BookKeeperLoad_Uint32(from);

  std::shared_ptr<jpyo0803::Tensor<uint32_t>> out =
      std::make_shared<jpyo0803::Tensor<uint32_t>>(retrieved_data->shape());

  secllm_ptr->Decrypt_PV(layer_idx, out, retrieved_data);

  secllm_ptr->BookKeeperStore_Uint32(locs, out);
}

void Internal_BookKeeperIsAvailable(int loc, bool* ret) {
  *ret = secllm_ptr->BookKeeperIsAvailable(loc);
}

void Internal_BookKeeperIsAvailable_Uint32(int loc, bool* ret) {
  *ret = secllm_ptr->BookKeeperIsAvailable_Uint32(loc);
}

void Internal_QKKeyIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKKeyIsAvailable(layer_idx);
}

void Internal_QKDecKeyIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->QKDecKeyIsAvailable(layer_idx);
}

void Internal_PVKeyIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVKeyIsAvailable(layer_idx);
}

void Internal_PVDecKeyIsAvailable(int layer_idx, bool* ret) {
  *ret = secllm_ptr->PVDecKeyIsAvailable(layer_idx);
}