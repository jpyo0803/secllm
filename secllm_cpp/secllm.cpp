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

jpyo0803::SecLLM::SecLLM(int hidden_size, int intermediate_size,
                         int max_position_embeddings, int num_attention_heads,
                         int num_hidden_layers, int num_key_value_heads,
                         int enc_key_pool_size)
    : num_hidden_layers_(num_hidden_layers) {
  std::cout << "SecLLM is created with " << num_hidden_layers_ << " layers."
            << std::endl;
  book_keeper_ =
      std::make_unique<BookKeeper<Tensor<float>>>(num_hidden_layers_ * 100 * 3);

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

void jpyo0803::SecLLM::BookKeeperStore(
    std::vector<int> locs, std::shared_ptr<jpyo0803::Tensor<float>>& data_ptr) {
  book_keeper_->Keep(locs, data_ptr);
}

std::shared_ptr<jpyo0803::Tensor<float>> jpyo0803::SecLLM::BookKeeperLoad(
    int loc) {
  return book_keeper_->Retrieve(loc);
}

void jpyo0803::SecLLM::SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool,
                                          int* dec_key, int type) {
  switch (type) {
    case 0:
      decoder_layers_->at(layer_idx).SetEncKeyAndDecKey_Q(enc_key_pool,
                                                          dec_key);
      break;
    default:
      break;
  }
}

void jpyo0803::SecLLM::SetLinearWeightScales(int layer_idx, float* weight_scale,
                                             int len, int type) {
  switch (type) {
    case 0:
      decoder_layers_->at(layer_idx).SetLinearWeightScales_Q(weight_scale, len);
      break;
    default:
      break;
  }
}

void jpyo0803::SecLLM::EncryptLinearActivation(
    int layer_idx, int* out, std::shared_ptr<Tensor<float>> in, int type) {
  switch (type) {
    case 0:
      decoder_layers_->at(layer_idx).EncryptLinearActivation_Q(out, in);
      break;
    default:
      break;
  }
}

void jpyo0803::SecLLM::DecryptLinearActivation(
    int layer_idx, std::shared_ptr<Tensor<float>> out, int* in, int type) {
  switch (type) {
    case 0:
      decoder_layers_->at(layer_idx).DecryptLinearActivation_Q(out, in);
      break;
    default:
      break;
  }
}

extern "C" {

void Ext_PrintTest(int a, int b) {
  std::cout << "Hello from C++: " << a << " / " << b << std::endl;
}

void Ext_CreateSecLLM(int hidden_size, int intermediate_size,
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

void Ext_Softmax_InPlace(float* x, int B, int M, int N, int K) {
  jpyo0803::Softmax_InPlace(x, B, M, N, K);
}

void Ext_Softmax(int from, int to) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);
  auto shape = retrieved_data->shape();

  int B = shape[0];
  int M = shape[1];
  int N = shape[2];
  int K = shape[3];

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape);

  jpyo0803::Softmax(out->data().data(), retrieved_data->data().data(), B, M, N,
                    K);
  secllm_ptr->BookKeeperStore({to}, out);
}

void Ext_SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N) {
  jpyo0803::SwiGLU_InPlace(gate_in, up_in, B, M, N);
}

void Ext_SwiGLU(int from1, int from2, int to) {
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

  secllm_ptr->BookKeeperStore({to}, out);
}

void Ext_RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                         int N, float eps) {
  jpyo0803::RMSNorm_InPlace(x, weight, B, M, N, eps);
}

void Ext_RMSNorm(int from, int to, const float* const weight, float eps) {
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

  secllm_ptr->BookKeeperStore({to}, out);
}

void Ext_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N) {
  jpyo0803::ElementWiseAdd_InPlace(x, y, B, M, N);
}

void Ext_ElementWiseAdd(int from1, int from2, int to) {
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

  secllm_ptr->BookKeeperStore({to}, out);
}

void Ext_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                           const float* const cos, const float* const sin,
                           int B, int Q_M, int K_M, int N, int K) {
  jpyo0803::ApplyRotaryPosEmb(q_tensor, k_tensor, cos, sin, B, Q_M, K_M, N, K);
}

void Ext_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                              const float* const position_ids,
                              int position_ids_M, float* cos, float* sin) {
  jpyo0803::LlamaRotaryEmbedding(inv_freq, inv_freq_M, position_ids,
                                 position_ids_M, cos, sin);
}

uint32_t Ext_GenerateCPRNG() {
  return jpyo0803::GenerateCPRNG();
}

uint32_t Ext_GenerateMultKey() {
  return jpyo0803::GenerateMultKey();
}

uint32_t Ext_GenerateAddKey() {
  return jpyo0803::GenerateAddKey();
}

void Ext_BookKeeperStore(int loc, float* data, int shape_len, int* shape) {
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

void Ext_BookKeeperLoad(int loc, float* out, int shape_len, int* shape) {
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

void Ext_ReplicateTensor(int from, int* to, int to_len) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);
  // Notice, this removes a tensor in the book keeper

  std::vector<int> locs(to, to + to_len);
  secllm_ptr->BookKeeperStore(locs, retrieved_data);
}

void Ext_GetCprngTensor(int* out, int shape_len, int* shape) {
  int num_elements =
      std::accumulate(shape, shape + shape_len, 1, std::multiplies<int>());

  for (int i = 0; i < num_elements; ++i) {
    out[i] = jpyo0803::GenerateCPRNG();
  }
}

void Ext_SetEncKeyAndDecKey(int layer_idx, int* src_enc_key_pool,
                            int* src_dec_key, int type) {
  secllm_ptr->SetEncKeyAndDecKey(layer_idx, src_enc_key_pool, src_dec_key,
                                 type);
}

void Ext_SetLinearWeightScales(int layer_idx, float* scales, int len,
                               int type) {
  // weight scales's dim == 1
  secllm_ptr->SetLinearWeightScales(layer_idx, scales, len, type);
}

void Ext_EncryptLinearActivation(int layer_idx, int* out, int from, int type) {
  std::shared_ptr<jpyo0803::Tensor<float>> retrieved_data =
      secllm_ptr->BookKeeperLoad(from);

  secllm_ptr->EncryptLinearActivation(layer_idx, out, retrieved_data, type);
}

void Ext_DecryptLinearActivation(int layer_idx, int to, int* enc_tensor,
                                 int shape_len, int* shape, int type) {
  std::vector<int> shape_vec(shape, shape + shape_len);

  std::shared_ptr<jpyo0803::Tensor<float>> out =
      std::make_shared<jpyo0803::Tensor<float>>(shape_vec);

  secllm_ptr->DecryptLinearActivation(layer_idx, out, enc_tensor, type);

  secllm_ptr->BookKeeperStore({to}, out);
}
}  // extern "C"