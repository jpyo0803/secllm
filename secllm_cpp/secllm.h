#ifndef SECLLM_SECLLM_H
#define SECLLM_SECLLM_H

#include <stdint.h>
#include <memory>

#include "book_keeper.h"
#include "tensor.h"

namespace jpyo0803 {

class SecLLM {
 public:
  SecLLM(int hidden_size, int intermediate_size, int max_position_embeddings,
         int num_attention_heads, int num_hidden_layers,
         int num_key_value_heads);

 public:
  // member methods
  void BookKeeperStore(std::vector<int> locs,
                       std::shared_ptr<Tensor<float>>& data);

  std::shared_ptr<Tensor<float>> BookKeeperLoad(int loc);

  void TestPrint();

 public:
  // static methods
  static void Softmax_InPlace(float* x, int B, int M, int N, int K);

  static void Softmax(float* out, float* in, int B, int M, int N, int K);

  static void SiLU(float* x, int B, int M, int N);

  static void SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N);

  static void SwiGLU(float* out, float* gate_in, float* up_in, int B, int M,
                     int N);

  static void RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                              int N, float eps);

  static void RMSNorm(float* out, float* in, const float* const weight, int B,
                      int M, int N, float eps);

  static void ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N);

  static void ElementWiseAdd(float* out, float* x, float* y, int B, int M,
                             int N);

  static void ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                                const float* const cos, const float* const sin,
                                int B, int Q_M, int K_M, int N, int K);

  static void LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                                   const float* const position_ids,
                                   int position_ids_M, float* cos, float* sin);

  static uint32_t GenerateCPRNG();

  static uint32_t GenerateMultKey();

  static uint32_t GenerateAddKey();

 private:
  int hidden_size_ = 0;
  int intermediate_size_ = 0;
  int max_position_embeddings_ = 0;
  int num_attention_heads_ = 0;
  int num_hidden_layers_ = 0;
  int num_key_value_heads_ = 0;

  std::unique_ptr<BookKeeper<Tensor<float>>> book_keepers_;
};

}  // namespace jpyo0803

extern "C" {

void Ext_PrintTest(int a, int b);

void Ext_CreateSecLLM(int hidden_size, int intermediate_size,
                      int max_position_embeddings, int num_attention_heads,
                      int num_hidden_layers, int num_key_value_heads);

void Ext_Softmax_InPlace(float* x, int B, int M, int N, int K);

void Ext_Softmax(int from, int to);

void Ext_SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N);

void Ext_SwiGLU(int from1, int from2, int to);

void Ext_RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                         int N, float eps);

void Ext_RMSNorm(int from, int to, const float* const weight, float eps);

void Ext_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N);

void Ext_ElementWiseAdd(int from1, int from2, int to);

void Ext_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                           const float* const cos, const float* const sin,
                           int B, int Q_M, int K_M, int N, int K);

void Ext_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                              const float* const position_ids,
                              int position_ids_M, float* cos, float* sin);

uint32_t Ext_GenerateCPRNG();

uint32_t Ext_GenerateMultKey();

uint32_t Ext_GenerateAddKey();

void Ext_ReplicateTensor(int from, int* to, int to_len);

}  // extern "C"

#endif  // SECLLM_SECLLM_H