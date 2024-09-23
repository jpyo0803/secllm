#ifndef SECLLM_SECLLM_H
#define SECLLM_SECLLM_H

#include <stdint.h>
#include <memory>

#include "book_keeper.h"
#include "tensor.h"

namespace jpyo0803 {

class SecLLM {
  public:
    explicit SecLLM(int num_layers);

  public: 
    // member methods
    void BookKeeperStore(int loc, std::shared_ptr<Tensor<float>>& data);

    std::shared_ptr<Tensor<float>> BookKeeperLoad(int loc);

    void TestPrint();

  public:
    // static methods
    static void Softmax(float* x, int B, int M, int N, int K);

    static void SiLU(float* x, int B, int M, int N);

    static void SwiGLU(float* gate_in, float* up_in, int B, int M, int N);

    static void RMSNorm(float* x, const float* const weight, int B, int M, int N, float eps);

    static void ElementwiseAdd(float* x, float* y, int B, int M, int N);

    static void ApplyRotaryPosEmb(float* q_tensor, float* k_tensor, const float* const cos, const float* const sin, int B, int Q_M, int K_M, int N, int K);

    static void LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M, const float* const position_ids, int position_ids_M, float* cos, float* sin);

    static uint32_t GenerateCPRNG();

    static uint32_t GenerateMultKey();

    static uint32_t GenerateAddKey();

  private:
    int num_layers_ = 0;

    std::unique_ptr<BookKeeper<Tensor<float>>> book_keepers_;
};

} // namespace jpyo0803

extern "C" {

void Ext_CreateSecLLM(int num_layers);

void Ext_Softmax(float* x, int B, int M, int N, int K);

void Ext_SwiGLU(float* gate_in, float* up_in, int B, int M, int N);

void Ext_RMSNorm(float* x, const float* const weight, int B, int M, int N, float eps);

void Ext_ElementwiseAdd(float* x, float* y, int B, int M, int N);

void Ext_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor, const float* const cos, const float* const sin, int B, int Q_M, int K_M, int N, int K);

void Ext_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M, const float* const position_ids, int position_ids_M, float* cos, float* sin);

uint32_t Ext_GenerateCPRNG();

uint32_t Ext_GenerateMultKey();

uint32_t Ext_GenerateAddKey();



} // extern "C"

#endif // SECLLM_SECLLM_H