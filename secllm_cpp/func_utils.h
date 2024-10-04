#ifndef SECLLM_CPP_FUNC_UTILS_H
#define SECLLM_CPP_FUNC_UTILS_H

#include <stdint.h>
#include <utility>
#include <vector>
#include "aes_stream.h"

namespace jpyo0803 {
std::pair<std::vector<int8_t>, std::vector<float>>
DynamicQuantizeActivationPerTokenAbsmax(const std::vector<float>& in, int B,
                                        int M, int N);

void DequantizeActivationWPerChannelAPerChannel(
    std::vector<float>& out,
    const std::vector<int32_t>& in,      // Quantized activations (B x dim)
    const std::vector<float>& w_scales,  // Weight scales (dim)
    const std::vector<float>& a_scales,  // Activation scales (B)
    size_t B,                            // Batch size
    size_t dim                           // Dimension
);

void QuantizeActivationPerTensor(std::vector<int8_t>& out,
                                 const std::vector<float>& in, int64_t len,
                                 float scale);

void DequantizeActivationPerTensor(std::vector<float>& out,
                                   const std::vector<int32_t>& in, int64_t len,
                                   float scale);

void Softmax_InPlace(float* x, int B, int M, int N, int K);

void Softmax(float* out, float* in, int B, int M, int N, int K);

void SiLU(float* x, int B, int M, int N);

void SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N);

void SwiGLU(float* out, float* gate_in, float* up_in, int B, int M, int N);

void RMSNorm_InPlace(float* x, const float* const weight, int B, int M, int N,
                     float eps);

void RMSNorm_Func(float* out, float* in, const float* const weight, int B,
                  int M, int N, float eps);

void ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N);

void ElementWiseAdd(float* out, float* x, float* y, int B, int M, int N);

void ElementWiseSubtract(float* out, float* x, float* y, int B, int M, int N);

void ApplyRotaryPosEmb(float* q_tensor, float* k_tensor, const float* const cos,
                       const float* const sin, int B, int Q_M, int K_M, int N,
                       int K);

void LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                          const float* const position_ids, int position_ids_M,
                          float* cos, float* sin);

uint32_t GenerateCPRNG();

uint32_t GenerateMultKey();

uint32_t GenerateAddKey();

uint64_t RepeatedSqr(uint64_t base, uint64_t exp, uint64_t mod);

void Matmul(int32_t* out, int8_t* x, int8_t* y, int B, int M, int K, int N);

void GetTimeStamp_Monotonic();

}  // namespace jpyo0803

#endif  // SECLLM_CPP_FUNC_UTILS_H