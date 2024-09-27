#ifndef SECLLM_CPP_FUNC_UTILS_H
#define SECLLM_CPP_FUNC_UTILS_H

#include <stdint.h>
#include <utility>
#include <vector>
#include "aes_stream.h"

namespace jpyo0803 {
std::pair<std::vector<int8_t>, std::vector<float>>
DynamicQuantizeActivationPerTokenAbsmax(const std::vector<float>& t, size_t B,
                                        size_t M, size_t N);

void DequantizeActivationWPerChannelAPerChannel(
    float* out,
    int* q_act,                          // Quantized activations (B x dim)
    const std::vector<float>& w_scales,  // Weight scales (dim)
    const std::vector<float>& a_scales,  // Activation scales (B)
    size_t B,                            // Batch size
    size_t dim                           // Dimension
);

void Softmax_InPlace(float* x, int B, int M, int N, int K);

void Softmax(float* out, float* in, int B, int M, int N, int K);

void SiLU(float* x, int B, int M, int N);

void SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N);

void SwiGLU(float* out, float* gate_in, float* up_in, int B, int M, int N);

void RMSNorm_InPlace(float* x, const float* const weight, int B, int M, int N,
                     float eps);

void RMSNorm(float* out, float* in, const float* const weight, int B, int M,
             int N, float eps);

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
}  // namespace jpyo0803

#endif  // SECLLM_CPP_FUNC_UTILS_H