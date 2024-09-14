#ifndef SECLLM_SECLLM_H
#define SECLLM_SECLLM_H

#include <stdint.h>

extern "C" {

void PrintHelloFromCpp();

void Softmax(float* x, int B, int M, int N, int K);

void SiLU(float* x, int B, int M, int N);

void SwiGLU(float* gate_in, float* up_in, int B, int M, int N);

void RMSNorm(float* x, const float* const weight, int B, int M, int N, float eps);

void ElementwiseAdd(float* x, float* y, int B, int M, int N);

void ApplyRotaryPosEmb(float* q_tensor, float* k_tensor, const float* const cos, const float* const sin, int B, int Q_M, int K_M, int N, int K);

void LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M, const float* const position_ids, int position_ids_M, float* cos, float* sin);

uint32_t GenerateCPRNG();

uint32_t GenerateMultKey();

uint32_t GenerateAddKey();

} // extern "C"

#endif // SECLLM_SECLLM_H