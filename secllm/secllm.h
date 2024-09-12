#ifndef SECLLM_SECLLM_H
#define SECLLM_SECLLM_H

extern "C" {

void PrintHelloFromCpp();

void Softmax(float* x, int B, int M, int N, int K);

void SiLU(float* x, int B, int M, int N);

void SwiGLU(float* gate_in, float* up_in, int B, int M, int N);

} // extern "C"

#endif // SECLLM_SECLLM_H