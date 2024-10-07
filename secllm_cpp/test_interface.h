#ifndef SECLLM_CPP_TEST_INTERFACE_H
#define SECLLM_CPP_TEST_INTERFACE_H

#include <stdint.h>

extern "C" {

void Test_Matmul_Eigen(int32_t* out, int8_t* x, int8_t* y, int B, int X_M,
                       int X_N, int Y_M, int Y_N, bool transpose);

void Test_Matmul_Naive(int32_t* out, int8_t* x, int8_t* y, int B, int M, int K,
                       int N, bool transpose_y);

void Test_RepeatKV(int8_t* out, int8_t* hidden_states, int batch,
                   int num_key_value_heads, int seqlen, int head_dim,
                   int n_rep);

void Test_GetTimeStamp_Monotonic();
}

#endif