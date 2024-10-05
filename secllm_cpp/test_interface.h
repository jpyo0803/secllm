#ifndef SECLLM_CPP_TEST_INTERFACE_H
#define SECLLM_CPP_TEST_INTERFACE_H

#include <stdint.h>

extern "C" {

void Test_Matmul_Eigen(int32_t* out, int8_t* x, int8_t* y, int B, int M, int K,
                       int N);

void Test_Matmul_Naive(int32_t* out, int8_t* x, int8_t* y, int B, int M, int K,
                       int N);

void Test_GetTimeStamp_Monotonic();
}

#endif