#include "test_interface.h"
#include "func_utils.h"

extern "C" {

void Test_Matmul_Eigen(int32_t* out, int8_t* x, int8_t* y, int B, int M, int K,
                       int N) {
  jpyo0803::Matmul_Eigen(out, x, y, B, M, K, N);
}

void Test_Matmul_Naive(int32_t* out, int8_t* x, int8_t* y, int B, int M, int K,
                       int N) {
  jpyo0803::Matmul_Naive(out, x, y, B, M, K, N);
}

void Test_GetTimeStamp_Monotonic() {
  jpyo0803::GetTimeStamp_Monotonic();
}
}
