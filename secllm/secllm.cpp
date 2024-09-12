#include "secllm.h"

#include <iostream>
#include <cmath>

extern "C" {

void PrintHelloFromCpp() {
  std::cout << "Hello from C++!" << std::endl;
}

void Softmax(float* x, int B, int M, int N, int K) {
  // x: [B, M, N, K]

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float max_val = x[b * M * N * K + m * N * K + n * K];
        for (int k = 1; k < K; ++k) {
          max_val = std::max(max_val, x[b * M * N * K + m * N * K + n * K + k]);
        }

        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          x[b * M * N * K + m * N * K + n * K + k] = std::exp(x[b * M * N * K + m * N * K + n * K + k] - max_val);
          sum += x[b * M * N * K + m * N * K + n * K + k];
        }

        for (int k = 0; k < K; ++k) {
          x[b * M * N * K + m * N * K + n * K + k] /= sum;
        }
      }
    }
  }
}

void SiLU(float* x, int B, int M, int N) {
  // x: [B, M, N]
  // silu(x) = x / (1 + exp(-x))
  
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        x[b * M * N + m * N + n] = x[b * M * N + m * N + n] / (1.0f + std::exp(-x[b * M * N + m * N + n]));
      }
    }
  }
}

void SwiGLU(float* gate_in, float* up_in, int B, int M, int N) {
  // gate_in: [B, M, N]
  // up_in: [B, M, N]
  // swiglu(gate_in, up_in) = silu(gate_in) * up_in

  SiLU(gate_in, B, M, N);
  // gate_in is silued

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        gate_in[b * M * N + m * N + n] *= up_in[b * M * N + m * N + n];
      }
    }
  }
}

void RMSNorm(float* x, const float* const weight, int B, int M, int N, float eps) {
  // weight, x: [B, M, N]

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      float sqr_sum = 0.0f;
      for (int n = 0; n < N; ++n) {
        sqr_sum += x[b * M * N + m * N + n] * x[b * M * N + m * N + n];
      }

      float variance = sqr_sum / N;

      for (int n = 0; n < N; ++n) {
        x[b * M * N + m * N + n] /= std::sqrt(variance + eps);
        x[b * M * N + m * N + n] *= weight[n];
      }
    }
  }
}

void ElementwiseAdd(float* x, float* y, int B, int M, int N) {
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        x[b * M * N + m * N + n] += y[b * M * N + m * N + n];
      }
    }
  }
}

}