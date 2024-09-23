#include "secllm.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

#include "aes_stream.h"

#include "secllm.h"

namespace {
  jpyo0803::SecLLM* secllm_ptr = nullptr;
} // namespace


void jpyo0803::SecLLM::Softmax(float* x, int B, int M, int N, int K) {
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

void jpyo0803::SecLLM::SiLU(float* x, int B, int M, int N) {
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

void jpyo0803::SecLLM::SwiGLU(float* gate_in, float* up_in, int B, int M, int N) {
  // gate_in: [B, M, N]
  // up_in: [B, M, N]
  // swiglu(gate_in, up_in) = silu(gate_in) * up_in

  jpyo0803::SecLLM::SiLU(gate_in, B, M, N);
  // gate_in is silued

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        gate_in[b * M * N + m * N + n] *= up_in[b * M * N + m * N + n];
      }
    }
  }
}

void jpyo0803::SecLLM::RMSNorm(float* x, const float* const weight, int B, int M, int N, float eps) {
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

void jpyo0803::SecLLM::ElementwiseAdd(float* x, float* y, int B, int M, int N) {
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        x[b * M * N + m * N + n] += y[b * M * N + m * N + n];
      }
    }
  }
}

void jpyo0803::SecLLM::ApplyRotaryPosEmb(float* q_tensor, float* k_tensor, const float* const cos, const float* const sin, int B, int Q_M, int K_M, int N, int K) {
/*
  shape of q = torch.Size([1, 32, 2048, 128])
  shape of k = torch.Size([1, 8, 2048, 128])
  shape of cos, sin = torch.Size([1, 2048, 128])
*/
  std::vector<float> result_buffer(N * K);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < Q_M; ++m) {
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          int k2 = (k + K / 2) % K;

          float q_cos_val = q_tensor[b * Q_M * N * K + m * N * K + n * K + k] * cos[n * K + k];
          float rh_q_sin_val = q_tensor[b * Q_M * N * K + m * N * K + n * K + k2] * sin[n * K + k];

          result_buffer.at(n * K + k) = k < K / 2 ? q_cos_val - rh_q_sin_val : q_cos_val + rh_q_sin_val;
        }
      }
      // copy result_buffer to q_tensor
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          q_tensor[b * Q_M * N * K + m * N * K + n * K + k] = result_buffer.at(n * K + k);
        }
      }
    } 

    for (int m = 0; m < K_M; ++m) {
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          int k2 = (k + K / 2) % K;

          float k_cos_val = k_tensor[b * K_M * N * K + m * N * K + n * K + k] * cos[n * K + k];
          float rh_k_sin_val = k_tensor[b * K_M * N * K + m * N * K + n * K + k2] * sin[n * K + k];

          result_buffer.at(n * K + k) = k < K / 2 ? k_cos_val - rh_k_sin_val : k_cos_val + rh_k_sin_val;
        }
      }
      // copy result_buffer to k_tensor
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          k_tensor[b * K_M * N * K + m * N * K + n * K + k] = result_buffer.at(n * K + k);
        }
      }
    }
  }
}

void jpyo0803::SecLLM::LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M, const float* const position_ids, int position_ids_M, float* cos, float* sin) {
  /*
      inv_freq: [64]
      position_ids: [1, 2048], but treat it [2048]

      cos, sin: [1, 2048, 128]
  */

  std::vector<float> half_emb_buffer(position_ids_M * inv_freq_M);

  for (int i = 0; i < position_ids_M; ++i) {
    for (int j = 0; j < inv_freq_M; ++j) {
      half_emb_buffer.at(i * inv_freq_M + j) = position_ids[i] * inv_freq[j];
    }
  }

  int col_size = inv_freq_M * 2;
  for (int i = 0; i < position_ids_M; ++i) {
    for (int j = 0; j < inv_freq_M; ++j) {
      cos[i * col_size + j] = cos[i * col_size + (inv_freq_M + j)] = std::cos(half_emb_buffer.at(i * inv_freq_M + j));
      sin[i * col_size + j] = sin[i * col_size + (inv_freq_M + j)] = std::sin(half_emb_buffer.at(i * inv_freq_M + j));
    }
  }
}

uint32_t jpyo0803::SecLLM::GenerateCPRNG() {
  uint32_t cprng;
  GetCPRNG((unsigned char*)&cprng, sizeof(cprng));
  return cprng;
}

uint32_t jpyo0803::SecLLM::GenerateMultKey() {
  uint64_t mod = 1ULL << 32;
  do {
    uint32_t key = GenerateCPRNG();
    if (std::gcd(mod, static_cast<uint64_t>(key)) == 1) {
      return key;
    }
  } while (true);
}

uint32_t jpyo0803::SecLLM::GenerateAddKey() {
  return GenerateCPRNG();
}



void Task0 () {
}

extern "C" {

void Ext_CreateSecLLM() {
  if (secllm_ptr == nullptr) {
    secllm_ptr = new jpyo0803::SecLLM();
  }
}

void Ext_Softmax(float* x, int B, int M, int N, int K) {
  secllm_ptr->Softmax(x, B, M, N, K);
}

void Ext_SwiGLU(float* gate_in, float* up_in, int B, int M, int N) {
  secllm_ptr->SwiGLU(gate_in, up_in, B, M, N);
}

void Ext_RMSNorm(float* x, const float* const weight, int B, int M, int N, float eps) {
  secllm_ptr->RMSNorm(x, weight, B, M, N, eps);
}

void Ext_ElementwiseAdd(float* x, float* y, int B, int M, int N) {
  secllm_ptr->ElementwiseAdd(x, y, B, M, N);
}

void Ext_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor, const float* const cos, const float* const sin, int B, int Q_M, int K_M, int N, int K) {
  secllm_ptr->ApplyRotaryPosEmb(q_tensor, k_tensor, cos, sin, B, Q_M, K_M, N, K);
}

void Ext_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M, const float* const position_ids, int position_ids_M, float* cos, float* sin) {
  secllm_ptr->LlamaRotaryEmbedding(inv_freq, inv_freq_M, position_ids, position_ids_M, cos, sin);
}

uint32_t Ext_GenerateCPRNG() {
  return secllm_ptr->GenerateCPRNG();
}

uint32_t Ext_GenerateMultKey() {
  return secllm_ptr->GenerateMultKey();
}

uint32_t Ext_GenerateAddKey() {
  return secllm_ptr->GenerateAddKey();
}

} // extern "C"