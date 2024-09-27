#include "func_utils.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {
// Helper function to find the max absolute value for each row in the flattened vector.
std::vector<float> max_abs_per_token(const std::vector<float>& t, size_t B,
                                     size_t M, size_t N) {
  std::vector<float> max_vals(B * M, 0.0);  // Store max values for each row

  for (size_t b = 0; b < B; ++b) {
    for (size_t m = 0; m < M; ++m) {
      float max_val = 0.0;
      for (size_t n = 0; n < N; ++n) {
        size_t index =
            b * M * N + m * N + n;  // Calculate 1D index for (b, m, n)
        max_val = std::max(max_val, std::abs(t[index]));
      }
      max_vals[b * M + m] =
          std::max(max_val, 1e-8f) / 127.0f;  // Clamp and normalize
    }
  }

  return max_vals;
}
}  // namespace

std::pair<std::vector<int8_t>, std::vector<float>>
jpyo0803::DynamicQuantizeActivationPerTokenAbsmax(const std::vector<float>& t,
                                                  size_t B, size_t M,
                                                  size_t N) {
  std::vector<int8_t> q_act(B * M * N);  // Quantized result in flattened form
  std::vector<float> max_vals =
      max_abs_per_token(t, B, M, N);  // Max values per row

  // Quantize the matrix
  for (size_t b = 0; b < B; ++b) {
    for (size_t m = 0; m < M; ++m) {
      float max_val = max_vals[b * M + m];  // Get max value for the row
      for (size_t n = 0; n < N; ++n) {
        size_t index =
            b * M * N + m * N + n;       // Calculate 1D index for (b, m, n)
        float val = t[index] / max_val;  // Normalize
        val = std::round(val);           // Round
        val = std::clamp(val, -128.0f, 127.0f);   // Clamp between -128 and 127
        q_act[index] = static_cast<int8_t>(val);  // Store quantized value
      }
    }
  }

  return {q_act, max_vals};
}

// Function to dequantize the activations

void jpyo0803::DequantizeActivationWPerChannelAPerChannel(
    float* out,
    int* q_act,                          // Quantized activations (B x dim)
    const std::vector<float>& w_scales,  // Weight scales (dim)
    const std::vector<float>& a_scales,  // Activation scales (B)
    size_t B,                            // Batch size
    size_t dim                           // Dimension
) {
  // Iterate through the batch and the dimension
  for (size_t b = 0; b < B; ++b) {
    for (size_t d = 0; d < dim; ++d) {
      size_t index = b * dim + d;  // Calculate the 1D index for (b, d)

      // Convert q_act to float, and apply the per-channel and per-token scaling
      float q_val = static_cast<float>(q_act[index]);
      out[index] = q_val * w_scales[d] * a_scales[b];
    }
  }
}

void jpyo0803::Softmax_InPlace(float* x, int B, int M, int N, int K) {
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
          x[b * M * N * K + m * N * K + n * K + k] =
              std::exp(x[b * M * N * K + m * N * K + n * K + k] - max_val);
          sum += x[b * M * N * K + m * N * K + n * K + k];
        }

        for (int k = 0; k < K; ++k) {
          x[b * M * N * K + m * N * K + n * K + k] /= sum;
        }
      }
    }
  }
}

void jpyo0803::Softmax(float* out, float* in, int B, int M, int N, int K) {
  // x: [B, M, N, K]

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float max_val = in[b * M * N * K + m * N * K + n * K];
        for (int k = 1; k < K; ++k) {
          max_val =
              std::max(max_val, in[b * M * N * K + m * N * K + n * K + k]);
        }

        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          out[b * M * N * K + m * N * K + n * K + k] =
              std::exp(in[b * M * N * K + m * N * K + n * K + k] - max_val);
          sum += out[b * M * N * K + m * N * K + n * K + k];
        }

        for (int k = 0; k < K; ++k) {
          out[b * M * N * K + m * N * K + n * K + k] /= sum;
        }
      }
    }
  }
}

void jpyo0803::SiLU(float* x, int B, int M, int N) {
  // x: [B, M, N]
  // silu(x) = x / (1 + exp(-x))

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        x[b * M * N + m * N + n] = x[b * M * N + m * N + n] /
                                   (1.0f + std::exp(-x[b * M * N + m * N + n]));
      }
    }
  }
}

void jpyo0803::SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M,
                              int N) {
  // gate_in: [B, M, N]
  // up_in: [B, M, N]
  // SwiGLU_InPlace(gate_in, up_in) = silu(gate_in) * up_in

  jpyo0803::SiLU(gate_in, B, M, N);
  // gate_in is silued

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        gate_in[b * M * N + m * N + n] *= up_in[b * M * N + m * N + n];
      }
    }
  }
}

void jpyo0803::SwiGLU(float* out, float* gate_in, float* up_in, int B, int M,
                      int N) {
  // gate_in: [B, M, N]
  // up_in: [B, M, N]
  // SwiGLU(gate_in, up_in) = silu(gate_in) * up_in

  jpyo0803::SiLU(gate_in, B, M, N);
  // gate_in is silued

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            gate_in[b * M * N + m * N + n] * up_in[b * M * N + m * N + n];
      }
    }
  }
}

void jpyo0803::RMSNorm_InPlace(float* x, const float* const weight, int B,
                               int M, int N, float eps) {
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

void jpyo0803::RMSNorm(float* out, float* in, const float* const weight, int B,
                       int M, int N, float eps) {
  // weight, x: [B, M, N]

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      float sqr_sum = 0.0f;
      for (int n = 0; n < N; ++n) {
        sqr_sum += in[b * M * N + m * N + n] * in[b * M * N + m * N + n];
      }

      float variance = sqr_sum / N;

      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            in[b * M * N + m * N + n] / std::sqrt(variance + eps);
        out[b * M * N + m * N + n] *= weight[n];
      }
    }
  }
}

void jpyo0803::ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N) {
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        x[b * M * N + m * N + n] += y[b * M * N + m * N + n];
      }
    }
  }
}

void jpyo0803::ElementWiseAdd(float* out, float* x, float* y, int B, int M,
                              int N) {
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            x[b * M * N + m * N + n] + y[b * M * N + m * N + n];
      }
    }
  }
}

void jpyo0803::ElementWiseSubtract(float* out, float* x, float* y, int B, int M,
                                   int N) {
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            x[b * M * N + m * N + n] - y[b * M * N + m * N + n];
      }
    }
  }
}

void jpyo0803::ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                                 const float* const cos, const float* const sin,
                                 int B, int Q_M, int K_M, int N, int K) {
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

          float q_cos_val = q_tensor[b * Q_M * N * K + m * N * K + n * K + k] *
                            cos[n * K + k];
          float rh_q_sin_val =
              q_tensor[b * Q_M * N * K + m * N * K + n * K + k2] *
              sin[n * K + k];

          result_buffer.at(n * K + k) =
              k < K / 2 ? q_cos_val - rh_q_sin_val : q_cos_val + rh_q_sin_val;
        }
      }
      // copy result_buffer to q_tensor
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          q_tensor[b * Q_M * N * K + m * N * K + n * K + k] =
              result_buffer.at(n * K + k);
        }
      }
    }

    for (int m = 0; m < K_M; ++m) {
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          int k2 = (k + K / 2) % K;

          float k_cos_val = k_tensor[b * K_M * N * K + m * N * K + n * K + k] *
                            cos[n * K + k];
          float rh_k_sin_val =
              k_tensor[b * K_M * N * K + m * N * K + n * K + k2] *
              sin[n * K + k];

          result_buffer.at(n * K + k) =
              k < K / 2 ? k_cos_val - rh_k_sin_val : k_cos_val + rh_k_sin_val;
        }
      }
      // copy result_buffer to k_tensor
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          k_tensor[b * K_M * N * K + m * N * K + n * K + k] =
              result_buffer.at(n * K + k);
        }
      }
    }
  }
}

void jpyo0803::LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                                    const float* const position_ids,
                                    int position_ids_M, float* cos,
                                    float* sin) {
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
      cos[i * col_size + j] = cos[i * col_size + (inv_freq_M + j)] =
          std::cos(half_emb_buffer.at(i * inv_freq_M + j));
      sin[i * col_size + j] = sin[i * col_size + (inv_freq_M + j)] =
          std::sin(half_emb_buffer.at(i * inv_freq_M + j));
    }
  }
}

uint32_t jpyo0803::GenerateCPRNG() {
  uint32_t cprng;
  GetCPRNG((unsigned char*)&cprng, sizeof(cprng));
  return cprng;
}

uint32_t jpyo0803::GenerateMultKey() {
  uint64_t mod = 1ULL << 32;
  do {
    uint32_t key = GenerateCPRNG();
    if (std::gcd(mod, static_cast<uint64_t>(key)) == 1) {
      return key;
    }
  } while (true);
}

uint32_t jpyo0803::GenerateAddKey() {
  return GenerateCPRNG();
}
