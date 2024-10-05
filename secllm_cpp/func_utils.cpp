#include "func_utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "Eigen/Dense"
#include "macro.h"

using namespace Eigen;

namespace {
// Helper function to find the max absolute value for each row in the flattened vector.
std::vector<float> max_abs_per_token(const std::vector<float>& t, size_t B,
                                     size_t M, size_t N) {
  std::vector<float> max_vals(B * M, 0.0);  // Store max values for each row

  for (size_t b = 0; b < B; ++b) {
    for (size_t m = 0; m < M; ++m) {
      // Map the current row to Eigen
      Eigen::Map<const Eigen::MatrixXf> row_vec(t.data() + b * M * N + m * N, 1,
                                                N);
      // Compute max absolute value in the row
      float max_val = row_vec.cwiseAbs().maxCoeff();
      // Store the normalized and clamped max value
      max_vals[b * M + m] = std::max(max_val, 1e-8f) / 127.0f;
    }
  }

  return max_vals;
}
}  // namespace

void jpyo0803::QuantizeActivationPerTensor(std::vector<int8_t>& out,
                                           const std::vector<float>& in,
                                           int64_t len, float scale) {

  // Quantize the matrix
  for (int64_t i = 0; i < len; ++i) {
    // Convert t to int8_t, and apply the scaling
    float q_val = in.at(i) / scale;
    q_val = std::round(q_val);                   // Round
    q_val = std::clamp(q_val, -128.0f, 127.0f);  // Clamp between -128 and 127
    out.at(i) = static_cast<int8_t>(q_val);      // Store quantized value
  }
}

void jpyo0803::DequantizeActivationPerTensor(std::vector<float>& out,
                                             const std::vector<int32_t>& in,
                                             int64_t len, float scale) {
  // Iterate through the batch and the dimension
  for (int64_t i = 0; i < len; ++i) {
    // Convert t to float, and apply the scaling
    out[i] = static_cast<float>(in[i]) * scale;
  }
}

std::pair<std::vector<int8_t>, std::vector<float>>
jpyo0803::DynamicQuantizeActivationPerTokenAbsmax(const std::vector<float>& in,
                                                  int B, int M, int N) {
  std::vector<int8_t> q_act(B * M * N);  // Quantized result in flattened form
  std::vector<float> max_vals =
      max_abs_per_token(in, B, M, N);  // Max values per row

  // Quantize the matrix
  for (size_t b = 0; b < B; ++b) {
    for (size_t m = 0; m < M; ++m) {
      float max_val = max_vals[b * M + m];  // Get max value for the row

      // Map the row of `in` and `q_act` for Eigen
      Eigen::Map<const Eigen::MatrixXf> in_vec(in.data() + b * M * N + m * N, 1,
                                               N);
      Eigen::Map<Eigen::Matrix<int8_t, 1, Eigen::Dynamic>> q_act_vec(
          q_act.data() + b * M * N + m * N, 1, N);

      // Perform quantization using Eigen
      Eigen::MatrixXf normalized =
          (in_vec / max_val).array().round().cwiseMin(127.0f).cwiseMax(-128.0f);
      q_act_vec = normalized.cast<int8_t>();
    }
  }

  return {q_act, max_vals};
}

// Function to dequantize the activations

void jpyo0803::DequantizeActivationWPerChannelAPerChannel(
    std::vector<float>& out,
    const std::vector<int32_t>& in,      // Quantized activations (B x dim)
    const std::vector<float>& w_scales,  // Weight scales (dim)
    const std::vector<float>& a_scales,  // Activation scales (B)
    size_t B,                            // Batch size
    size_t dim                           // Dimension
) {
  for (size_t b = 0; b < B; ++b) {
    for (size_t d = 0; d < dim; ++d) {
      size_t index = b * dim + d;  // Calculate the 1D index for (b, d)

      // Convert q_act to float, and apply the per-channel and per-token scaling

      // Becareful data is represented in uint32_t but it is actually int32
      int tmp = static_cast<int>(in[index]);
      float q_val = static_cast<float>(tmp);
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
#if INTERNAL_TIME_MEASURE == 1
  auto start = std::chrono::steady_clock::now();
#endif

  // for (int b = 0; b < B; ++b) {
  //   for (int m = 0; m < M; ++m) {
  //     for (int n = 0; n < N; ++n) {
  //       float max_val = in[b * M * N * K + m * N * K + n * K];
  //       for (int k = 1; k < K; ++k) {
  //         max_val =
  //             std::max(max_val, in[b * M * N * K + m * N * K + n * K + k]);
  //       }

  //       float sum = 0.0f;
  //       for (int k = 0; k < K; ++k) {
  //         out[b * M * N * K + m * N * K + n * K + k] =
  //             std::exp(in[b * M * N * K + m * N * K + n * K + k] - max_val);
  //         sum += out[b * M * N * K + m * N * K + n * K + k];
  //       }

  //       for (int k = 0; k < K; ++k) {
  //         out[b * M * N * K + m * N * K + n * K + k] /= sum;
  //       }
  //     }
  //   }
  // }
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int base_index = b * M * N * K + m * N * K + n * K;

        // Initialize Eigen::Map for input and output vectors (in-place processing)
        Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> input(
            &in[base_index], K);
        Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> output(
            &out[base_index], K);

        // Step 1: Find the maximum value (to improve numerical stability)
        float max_val = input.maxCoeff();

        // Step 2: Compute the exponentials and their sum
        Eigen::Array<float, Eigen::Dynamic, 1> exp_values =
            (input - max_val).exp();
        float sum_exp = exp_values.sum();

        // Step 3: Normalize the exponentials (in-place, no extra memory allocation)
        output = exp_values / sum_exp;
      }
    }
  }

#if INTERNAL_TIME_MEASURE == 1
  // display in milli
  auto end = std::chrono::steady_clock::now();
  std::cout << "Softmax: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
#endif
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

void jpyo0803::RMSNorm_Func(float* out, float* in, const float* const weight,
                            int B, int M, int N, float eps) {
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

uint64_t jpyo0803::RepeatedSqr(uint64_t base, uint64_t exp, uint64_t mod) {
  uint64_t result = 1;
  base = base % mod;
  while (exp > 0) {
    if (exp % 2 == 1) {
      result = (result * base) % mod;
    }
    exp = exp >> 1;
    base = (base * base) % mod;
  }
  return result;
}

void jpyo0803::Matmul_Eigen(int32_t* out, int8_t* x, int8_t* y, int B, int M,
                            int N, int K) {
  // Notice the dim K is the shared dim
  // a: [B, M, K]
  // b: [B, N, K]

  for (int b = 0; b < B; ++b) {
    // Map raw data to Eigen matrices
    Eigen::Map<Matrix<int8_t, Dynamic, Dynamic, RowMajor>> mat_x(x + b * M * K,
                                                                 M, K);
    Eigen::Map<Matrix<int8_t, Dynamic, Dynamic, RowMajor>> mat_y(y + b * N * K,
                                                                 N, K);

    // Compute the matrix product
    Matrix<int32_t, Dynamic, Dynamic, RowMajor> result =
        (mat_x.cast<int32_t>() * mat_y.transpose().cast<int32_t>());

    // Copy result back to output array
    std::memcpy(out + b * M * N, result.data(), M * N * sizeof(int32_t));
  }
}

void jpyo0803::Matmul_Naive(int32_t* out, int8_t* x, int8_t* y, int B, int M,
                            int N, int K) {
  // Notice the dim K is the shared dim
  // a: [B, M, K]
  // b: [B, N, K]

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += static_cast<int32_t>(x[b * M * K + m * K + k]) *
                 static_cast<int32_t>(y[b * N * K + n * K + k]);
        }
        out[b * M * N + m * N + n] = sum;
      }
    }
  }
}

void jpyo0803::GetTimeStamp_Monotonic() {
  auto start = std::chrono::steady_clock::now();
  std::cout << "Current time: " << start.time_since_epoch().count() / 1e9
            << std::endl;
}