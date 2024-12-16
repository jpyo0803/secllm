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

std::vector<int8_t> Flatten4Dto1D(
    const std::vector<std::vector<std::vector<std::vector<int8_t>>>>& input,
    int batch, int num_heads, int seqlen, int head_dim) {
  std::vector<int8_t> flattened(batch * num_heads * seqlen * head_dim);
  int idx = 0;

  // Flattening the 4D vector
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      for (int s = 0; s < seqlen; ++s) {
        for (int d = 0; d < head_dim; ++d) {
          flattened[idx++] = input[b][h][s][d];
        }
      }
    }
  }

  return flattened;
}

}  // namespace

void jpyo0803::QuantizeActivationPerTensor(std::vector<int8_t>& out,
                                           const std::vector<float>& in,
                                           int64_t len, float scale) {
  // Ensure the output vector is resized to match the input length
  out.resize(len);

  // Map input and output data to Eigen arrays
  Eigen::Map<const Eigen::ArrayXf> in_array(in.data(), len);
  Eigen::Map<Eigen::Array<int8_t, Eigen::Dynamic, 1>> out_array(out.data(),
                                                                len);

  // Apply the quantization process using Eigen
  out_array = (in_array / scale).round().unaryExpr([](float val) {
    return static_cast<int8_t>(std::clamp(val, -128.0f, 127.0f));
  });
}

void jpyo0803::DequantizeActivationPerTensor(std::vector<float>& out,
                                             const std::vector<int32_t>& in,
                                             int64_t len, float scale) {
  // Ensure the output vector is resized to match the input length
  out.resize(len);

  // Map input and output data to Eigen arrays
  Eigen::Map<const Eigen::ArrayXi> in_array(in.data(), len);
  Eigen::Map<Eigen::ArrayXf> out_array(out.data(), len);

  // Apply the dequantization process using Eigen
  out_array = in_array.cast<float>() * scale;
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
  // Map the weight scales and activation scales to Eigen vectors
  Eigen::Map<const Eigen::VectorXf> w_scales_vec(w_scales.data(), dim);
  Eigen::Map<const Eigen::VectorXf> a_scales_vec(a_scales.data(), B);

  // Iterate over each batch
  for (size_t b = 0; b < B; ++b) {
    // Map the input and output slices for the current batch (size `dim`)
    Eigen::Map<const Eigen::VectorXi> in_vec(in.data() + b * dim, dim);
    Eigen::Map<Eigen::VectorXf> out_vec(out.data() + b * dim, dim);

    // Dequantize: Convert `in` to float and apply per-channel (w_scales) and per-batch (a_scales[b]) scaling
    out_vec =
        in_vec.cast<float>().array() * w_scales_vec.array() * a_scales_vec[b];
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
  // for (int b = 0; b < B; ++b) {
  //   for (int m = 0; m < M; ++m) {
  //     for (int n = 0; n < N; ++n) {
  //       int base_index = b * M * N * K + m * N * K + n * K;

  //       // Initialize Eigen::Map for input and output vectors (in-place processing)
  //       Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> input(
  //           &in[base_index], K);
  //       Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> output(
  //           &out[base_index], K);

  //       // Step 1: Find the maximum value (to improve numerical stability)
  //       float max_val = input.maxCoeff();

  //       // Step 2: Compute the exponentials and their sum
  //       Eigen::Array<float, Eigen::Dynamic, 1> exp_values =
  //           (input - max_val).exp();
  //       float sum_exp = exp_values.sum();

  //       // Step 3: Normalize the exponentials (in-place, no extra memory allocation)
  //       output = exp_values / sum_exp;
  //     }
  //   }
  // }

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
  // Map the input and output data as Eigen matrices
  Eigen::Map<Eigen::MatrixXf> gate_mat(gate_in, B * M, N);
  Eigen::Map<Eigen::MatrixXf> up_mat(up_in, B * M, N);
  Eigen::Map<Eigen::MatrixXf> out_mat(out, B * M, N);

  // Apply the SiLU activation on gate_in
  gate_mat = gate_mat.array() / (1.0f + (-gate_mat.array()).exp());

  // Perform element-wise multiplication for SwiGLU
  out_mat = gate_mat.array() * up_mat.array();
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
  // Map the weight array as an Eigen vector
  Eigen::Map<const Eigen::VectorXf> weight_vec(weight, N);

  // Iterate over batches and M
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      // Map the current slice of in and out for the current batch and M index
      Eigen::Map<Eigen::VectorXf> out_vec(out + b * M * N + m * N, N);
      Eigen::Map<const Eigen::VectorXf> in_vec(in + b * M * N + m * N, N);

      // Calculate the square sum (L2 norm squared)
      float sqr_sum = in_vec.squaredNorm();
      float variance = sqr_sum / N;

      // Perform normalization and apply the weight
      out_vec = in_vec.array() / std::sqrt(variance + eps);
      out_vec = out_vec.array() * weight_vec.array();
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
  // Map the input and output arrays as Eigen matrices
  Eigen::Map<Eigen::MatrixXf> out_mat(out, B * M, N);
  Eigen::Map<const Eigen::MatrixXf> x_mat(x, B * M, N);
  Eigen::Map<const Eigen::MatrixXf> y_mat(y, B * M, N);

  // Perform element-wise addition
  out_mat = x_mat + y_mat;
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

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < Q_M; ++m) {
      for (int n = 0; n < N; ++n) {
        Eigen::Map<Eigen::ArrayXf> q_input(
            &q_tensor[b * Q_M * N * K + m * N * K + n * K], K);
        Eigen::Map<const Eigen::ArrayXf> cos_map(&cos[n * K], K);
        Eigen::Map<const Eigen::ArrayXf> sin_map(&sin[n * K], K);

        // Split into two halves
        Eigen::ArrayXf q_cos_part = q_input.head(K / 2) * cos_map.head(K / 2) -
                                    q_input.tail(K / 2) * sin_map.head(K / 2);
        Eigen::ArrayXf q_sin_part = q_input.head(K / 2) * sin_map.head(K / 2) +
                                    q_input.tail(K / 2) * cos_map.head(K / 2);

        // Concatenate the result
        q_input.head(K / 2) = q_cos_part;
        q_input.tail(K / 2) = q_sin_part;
      }
    }

    for (int m = 0; m < K_M; ++m) {
      for (int n = 0; n < N; ++n) {
        Eigen::Map<Eigen::ArrayXf> k_input(
            &k_tensor[b * K_M * N * K + m * N * K + n * K], K);
        Eigen::Map<const Eigen::ArrayXf> cos_map(&cos[n * K], K);
        Eigen::Map<const Eigen::ArrayXf> sin_map(&sin[n * K], K);

        // Split into two halves
        Eigen::ArrayXf k_cos_part = k_input.head(K / 2) * cos_map.head(K / 2) -
                                    k_input.tail(K / 2) * sin_map.head(K / 2);
        Eigen::ArrayXf k_sin_part = k_input.head(K / 2) * sin_map.head(K / 2) +
                                    k_input.tail(K / 2) * cos_map.head(K / 2);

        // Concatenate the result
        k_input.head(K / 2) = k_cos_part;
        k_input.tail(K / 2) = k_sin_part;
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

  Eigen::Map<const Eigen::ArrayXf> inv_freq_map(inv_freq, inv_freq_M);
  Eigen::Map<const Eigen::ArrayXf> pos_ids_map(position_ids, position_ids_M);

  for (int i = 0; i < position_ids_M; ++i) {
    Eigen::ArrayXf half_emb = pos_ids_map(i) * inv_freq_map;

    for (int j = 0; j < inv_freq_M; ++j) {
      cos[i * inv_freq_M * 2 + j] = cos[i * inv_freq_M * 2 + (inv_freq_M + j)] =
          std::cos(half_emb(j));
      sin[i * inv_freq_M * 2 + j] = sin[i * inv_freq_M * 2 + (inv_freq_M + j)] =
          std::sin(half_emb(j));
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

void jpyo0803::Matmul_Eigen(int32_t* out, int8_t* x, int8_t* y, int B, int X_M,
                            int X_N, int Y_M, int Y_N, bool need_transpose) {
  // B is the batch size, x and y shares
  // x: [B, X_M, X_N]
  // y: [B, Y_M, Y_N]

  // if need_transpose is true, then X_N == Y_N
  // if need_transpose is false, then X_N == Y_M

  for (int b = 0; b < B; ++b) {
    // Map x and y as Eigen matrices
    Eigen::Map<const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        x_map(&x[b * X_M * X_N], X_M, X_N);

    Eigen::Map<const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        y_map(&y[b * Y_M * Y_N], Y_M, Y_N);

    // Ensure the dimensions of `out` are correct before assigning
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        result(X_M, Y_N);

    if (need_transpose) {
      // Perform matrix multiplication with y_transposed if need_transpose is true
      result.noalias() =
          x_map.cast<int32_t>() * y_map.transpose().cast<int32_t>();
    } else {
      // Perform regular matrix multiplication
      result.noalias() = x_map.cast<int32_t>() * y_map.cast<int32_t>();
    }

    // Copy the result into the output array
    std::memcpy(&out[b * X_M * Y_N], result.data(),
                X_M * Y_N * sizeof(int32_t));
  }
}

void jpyo0803::Matmul_Naive(int32_t* out, int8_t* x, int8_t* y, int B, int M,
                            int N, int K, bool transpose_y) {
  // Notice the dim K is the shared dim
  // a: [B, M, K]
  // b: [B, N, K]

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          int8_t x_val = x[b * M * K + m * K + k];
          int8_t y_val;

          if (transpose_y) {
            // Access y as [B, K, N]
            y_val = y[b * K * N + k * N + n];
          } else {
            // Access y as [B, N, K]
            y_val = y[b * N * K + n * K + k];
          }

          sum += static_cast<int32_t>(x_val) * static_cast<int32_t>(y_val);
        }
        out[b * M * N + m * N + n] = sum;
      }
    }
  }
}

std::vector<int8_t> jpyo0803::RepeatKV(
    const std::vector<std::vector<std::vector<std::vector<int8_t>>>>&
        hidden_states,
    int batch, int num_key_value_heads, int seqlen, int head_dim, int n_rep) {
  // If n_rep is 1, return the input as a flattened 1D vector
  if (n_rep == 1) {
    return Flatten4Dto1D(hidden_states, batch, num_key_value_heads, seqlen,
                         head_dim);
  }

  // The new dimension size for num_attention_heads
  int num_attention_heads = num_key_value_heads * n_rep;

  // Create a 4D vector to hold the repeated hidden states
  std::vector<std::vector<std::vector<std::vector<int8_t>>>>
      repeated_hidden_states(
          batch,
          std::vector<std::vector<std::vector<int8_t>>>(
              num_attention_heads, std::vector<std::vector<int8_t>>(
                                       seqlen, std::vector<int8_t>(head_dim))));

  // Perform the repeat operation
  for (int b = 0; b < batch; ++b) {
    for (int kvh = 0; kvh < num_key_value_heads; ++kvh) {
      for (int rep = 0; rep < n_rep; ++rep) {
        for (int s = 0; s < seqlen; ++s) {
          for (int h = 0; h < head_dim; ++h) {
            repeated_hidden_states[b][kvh * n_rep + rep][s][h] =
                hidden_states[b][kvh][s][h];
          }
        }
      }
    }
  }

  // Flatten the 4D vector and return it
  return Flatten4Dto1D(repeated_hidden_states, batch, num_attention_heads,
                       seqlen, head_dim);
}

void jpyo0803::GetTimeStamp_Monotonic() {
#if SGX_ENABLE == 0
  auto start = std::chrono::steady_clock::now();
  std::cout << "Current time: " << start.time_since_epoch().count() / 1e9
            << std::endl;
#endif
}