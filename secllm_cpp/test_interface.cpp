#include "test_interface.h"
#include "func_utils.h"

extern "C" {

void Test_Matmul_Eigen(int32_t* out, int8_t* x, int8_t* y, int B, int X_M,
                       int X_N, int Y_M, int Y_N, bool transpose) {
  jpyo0803::Matmul_Eigen(out, x, y, B, X_M, X_N, Y_M, Y_N, transpose);
}

void Test_Matmul_Naive(int32_t* out, int8_t* x, int8_t* y, int B, int M, int K,
                       int N, bool transpose_y) {
  jpyo0803::Matmul_Naive(out, x, y, B, M, K, N, transpose_y);
}

void Test_GetTimeStamp_Monotonic() {
  jpyo0803::GetTimeStamp_Monotonic();
}

void Test_RepeatKV(int8_t* out, int8_t* hidden_states, int batch,
                   int num_key_value_heads, int seqlen, int head_dim,
                   int n_rep) {
  std::vector<std::vector<std::vector<std::vector<int8_t>>>> hidden_states_vec(
      batch,
      std::vector<std::vector<std::vector<int8_t>>>(
          num_key_value_heads, std::vector<std::vector<int8_t>>(
                                   seqlen, std::vector<int8_t>(head_dim))));
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_key_value_heads; ++h) {
      for (int s = 0; s < seqlen; ++s) {
        for (int d = 0; d < head_dim; ++d) {
          hidden_states_vec[b][h][s][d] =
              hidden_states[b * num_key_value_heads * seqlen * head_dim +
                            h * seqlen * head_dim + s * head_dim + d];
        }
      }
    }
  }

  std::vector<int8_t> out_vec = jpyo0803::RepeatKV(
      hidden_states_vec, batch, num_key_value_heads, seqlen, head_dim, n_rep);
  for (int i = 0; i < out_vec.size(); ++i) {
    out[i] = out_vec[i];
  }
}
}
