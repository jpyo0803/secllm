#include "decoder_layer.h"

#include <iostream>

#include "aes_stream.h"

#include "func_utils.h"

namespace jpyo0803 {

DecoderLayer::DecoderLayer(int layer_idx, int hidden_size,
                           int intermediate_size, int max_position_embeddings,
                           int num_attention_heads, int num_key_value_heads,
                           int enc_key_pool_size)
    : layer_idx_(layer_idx),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      max_position_embeddings_(max_position_embeddings),
      num_attention_heads_(num_attention_heads),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(hidden_size / num_attention_heads),
      enc_key_pool_size_(enc_key_pool_size) {

  std::cout << "Decoder Layer " << layer_idx_ << " is created." << std::endl;

  q_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  k_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  v_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  o_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  up_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  gate_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(hidden_size_));
  down_enc_key_pool_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(intermediate_size_));

  q_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(num_attention_heads_ * head_dim_));
  k_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(num_key_value_heads_ * head_dim_));
  v_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(num_key_value_heads_ * head_dim_));
  o_dec_key_ = std::vector<std::vector<int>>(enc_key_pool_size_,
                                             std::vector<int>(hidden_size_));
  up_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(intermediate_size_));
  gate_dec_key_ = std::vector<std::vector<int>>(
      enc_key_pool_size_, std::vector<int>(intermediate_size_));
  down_dec_key_ = std::vector<std::vector<int>>(enc_key_pool_size_,
                                                std::vector<int>(hidden_size_));

  q_weight_scales_ = std::vector<float>(num_attention_heads_ * head_dim_);
  k_weight_scales_ = std::vector<float>(num_key_value_heads_ * head_dim_);
  v_weight_scales_ = std::vector<float>(num_key_value_heads_ * head_dim_);
  o_weight_scales_ = std::vector<float>(hidden_size_);

  up_weight_scales_ = std::vector<float>(intermediate_size_);
  gate_weight_scales_ = std::vector<float>(intermediate_size_);
  down_weight_scales_ = std::vector<float>(hidden_size_);
}

void DecoderLayer::SetEncKeyAndDecKey(
    int* src_enc_key_pool, std::vector<std::vector<int>>& dst_enc_key_pool,
    int* src_dec_key, std::vector<std::vector<int>>& dst_dec_key) {
  for (int i = 0; i < dst_enc_key_pool.size(); ++i) {
    for (int j = 0; j < dst_enc_key_pool[0].size(); ++j) {
      dst_enc_key_pool[i][j] =
          src_enc_key_pool[i * dst_enc_key_pool[0].size() + j];
    }
  }

  for (int i = 0; i < dst_dec_key.size(); ++i) {
    for (int j = 0; j < dst_dec_key[0].size(); ++j) {
      dst_dec_key[i][j] = src_dec_key[i * dst_dec_key[0].size() + j];
    }
  }
}

void DecoderLayer::SetLinearWeightScales_Q(float* weight_scales, int len) {
  q_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_K(float* weight_scales, int len) {
  k_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_V(float* weight_scales, int len) {
  v_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_O(float* weight_scales, int len) {
  o_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_Up(float* weight_scales, int len) {
  up_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_Gate(float* weight_scales, int len) {
  gate_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetLinearWeightScales_Down(float* weight_scales, int len) {
  down_weight_scales_.assign(weight_scales, weight_scales + len);
}

void DecoderLayer::SetEncKeyAndDecKey_Q(int* src_enc_key_pool,
                                        int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, q_enc_key_pool_, src_dec_key,
                     q_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_K(int* src_enc_key_pool,
                                        int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, k_enc_key_pool_, src_dec_key,
                     k_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_V(int* src_enc_key_pool,
                                        int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, v_enc_key_pool_, src_dec_key,
                     v_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_O(int* src_enc_key_pool,
                                        int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, o_enc_key_pool_, src_dec_key,
                     o_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_Up(int* src_enc_key_pool,
                                         int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, up_enc_key_pool_, src_dec_key,
                     up_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_Gate(int* src_enc_key_pool,
                                           int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, gate_enc_key_pool_, src_dec_key,
                     gate_dec_key_);
}

void DecoderLayer::SetEncKeyAndDecKey_Down(int* src_enc_key_pool,
                                           int* src_dec_key) {
  SetEncKeyAndDecKey(src_enc_key_pool, down_enc_key_pool_, src_dec_key,
                     down_dec_key_);
}

void DecoderLayer::EncryptLinearActivation_Q(
    int* out, std::shared_ptr<Tensor<float>> q_tensor) {
  int B = q_tensor->shape()[0];
  int M = q_tensor->shape()[1];
  int N = q_tensor->shape()[2];

  // Apply Quantization
  auto [q_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(q_tensor->data(), B, M, N);
  q_act_scales_ = std::move(max_vals);

  if (!sampled_q_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_q_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)q_act[b * M * N + m * N + n] +
            q_enc_key_pool_[sampled_q_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_K(
    int* out, std::shared_ptr<Tensor<float>> k_tensor) {
  int B = k_tensor->shape()[0];
  int M = k_tensor->shape()[1];
  int N = k_tensor->shape()[2];

  auto [k_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(k_tensor->data(), B, M, N);
  k_act_scales_ = std::move(max_vals);

  if (!sampled_k_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_k_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)k_act[b * M * N + m * N + n] +
            k_enc_key_pool_[sampled_k_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_V(
    int* out, std::shared_ptr<Tensor<float>> v_tensor) {
  int B = v_tensor->shape()[0];
  int M = v_tensor->shape()[1];
  int N = v_tensor->shape()[2];

  auto [v_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(v_tensor->data(), B, M, N);
  v_act_scales_ = std::move(max_vals);

  if (!sampled_v_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_v_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)v_act[b * M * N + m * N + n] +
            v_enc_key_pool_[sampled_v_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_O(
    int* out, std::shared_ptr<Tensor<float>> o_tensor) {
  int B = o_tensor->shape()[0];
  int M = o_tensor->shape()[1];
  int N = o_tensor->shape()[2];

  auto [o_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(o_tensor->data(), B, M, N);
  o_act_scales_ = std::move(max_vals);

  if (!sampled_o_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_o_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)o_act[b * M * N + m * N + n] +
            o_enc_key_pool_[sampled_o_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_Up(
    int* out, std::shared_ptr<Tensor<float>> up_tensor) {
  int B = up_tensor->shape()[0];
  int M = up_tensor->shape()[1];
  int N = up_tensor->shape()[2];

  auto [up_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(up_tensor->data(), B, M, N);
  up_act_scales_ = std::move(max_vals);

  if (!sampled_up_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_up_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)up_act[b * M * N + m * N + n] +
            up_enc_key_pool_[sampled_up_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_Gate(
    int* out, std::shared_ptr<Tensor<float>> gate_tensor) {
  int B = gate_tensor->shape()[0];
  int M = gate_tensor->shape()[1];
  int N = gate_tensor->shape()[2];

  auto [gate_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(gate_tensor->data(), B, M, N);
  gate_act_scales_ = std::move(max_vals);

  if (!sampled_gate_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_gate_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)gate_act[b * M * N + m * N + n] +
            gate_enc_key_pool_[sampled_gate_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::EncryptLinearActivation_Down(
    int* out, std::shared_ptr<Tensor<float>> down_tensor) {
  int B = down_tensor->shape()[0];
  int M = down_tensor->shape()[1];
  int N = down_tensor->shape()[2];

  auto [down_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(down_tensor->data(), B, M, N);
  down_act_scales_ = std::move(max_vals);

  if (!sampled_down_enc_key_index_.empty()) {
    std::cout << "Encryption is called twice in a row!" << std::endl;
    exit(-1);
  }

  // Sample encryption keys from the pool
  for (int i = 0; i < B * M; ++i) {
    sampled_down_enc_key_index_.push_back(GenerateCPRNG() % enc_key_pool_size_);
  }

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] =
            (int)down_act[b * M * N + m * N + n] +
            down_enc_key_pool_[sampled_down_enc_key_index_[b * M + m]][n];
      }
    }
  }
}

void DecoderLayer::DecryptLinearActivation_Q(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  // Decrypt Result
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            q_dec_key_[sampled_q_enc_key_index_[b * M + m]][n];
      }
    }
  }
  sampled_q_enc_key_index_.clear();

  // Dequantize
  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, q_weight_scales_, q_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_K(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            k_dec_key_[sampled_k_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_k_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, k_weight_scales_, k_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_V(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            v_dec_key_[sampled_v_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_v_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, v_weight_scales_, v_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_O(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            o_dec_key_[sampled_o_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_o_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, o_weight_scales_, o_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_Up(
    std::shared_ptr<Tensor<float>> out, int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            up_dec_key_[sampled_up_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_up_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, up_weight_scales_, up_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_Gate(
    std::shared_ptr<Tensor<float>> out, int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            gate_dec_key_[sampled_gate_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_gate_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, gate_weight_scales_, gate_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_Down(
    std::shared_ptr<Tensor<float>> out, int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        in[b * M * N + m * N + n] -=
            down_dec_key_[sampled_down_enc_key_index_[b * M + m]][n];
      }
    }
  }

  sampled_down_enc_key_index_.clear();

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, down_weight_scales_, down_act_scales_, B * M, N);
}

}  // namespace jpyo0803