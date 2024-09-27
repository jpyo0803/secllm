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

void DecoderLayer::EncryptLinearActivation_Q(
    int* out, std::shared_ptr<Tensor<float>> q_tensor) {
  int B = q_tensor->shape()[0];
  int M = q_tensor->shape()[1];
  int N = q_tensor->shape()[2];

  auto [q_act, max_vals] =
      DynamicQuantizeActivationPerTokenAbsmax(q_tensor->data(), B, M, N);
  q_act_scales_ = std::move(max_vals);

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] = q_act[b * M * N + m * N + n];
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

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] = k_act[b * M * N + m * N + n];
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

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out[b * M * N + m * N + n] = v_act[b * M * N + m * N + n];
      }
    }
  }
}


void DecoderLayer::DecryptLinearActivation_Q(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, q_weight_scales_, q_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_K(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, k_weight_scales_, k_act_scales_, B * M, N);
}

void DecoderLayer::DecryptLinearActivation_V(std::shared_ptr<Tensor<float>> out,
                                             int* in) {
  int B = out->shape()[0];
  int M = out->shape()[1];
  int N = out->shape()[2];

  DequantizeActivationWPerChannelAPerChannel(
      out->data().data(), in, v_weight_scales_, v_act_scales_, B * M, N);
}

}  // namespace jpyo0803