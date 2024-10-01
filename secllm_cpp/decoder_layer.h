#ifndef SECLLM_CPP_DECODER_LAYER_H
#define SECLLM_CPP_DECODER_LAYER_H

#include <memory>
#include <utility>
#include <vector>

#include "tensor.h"

namespace jpyo0803 {

class DecoderLayer {
 public:
  DecoderLayer(int layer_idx, int hidden_size, int intermediate_size,
               int max_position_embeddings, int num_attention_heads,
               int num_key_value_heads, int enc_key_pool_size);

 public:
  void SetEncKeyAndDecKey(int* src_enc_key_pool,
                          std::vector<std::vector<int>>& dst_enc_key_pool,
                          int* src_dec_key,
                          std::vector<std::vector<int>>& dst_dec_key);

  void SetEncKeyAndDecKey_Q(int* src_enc_key_pool, int* src_dec_key);
  void SetEncKeyAndDecKey_K(int* src_enc_key_pool, int* src_dec_key);
  void SetEncKeyAndDecKey_V(int* src_enc_key_pool, int* src_dec_key);
  void SetEncKeyAndDecKey_O(int* src_enc_key_pool, int* src_dec_key);
  void SetEncKeyAndDecKey_Up(int* src_enc_key_pool, int* src_dec_key);
  void SetEncKeyAndDecKey_Gate(int* src_enc_key_pool, int* src_dec_key);
  void SetEncKeyAndDecKey_Down(int* src_enc_key_pool, int* src_dec_key);

  void SetLinearWeightScales_Q(float* weight_scales, int len);
  void SetLinearWeightScales_K(float* weight_scales, int len);
  void SetLinearWeightScales_V(float* weight_scales, int len);
  void SetLinearWeightScales_O(float* weight_scales, int len);
  void SetLinearWeightScales_Up(float* weight_scales, int len);
  void SetLinearWeightScales_Gate(float* weight_scales, int len);
  void SetLinearWeightScales_Down(float* weight_scales, int len);

  void EncryptLinearActivation_Q(int* out,
                                 std::shared_ptr<Tensor<float>> q_tensor);
  void EncryptLinearActivation_K(int* out,
                                 std::shared_ptr<Tensor<float>> k_tensor);
  void EncryptLinearActivation_V(int* out,
                                 std::shared_ptr<Tensor<float>> v_tensor);
  void EncryptLinearActivation_O(int* out,
                                 std::shared_ptr<Tensor<float>> o_tensor);
  void EncryptLinearActivation_Up(int* out,
                                  std::shared_ptr<Tensor<float>> up_tensor);
  void EncryptLinearActivation_Gate(int* out,
                                    std::shared_ptr<Tensor<float>> gate_tensor);
  void EncryptLinearActivation_Down(int* out,
                                    std::shared_ptr<Tensor<float>> down_tensor);

  void DecryptLinearActivation_Q(std::shared_ptr<Tensor<float>> out, int* in);
  void DecryptLinearActivation_K(std::shared_ptr<Tensor<float>> out, int* in);
  void DecryptLinearActivation_V(std::shared_ptr<Tensor<float>> out, int* in);
  void DecryptLinearActivation_O(std::shared_ptr<Tensor<float>> out, int* in);
  void DecryptLinearActivation_Up(std::shared_ptr<Tensor<float>> out, int* in);
  void DecryptLinearActivation_Gate(std::shared_ptr<Tensor<float>> out,
                                    int* in);
  void DecryptLinearActivation_Down(std::shared_ptr<Tensor<float>> out,
                                    int* in);

  void SetQKVOutputScales(float q_output_scale, float k_output_scale,
                          float v_output_scale);

  void QuantizeAndShiftQ(std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<float>> in);
  void QuantizeAndShiftK(std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<float>> in);
  void UnshiftAndDequantizeQK(std::shared_ptr<Tensor<float>> out,
                              std::shared_ptr<Tensor<uint32_t>> in);

  void QuantizeAndShiftP(std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<float>> in);
  void QuantizeAndShiftV(std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<float>> in);
  std::shared_ptr<Tensor<float>> UnshiftAndDequantizePV(
      std::shared_ptr<Tensor<uint32_t>> in);

  void Reset();

  void SetBatchSizeAndTokenLength(int bsz, int token_len);

  void GenerateSecretKey_QK();

  void GenerateDecryptionKey_QK(std::shared_ptr<Tensor<uint32_t>> x,
                                std::shared_ptr<Tensor<uint32_t>> y);

  void EncryptX_QK(std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void EncryptY_QK(std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void Decrypt_QK(std::shared_ptr<Tensor<uint32_t>> out,
                  std::shared_ptr<Tensor<uint32_t>> in);

  void GenerateSecretKey_PV();

  void GenerateDecryptionKey_PV(std::shared_ptr<Tensor<uint32_t>> x,
                                std::shared_ptr<Tensor<uint32_t>> y);

  void EncryptX_PV(std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void EncryptY_PV(std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void Decrypt_PV(std::shared_ptr<Tensor<uint32_t>> out,  // D_ROW
                  std::shared_ptr<Tensor<uint32_t>> in);  // D_COL

  // This is not actually thread-safe but it is okay, task scheduler can run it next cycle
  bool IsQKKeyGenerated() const { return is_qk_key_generated_; }
  bool IsQKDecKeyGenerated() const { return is_qk_dec_key_generated_; }
  bool IsPVKeyGenerated() const { return is_pv_key_generated_; }
  bool IsPVDecKeyGenerated() const { return is_pv_dec_key_generated_; }

 private:
  int layer_idx_;
  int hidden_size_;
  int intermediate_size_;
  int max_position_embeddings_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int head_dim_;
  int enc_key_pool_size_;
  int num_key_value_groups_;

  int present_token_len_;
  int culmulative_token_len_;
  int bsz_;

  std::vector<std::vector<int>> q_enc_key_pool_;
  std::vector<std::vector<int>> q_dec_key_;
  std::vector<int> sampled_q_enc_key_index_;
  std::vector<float> q_act_scales_;
  std::vector<float> q_weight_scales_;
  float q_output_scale_;

  std::vector<std::vector<int>> k_enc_key_pool_;
  std::vector<std::vector<int>> k_dec_key_;
  std::vector<int> sampled_k_enc_key_index_;
  std::vector<float> k_act_scales_;
  std::vector<float> k_weight_scales_;
  float k_output_scale_;

  std::vector<std::vector<int>> v_enc_key_pool_;
  std::vector<std::vector<int>> v_dec_key_;
  std::vector<int> sampled_v_enc_key_index_;
  std::vector<float> v_act_scales_;
  std::vector<float> v_weight_scales_;
  float v_output_scale_;

  std::vector<std::vector<int>> o_enc_key_pool_;
  std::vector<std::vector<int>> o_dec_key_;
  std::vector<int> sampled_o_enc_key_index_;
  std::vector<float> o_act_scales_;
  std::vector<float> o_weight_scales_;

  std::vector<std::vector<int>> up_enc_key_pool_;
  std::vector<std::vector<int>> up_dec_key_;
  std::vector<int> sampled_up_enc_key_index_;
  std::vector<float> up_act_scales_;
  std::vector<float> up_weight_scales_;

  std::vector<std::vector<int>> gate_enc_key_pool_;
  std::vector<std::vector<int>> gate_dec_key_;
  std::vector<int> sampled_gate_enc_key_index_;
  std::vector<float> gate_act_scales_;
  std::vector<float> gate_weight_scales_;

  std::vector<std::vector<int>> down_enc_key_pool_;
  std::vector<std::vector<int>> down_dec_key_;
  std::vector<int> sampled_down_enc_key_index_;
  std::vector<float> down_act_scales_;
  std::vector<float> down_weight_scales_;

  std::vector<std::vector<std::vector<int>>>
      qk_x_row_shift_sum_;  // input is 4D
  std::vector<std::vector<std::vector<int>>> qk_y_col_shift_sum_;

  std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>>
      qk_x_mult_key_;
  std::vector<std::vector<std::vector<uint32_t>>> qk_x_add_key_;
  std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>>
      qk_y_mult_key_;
  std::vector<std::vector<std::vector<uint32_t>>> qk_y_add_key_;

  std::vector<std::vector<std::vector<uint32_t>>> qk_dec_row_;  // D_ROW
  std::vector<std::vector<std::vector<uint32_t>>> qk_dec_col_;  // D_COL
  std::vector<std::vector<uint32_t>> qk_dec_glob_;              // D_GLOB

  std::vector<std::vector<std::vector<int>>> pv_x_row_shift_sum_;
  std::vector<std::vector<std::vector<int>>> pv_y_col_shift_sum_;

  std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>>
      pv_x_mult_key_;
  std::vector<std::vector<std::vector<uint32_t>>> pv_x_add_key_;
  std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>>
      pv_y_mult_key_;
  std::vector<std::vector<std::vector<uint32_t>>> pv_y_add_key_;

  std::vector<std::vector<std::vector<uint32_t>>> pv_dec_row_;  // D_ROW
  std::vector<std::vector<std::vector<uint32_t>>> pv_dec_col_;  // D_COL
  std::vector<std::vector<uint32_t>> pv_dec_glob_;              // D_GLOB

  std::vector<uint32_t> x_mult_key_pool_;
  std::vector<uint32_t> y_mult_key_pool_;
  std::vector<std::vector<uint32_t>> precomputed_key_inv_;

  bool is_qk_key_generated_;      // required for decryption key generation
  bool is_qk_dec_key_generated_;  // required for decryption

  bool is_pv_key_generated_;
  bool is_pv_dec_key_generated_;
};

}  // namespace jpyo0803

#endif  // SECLLM_CPP_DECODER_LAYER_H