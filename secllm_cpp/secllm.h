#ifndef SECLLM_SECLLM_H
#define SECLLM_SECLLM_H

#include <stdint.h>
#include <memory>

#include "book_keeper.h"
#include "decoder_layer.h"
#include "tensor.h"

namespace jpyo0803 {

class SecLLM {
 public:
  SecLLM(int hidden_size, int intermediate_size, int max_position_embeddings,
         int num_attention_heads, int num_hidden_layers,
         int num_key_value_heads, int enc_key_pool_size);

 public:
  void Reset();

  // member methods
  void BookKeeperStore(std::vector<int> locs,
                       std::shared_ptr<Tensor<float>>& data);

  std::shared_ptr<Tensor<float>> BookKeeperLoad(int loc);

  void BookKeeperStore_Uint32(std::vector<int> locs,
                              std::shared_ptr<Tensor<uint32_t>>& data);

  std::shared_ptr<Tensor<uint32_t>> BookKeeperLoad_Uint32(int loc);

  void SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool, int* dec_key,
                          int type);

  void SetLinearWeightScales(int layer_idx, float* weight_scale, int len,
                             int type);

  void EncryptLinearActivation(int layer_idx, int* out,
                               std::shared_ptr<Tensor<float>> in, int type);

  void DecryptLinearActivation(int layer_idx,
                               std::shared_ptr<Tensor<float>> out, int* in,
                               int type);

  void SetQKVOutputScales(int layer_idx, float q_output_scale,
                          float k_output_scale, float v_output_scale);

  void QuantizeAndShiftQ(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<float>> in);

  void QuantizeAndShiftK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<float>> in);

  void UnshiftAndDequantizeQK(int layer_idx, std::shared_ptr<Tensor<float>> out,
                              std::shared_ptr<Tensor<uint32_t>> in);

  void QuantizeAndShiftP(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<float>> in);

  void QuantizeAndShiftV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                         std::shared_ptr<Tensor<float>> in);

  std::shared_ptr<Tensor<float>> UnshiftAndDequantizePV(
      int layer_idx, std::shared_ptr<Tensor<uint32_t>> in);

  void SetAttentionMask(float* mask, int M, int N);

 private:
  int num_hidden_layers_ = -1;

  std::unique_ptr<BookKeeper<Tensor<float>>> book_keeper_;

  // NOTE(jpyo0803): Although QK^T and PV operation will be done in uint32,
  // here I use int32 since torch does not support uint32.
  std::unique_ptr<BookKeeper<Tensor<uint32_t>>> book_keeper_uint32_;

  std::unique_ptr<std::vector<DecoderLayer>> decoder_layers_;

 public:
  std::vector<std::vector<float>> attn_mask_;
  // std::unique_ptr<Tensor<float>> attn_mask_tensor_;
};

}  // namespace jpyo0803

extern "C" {

void Ext_PrintTest(int a, int b);

void Ext_CreateSecLLM(int hidden_size, int intermediate_size,
                      int max_position_embeddings, int num_attention_heads,
                      int num_hidden_layers, int num_key_value_heads,
                      int enc_key_pool_size);

void Ext_Softmax_InPlace(float* x, int B, int M, int N, int K);

void Ext_Softmax(int from, int to);

void Ext_SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N);

void Ext_SwiGLU(int from1, int from2, int to);

void Ext_RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                         int N, float eps);

void Ext_RMSNorm(int from, int to, const float* const weight, float eps);

void Ext_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N);

void Ext_ElementWiseAdd(int from1, int from2, int to);

void Ext_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                           const float* const cos, const float* const sin,
                           int B, int Q_M, int K_M, int N, int K);

void Ext_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                              const float* const position_ids,
                              int position_ids_M, float* cos, float* sin);

uint32_t Ext_GenerateCPRNG();

uint32_t Ext_GenerateMultKey();

uint32_t Ext_GenerateAddKey();

void Ext_Reset();

void Ext_ReplicateTensor(int from, int* to, int to_len);

void Ext_ReplicateTensor_Uint32(int from, int* to, int to_len);

void Ext_GetCprngTensor(int* out, int shape_len, int* shape);

void Ext_SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool, int* dec_key,
                            int type);

void Ext_SetLinearWeightScales(int layer_idx, float* scales, int len, int type);

void Ext_EncryptLinearActivation(int layer_idx, int* out, int from, int type);

void Ext_DecryptLinearActivation(int layer_idx, int to, int* enc_tensor,
                                 int shape_len, int* shape, int type);

void Ext_SetQKVOutputScales(int layer_idx, float q_output_scale,
                            float k_output_scale, float v_output_scale);

void Ext_QuantizeAndShiftQ(int layer_idx, int from, int to_len, int* to);

void Ext_QuantizeAndShiftK(int layer_idx, int from, int to_len, int* to);

void Ext_SetAttentionMask(float* mask, int M, int N);

}  // extern "C"

#endif  // SECLLM_SECLLM_H