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

  bool BookKeeperIsAvailable(int loc);

  bool BookKeeperIsAvailable_Uint32(int loc);

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

  void SetBatchSizeAndTokenLength(int layer_idx, int bsz, int token_length);

  void GenerateSecretKey_QK(int layer_idx);

  void GenerateDecryptionKey_QK(int layer_idx,
                                std::shared_ptr<Tensor<uint32_t>> x,
                                std::shared_ptr<Tensor<uint32_t>> y);

  void EncryptX_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void EncryptY_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void Decrypt_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                  std::shared_ptr<Tensor<uint32_t>> in);

  void GenerateSecretKey_PV(int layer_idx);

  void GenerateDecryptionKey_PV(int layer_idx,
                                std::shared_ptr<Tensor<uint32_t>> x,
                                std::shared_ptr<Tensor<uint32_t>> y);

  void EncryptX_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void EncryptY_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void Decrypt_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                  std::shared_ptr<Tensor<uint32_t>> in);

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

void Internal_PrintTest(int a, int b);

void Internal_CreateSecLLM(int hidden_size, int intermediate_size,
                           int max_position_embeddings, int num_attention_heads,
                           int num_hidden_layers, int num_key_value_heads,
                           int enc_key_pool_size);

void Internal_Softmax_InPlace(float* x, int B, int M, int N, int K);

void Internal_Softmax(int from, int to_len, int* to);

void Internal_SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N);

void Internal_SwiGLU(int from1, int from2, int to_len, int* to);

void Internal_RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                              int N, float eps);

void Internal_RMSNorm(int from, int to_len, int* to, const float* const weight,
                      float eps);

void Internal_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N);

void Internal_ElementWiseAdd(int from1, int from2, int to_len, int* to);

void Internal_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                                const float* const cos, const float* const sin,
                                int B, int Q_M, int K_M, int N, int K);

void Internal_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                                   const float* const position_ids,
                                   int position_ids_M, float* cos, float* sin);

uint32_t Internal_GenerateCPRNG();

uint32_t Internal_GenerateMultKey();

uint32_t Internal_GenerateAddKey();

void Internal_Reset();

void Internal_ReplicateTensor(int from, int* to, int to_len);

void Internal_ReplicateTensor_Uint32(int from, int* to, int to_len);

void Internal_GetCprngTensor(int* out, int shape_len, int* shape);

void Internal_SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool, int* dec_key,
                                 int type);

void Internal_SetLinearWeightScales(int layer_idx, float* scales, int len,
                                    int type);

void Internal_EncryptLinearActivation(int layer_idx, int* out, int from,
                                      int type);

void Internal_DecryptLinearActivation(int layer_idx, int to_len, int* to,
                                      int* enc_tensor, int shape_len,
                                      int* shape, int type);

void Internal_SetQKVOutputScales(int layer_idx, float q_output_scale,
                                 float k_output_scale, float v_output_scale);

void Internal_QuantizeAndShiftQ(int layer_idx, int from, int to_len, int* to);

void Internal_QuantizeAndShiftK(int layer_idx, int from, int to_len, int* to);

void Internal_SetAttentionMask(float* mask, int M, int N);

void Internal_SetBatchSizeAndTokenLength(int layer_idx, int bsz,
                                         int token_length);

void Internal_GenerateSecretKey_QK(int layer_idx);

void Internal_GenerateDecryptionKey_QK(int layer_idx, int from_x, int from_y);

void Internal_QuantizeAndShiftQ(int layer_idx, int from, int to_len, int* to);

void Internal_QuantizeAndShiftK(int layer_idx, int from, int to_len, int* to);

void Internal_EncryptX_QK(int layer_idx, int from, int to_len, int* to);

void Internal_EncryptY_QK(int layer_idx, int from, int to_len, int* to);

void Internal_Decrypt_QK(int layer_idx, int from, int to_len, int* to);

void Internal_UnshiftAndDequantizeQK(int layer_idx, int from, int to_len,
                                     int* to);

void Internal_GenerateSecretKey_PV(int layer_idx);

void Internal_GenerateDecryptionKey_PV(int layer_idx, int from_x, int from_y);

void Internal_QuantizeAndShiftP(int layer_idx, int from, int to_len, int* to);

void Internal_QuantizeAndShiftV(int layer_idx, int from, int to_len, int* to);

void Internal_EncryptX_PV(int layer_idx, int from, int to_len, int* to);

void Internal_EncryptY_PV(int layer_idx, int from, int to_len, int* to);

void Internal_Decrypt_PV(int layer_idx, int from, int to_len, int* to);

void Internal_UnshiftAndDequantizePV(int layer_idx, int from, int to_len,
                                     int* to);

void Internal_BookKeeperStore(int loc, float* data, int shape_len, int* shape);

void Internal_BookKeeperLoad(int loc, float* data, int shape_len, int* shape);

void Internal_BookKeeperStore_Uint32(int loc, int* data, int shape_len,
                                     int* shape);

void Internal_BookKeeperLoad_Uint32(int loc, int* data, int shape_len,
                                    int* shape);

void Internal_BookKeeperIsAvailable(int loc, bool* ret);

void Internal_BookKeeperIsAvailable_Uint32(int loc, bool* ret);

#endif  // SECLLM_SECLLM_H