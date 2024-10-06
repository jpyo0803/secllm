#ifndef SECLLM_SECLLM_H
#define SECLLM_SECLLM_H

#include <stdint.h>
#include <memory>

#include "book_keeper.h"
#include "decoder_layer.h"
#include "tensor.h"
#include "types.h"

namespace jpyo0803 {

class SecLLM {
 public:
  SecLLM(int hidden_size, int intermediate_size, int max_position_embeddings,
         int num_attention_heads, int num_hidden_layers,
         int num_key_value_heads, int enc_key_pool_size);

 public:
  void Reset();

  // member methods
  std::shared_ptr<Tensor<float>> BookKeeperLoad_Float(int loc);
  void BookKeeperStore_Float(std::vector<int> locs,
                             std::shared_ptr<Tensor<float>>& data);

  std::shared_ptr<Tensor<uint32_t>> BookKeeperLoad_Uint32(int loc);
  void BookKeeperStore_Uint32(std::vector<int> locs,
                              std::shared_ptr<Tensor<uint32_t>>& data);

  std::shared_ptr<Tensor<int32_t>> BookKeeperLoad_Int32(int loc);
  void BookKeeperStore_Int32(std::vector<int> locs,
                             std::shared_ptr<Tensor<int32_t>>& data);

  std::shared_ptr<Tensor<int8_t>> BookKeeperLoad_Int8(int loc);
  void BookKeeperStore_Int8(std::vector<int> locs,
                            std::shared_ptr<Tensor<int8_t>>& data);

  bool BookKeeperIsAvailable_Float(int loc);
  bool BookKeeperIsAvailable_Int32(int loc);
  bool BookKeeperIsAvailable_Uint32(int loc);
  bool BookKeeperIsAvailable_Int8(int loc);

  int BookKeeperGetShapeLength_Float(int loc);
  int BookKeeperGetShapeLength_Int32(int loc);
  int BookKeeperGetShapeLength_Uint32(int loc);
  int BookKeeperGetShapeLength_Int8(int loc);

  void BookKeeperGetShape_Float(int loc, int* out);
  void BookKeeperGetShape_Int32(int loc, int* out);
  void BookKeeperGetShape_Uint32(int loc, int* out);
  void BookKeeperGetShape_Int8(int loc, int* out);

  void SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool, int* dec_key,
                          ProjectionType type);

  void SetLinearWeightScales(int layer_idx, float* weight_scale, int len,
                             ProjectionType type);

  void SetRMSNormWeight(int layer_idx, float* weight, float eps,
                        int type);  // type 0:

  void RMSNorm(int layer_idx, std::shared_ptr<Tensor<float>> out,
               std::shared_ptr<Tensor<float>> in, int type);

  void QuantizeLinearActivation(int layer_idx,
                                std::shared_ptr<Tensor<int8_t>> out,
                                std::shared_ptr<Tensor<float>> in,
                                ProjectionType type);

  void EncryptLinearActivation(int layer_idx,
                               std::shared_ptr<Tensor<int32_t>> out,
                               std::shared_ptr<Tensor<int8_t>> in,
                               ProjectionType type);

  void DecryptLinearActivation(int layer_idx,
                               std::shared_ptr<Tensor<int32_t>> out,
                               std::shared_ptr<Tensor<int32_t>> in,
                               ProjectionType type);

  void DequantizeLinearActivation(int layer_idx,
                                  std::shared_ptr<Tensor<float>> out,
                                  std::shared_ptr<Tensor<int32_t>> in,
                                  ProjectionType type);

  void SetQKVOutputScales(int layer_idx, float q_output_scale,
                          float k_output_scale, float v_output_scale);

  void QuantizeQ_QK(int layer_idx, std::shared_ptr<Tensor<int8_t>> out,
                    std::shared_ptr<Tensor<float>> in);
  void ShiftQ_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                 std::shared_ptr<Tensor<int8_t>> in);

  void QuantizeK_QK(int layer_idx, std::shared_ptr<Tensor<int8_t>> out,
                    std::shared_ptr<Tensor<float>> in);
  void ShiftK_QK(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                 std::shared_ptr<Tensor<int8_t>> in);

  void Unshift_QK(int layer_idx, std::shared_ptr<Tensor<int32_t>> out,
                  std::shared_ptr<Tensor<uint32_t>> in);
  void Dequantize_QK(int layer_idx, std::shared_ptr<Tensor<float>> out,
                     std::shared_ptr<Tensor<int32_t>> in);

  void QuantizeP_PV(int layer_idx, std::shared_ptr<Tensor<int8_t>> out,
                    std::shared_ptr<Tensor<float>> in);
  void ShiftP_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                 std::shared_ptr<Tensor<int8_t>> in);
  void QuantizeV_PV(int layer_idx, std::shared_ptr<Tensor<int8_t>> out,
                    std::shared_ptr<Tensor<float>> in);
  void ShiftV_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                 std::shared_ptr<Tensor<int8_t>> in);

  void Unshift_PV(int layer_idx, std::shared_ptr<Tensor<int32_t>> out,
                  std::shared_ptr<Tensor<uint32_t>> in);
  void Dequantize_PV(int layer_idx, std::shared_ptr<Tensor<float>> out,
                     std::shared_ptr<Tensor<int32_t>> in);

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

  void GenerateDecAddBuffer_QK(int layer_idx);
  void GenerateDecMultBuffer_QK(int layer_idx);
  void GenerateUnshiftBuffer_QK(int layer_idx);

  void EncryptX_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void EncryptY_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                   std::shared_ptr<Tensor<uint32_t>> in);

  void Decrypt_PV(int layer_idx, std::shared_ptr<Tensor<uint32_t>> out,
                  std::shared_ptr<Tensor<uint32_t>> in);

  void GenerateDecAddBuffer_PV(int layer_idx);
  void GenerateDecMultBuffer_PV(int layer_idx);
  void GenerateUnshiftBuffer_PV(int layer_idx);

  bool QKKeyIsAvailable(int layer_idx);
  bool QKDecKeyIsAvailable(int layer_idx);
  bool QKDecAddBufferIsAvailable(int layer_idx);
  bool QKDecMultBufferIsAvailable(int layer_idx);

  bool QKShiftedQIsAvailable(int layer_idx);
  bool QKShiftedKIsAvailable(int layer_idx);
  bool QKUnshiftBufferIsAvailable(int layer_idx);

  bool PVKeyIsAvailable(int layer_idx);
  bool PVDecKeyIsAvailable(int layer_idx);
  bool PVDecAddBufferIsAvailable(int layer_idx);
  bool PVDecMultBufferIsAvailable(int layer_idx);

  bool PVShiftedPIsAvailable(int layer_idx);
  bool PVShiftedVIsAvailable(int layer_idx);
  bool PVUnshiftBufferIsAvailable(int layer_idx);

 private:
  int num_hidden_layers_ = -1;

  std::unique_ptr<BookKeeper<Tensor<float>>> book_keeper_float_;

  // NOTE(jpyo0803): Although QK^T and PV operation will be done in uint32,
  // here I use int32 since torch does not support uint32.
  std::unique_ptr<BookKeeper<Tensor<int32_t>>> book_keeper_int32_;
  std::unique_ptr<BookKeeper<Tensor<uint32_t>>> book_keeper_uint32_;
  std::unique_ptr<BookKeeper<Tensor<int8_t>>> book_keeper_int8_;

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

void Internal_Softmax(int from, std::vector<int> locs);

void Internal_SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N);

void Internal_SwiGLU(int from1, int from2, std::vector<int> locs);

void Internal_RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                              int N, float eps);

void Internal_RMSNorm(int layer_idx, int from, std::vector<int> locs, int type);

void Internal_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N);

void Internal_ElementWiseAdd(int from1, int from2, std::vector<int> locs);

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

void Internal_ReplicateTensor_Float(int from, int* to, int to_len);
void Internal_ReplicateTensor_Int32(int from, int* to, int to_len);
void Internal_ReplicateTensor_Uint32(int from, int* to, int to_len);
void Internal_ReplicateTensor_Int8(int from, int* to, int to_len);

void Internal_GetCprngTensor(int* out, int shape_len, int* shape);

void Internal_SetEncKeyAndDecKey(int layer_idx, int* enc_key_pool, int* dec_key,
                                 jpyo0803::ProjectionType type);

void Internal_SetLinearWeightScales(int layer_idx, float* scales, int len,
                                    jpyo0803::ProjectionType type);

void Internal_SetRMSNormWeight(int layer_idx, float* weight, float eps,
                               int type);

void Internal_QuantizeLinearActivation(int layer_idx, int from,
                                       std::vector<int> locs,
                                       jpyo0803::ProjectionType type);

void Internal_EncryptLinearActivation(int layer_idx, int from,
                                      std::vector<int> locs,
                                      jpyo0803::ProjectionType type);

void Internal_DecryptLinearActivation(int layer_idx, int from,
                                      std::vector<int> locs,
                                      jpyo0803::ProjectionType type);

void Internal_DequantizeLinearActivation(int layer_idx, int from,
                                         std::vector<int> locs,
                                         jpyo0803::ProjectionType type);

void Internal_SetQKVOutputScales(int layer_idx, float q_output_scale,
                                 float k_output_scale, float v_output_scale);

void Internal_QuantizeAndShiftQ(int layer_idx, int from, std::vector<int> locs);

void Internal_QuantizeAndShiftK(int layer_idx, int from, std::vector<int> locs);

void Internal_SetAttentionMask(float* mask, int M, int N);

void Internal_SetBatchSizeAndTokenLength(int layer_idx, int bsz,
                                         int token_length);

void Internal_GenerateSecretKey_QK(int layer_idx);
void Internal_GenerateDecryptionKey_QK(int layer_idx, int from_x, int from_y);
void Internal_GenerateDecAddBuffer_QK(int layer_idx);
void Internal_GenerateDecMultBuffer_QK(int layer_idx);
void Internal_GenerateUnshiftBuffer_QK(int layer_idx);

void Internal_QuantizeQ_QK(int layer_idx, int from, std::vector<int> locs);
void Internal_ShiftQ_QK(int layer_idx, int from, std::vector<int> locs);
void Internal_QuantizeK_QK(int layer_idx, int from, std::vector<int> locs);
void Internal_ShiftK_QK(int layer_idx, int from, std::vector<int> locs);
void Internal_EncryptX_QK(int layer_idx, int from, std::vector<int> locs);
void Internal_EncryptY_QK(int layer_idx, int from, std::vector<int> locs);
void Internal_Decrypt_QK(int layer_idx, int from, std::vector<int> locs);
void Internal_Unshift_QK(int layer_idx, int from, std::vector<int> locs);
void Internal_Dequantize_QK(int layer_idx, int from, std::vector<int> locs);

void Internal_GenerateSecretKey_PV(int layer_idx);
void Internal_GenerateDecryptionKey_PV(int layer_idx, int from_x, int from_y);
void Internal_GenerateDecAddBuffer_PV(int layer_idx);
void Internal_GenerateDecMultBuffer_PV(int layer_idx);
void Internal_GenerateUnshiftBuffer_PV(int layer_idx);

void Internal_QuantizeP_PV(int layer_idx, int from, std::vector<int> locs);
void Internal_ShiftP_PV(int layer_idx, int from, std::vector<int> locs);
void Internal_QuantizeV_PV(int layer_idx, int from, std::vector<int> locs);
void Internal_ShiftV_PV(int layer_idx, int from, std::vector<int> locs);
void Internal_EncryptX_PV(int layer_idx, int from, std::vector<int> locs);
void Internal_EncryptY_PV(int layer_idx, int from, std::vector<int> locs);
void Internal_Decrypt_PV(int layer_idx, int from, std::vector<int> locs);
void Internal_Unshift_PV(int layer_idx, int from, std::vector<int> locs);
void Internal_Dequantize_PV(int layer_idx, int from, std::vector<int> locs);

void Internal_BookKeeperStore_Float(int loc, float* data, int shape_len,
                                    int* shape);
void Internal_BookKeeperStore_Int32(int loc, int32_t* data, int shape_len,
                                    int* shape);
void Internal_BookKeeperStore_Uint32(int loc, uint32_t* data, int shape_len,
                                     int* shape);
void Internal_BookKeeperStore_Int8(int loc, int8_t* data, int shape_len,
                                   int* shape);

void Internal_BookKeeperLoad_Float(int loc, float* out, int shape_len,
                                   int* shape);
void Internal_BookKeeperLoad_Int32(int loc, int32_t* out, int shape_len,
                                   int* shape);
void Internal_BookKeeperLoad_Uint32(int loc, uint32_t* out, int shape_len,
                                    int* shape);
void Internal_BookKeeperLoad_Int8(int loc, int8_t* out, int shape_len,
                                  int* shape);

void Internal_BookKeeperIsAvailable_Float(int loc, bool* ret);
void Internal_BookKeeperIsAvailable_Int32(int loc, bool* ret);
void Internal_BookKeeperIsAvailable_Uint32(int loc, bool* ret);
void Internal_BookKeeperIsAvailable_Int8(int loc, bool* ret);

void Internal_BookKeeperGetShapeLength_Float(int loc, int* ret);
void Internal_BookKeeperGetShapeLength_Int32(int loc, int* ret);
void Internal_BookKeeperGetShapeLength_Uint32(int loc, int* ret);
void Internal_BookKeeperGetShapeLength_Int8(int loc, int* ret);

void Internal_BookKeeperGetShape_Float(int loc, int* out);
void Internal_BookKeeperGetShape_Int32(int loc, int* out);
void Internal_BookKeeperGetShape_Uint32(int loc, int* out);
void Internal_BookKeeperGetShape_Int8(int loc, int* out);

void Internal_QKKeyIsAvailable(int layer_idx, bool* ret);
void Internal_QKDecKeyIsAvailable(int layer_idx, bool* ret);
void Internal_QKDecAddBufferIsAvailable(int layer_idx, bool* ret);
void Internal_QKDecMultBufferIsAvailable(int layer_idx, bool* ret);

void Internal_QKShiftedQIsAvailable(int layer_idx, bool* ret);
void Internal_QKShiftedKIsAvailable(int layer_idx, bool* ret);

void Internal_QKUnshiftBufferIsAvailable(int layer_idx, bool* ret);

void Internal_PVKeyIsAvailable(int layer_idx, bool* ret);
void Internal_PVDecKeyIsAvailable(int layer_idx, bool* ret);
void Internal_PVDecAddBufferIsAvailable(int layer_idx, bool* ret);
void Internal_PVDecMultBufferIsAvailable(int layer_idx, bool* ret);

void Internal_PVShiftedPIsAvailable(int layer_idx, bool* ret);
void Internal_PVShiftedVIsAvailable(int layer_idx, bool* ret);

void Internal_PVUnshiftBufferIsAvailable(int layer_idx, bool* ret);

#endif  // SECLLM_SECLLM_H