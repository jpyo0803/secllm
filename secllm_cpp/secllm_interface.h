#ifndef SECLLM_CPP_SECLLM_INTERFACE_H
#define SECLLM_CPP_SECLLM_INTERFACE_H

#include "secllm.h"

extern "C" {

void Ext_PrintTest(int a, int b);

void Ext_CreateSecLLM(int hidden_size, int intermediate_size,
                      int max_position_embeddings, int num_attention_heads,
                      int num_hidden_layers, int num_key_value_heads,
                      int enc_key_pool_size);

void Ext_Softmax(int from, int to_len, int* to);

void Ext_SwiGLU(int from1, int from2, int to_len, int* to);

void Ext_RMSNorm(int from, int to_len, int* to, const float* const weight,
                 float eps);
void Ext_ElementWiseAdd(int from1, int from2, int to_len, int* to);

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

// TODO(jpyo0803): Where is its declaration?
void Ext_BookKeeperStore_Float(int loc, float* data, int shape_len, int* shape);
void Ext_BookKeeperStore_Int32(int loc, int32_t* data, int shape_len, int* shape);
void Ext_BookKeeperStore_Uint32(int loc, uint32_t* data, int shape_len, int* shape);
void Ext_BookKeeperStore_Int8(int loc, int8_t* data, int shape_len, int* shape);

void Ext_BookKeeperLoad_Float(int loc, float* out, int shape_len, int* shape);
void Ext_BookKeeperLoad_Int32(int loc, int32_t* out, int shape_len, int* shape);
void Ext_BookKeeperLoad_Uint32(int loc, uint32_t* out, int shape_len, int* shape);
void Ext_BookKeeperLoad_Int8(int loc, int8_t* out, int shape_len, int* shape);

void Ext_ReplicateTensor_Float(int from, int* to, int to_len);
void Ext_ReplicateTensor_Int32(int from, int* to, int to_len);
void Ext_ReplicateTensor_Uint32(int from, int* to, int to_len);
void Ext_ReplicateTensor_Int8(int from, int* to, int to_len);

void Ext_GetCprngTensor(int* out, int shape_len, int* shape);

void Ext_SetEncKeyAndDecKey(int layer_idx, int* src_enc_key_pool,
                            int* src_dec_key, int type);

void Ext_SetLinearWeightScales(int layer_idx, float* scales, int len, int type);

void Ext_QuantizeLinearActivation(int layer_idx, int from, int to_len, int* to,
                                  int type);

void Ext_EncryptLinearActivation(int layer_idx, int from, int to_len, int* to,
                                 int type);

void Ext_DecryptLinearActivation(int layer_idx, int from, int to_len, int* to,
                                 int type);

void Ext_DequantizeLinearActivation(int layer_idx, int from, int to_len, int* to,
                                   int type);

void Ext_SetQKVOutputScales(int layer_idx, float q_output_scale,
                            float k_output_scale, float v_output_scale);


void Ext_QuantizeQ_QK(int layer_idx, int from, int to_len, int* to);
void Ext_ShiftQ_QK(int layer_idx, int from, int to_len, int* to);
void Ext_QuantizeK_QK(int layer_idx, int from, int to_len, int* to);
void Ext_ShiftK_QK(int layer_idx, int from, int to_len, int* to);
void Ext_Unshift_QK(int layer_idx, int from, int to_len, int* to);
void Ext_Dequantize_QK(int layer_idx, int from, int to_len, int* to);

void Ext_QuantizeP_PV(int layer_idx, int from, int to_len, int* to);
void Ext_ShiftP_PV(int layer_idx, int from, int to_len, int* to);
void Ext_QuantizeV_PV(int layer_idx, int from, int to_len, int* to);
void Ext_ShiftV_PV(int layer_idx, int from, int to_len, int* to);
void Ext_Unshift_PV(int layer_idx, int from, int to_len, int* to);
void Ext_Dequantize_PV(int layer_idx, int from, int to_len, int* to);

void Ext_SetAttentionMask(float* mask, int M, int N);

void Ext_SetBatchSizeAndTokenLength(int layer_idx, int bsz, int token_length);

void Ext_GenerateSecretKey_QK(int layer_idx);

void Ext_GenerateDecryptionKey_QK(int layer_idx, int from_x, int from_y);

void Ext_EncryptX_QK(int layer_idx, int from, int to_len, int* to);

void Ext_EncryptY_QK(int layer_idx, int from, int to_len, int* to);

void Ext_Decrypt_QK(int layer_idx, int from, int to_len, int* to);

void Ext_GenerateSecretKey_PV(int layer_idx);

void Ext_GenerateDecryptionKey_PV(int layer_idx, int from_x, int from_y);

void Ext_EncryptX_PV(int layer_idx, int from, int to_len, int* to);

void Ext_EncryptY_PV(int layer_idx, int from, int to_len, int* to);

void Ext_Decrypt_PV(int layer_idx, int from, int to_len, int* to);

void Ext_BookKeeperIsAvailable_Float(int loc, bool* ret);
void Ext_BookKeeperIsAvailable_Int32(int loc, bool* ret);
void Ext_BookKeeperIsAvailable_Uint32(int loc, bool* ret);
void Ext_BookKeeperIsAvailable_Int8(int loc, bool* ret);

void Ext_QKKeyIsAvailable(int layer_idx, bool* ret);
void Ext_PVKeyIsAvailable(int layer_idx, bool* ret);
void Ext_QKDecKeyIsAvailable(int layer_idx, bool* ret);
void Ext_PVDecKeyIsAvailable(int layer_idx, bool* ret);
}

#endif