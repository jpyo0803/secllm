#include "secllm_interface.h"
#include "secllm.h"

extern "C" {

void Ext_PrintTest(int a, int b) {
  Internal_PrintTest(a, b);
}

void Ext_CreateSecLLM(int hidden_size, int intermediate_size,
                      int max_position_embeddings, int num_attention_heads,
                      int num_hidden_layers, int num_key_value_heads,
                      int enc_key_pool_size) {
  Internal_CreateSecLLM(hidden_size, intermediate_size, max_position_embeddings,
                        num_attention_heads, num_hidden_layers,
                        num_key_value_heads, enc_key_pool_size);
}

void Ext_Softmax_InPlace(float* x, int B, int M, int N, int K) {
  Internal_Softmax_InPlace(x, B, M, N, K);
}

void Ext_Softmax(int from, int to_len, int* to) {
  Internal_Softmax(from, to_len, to);
}

void Ext_SwiGLU_InPlace(float* gate_in, float* up_in, int B, int M, int N) {
  Internal_SwiGLU_InPlace(gate_in, up_in, B, M, N);
}

void Ext_SwiGLU(int from1, int from2, int to_len, int* to) {
  Internal_SwiGLU(from1, from2, to_len, to);
}

void Ext_RMSNorm_InPlace(float* x, const float* const weight, int B, int M,
                         int N, float eps) {
  Internal_RMSNorm_InPlace(x, weight, B, M, N, eps);
}

void Ext_RMSNorm(int from, int to_len, int* to, const float* const weight,
                 float eps) {
  Internal_RMSNorm(from, to_len, to, weight, eps);
}

void Ext_ElementWiseAdd_InPlace(float* x, float* y, int B, int M, int N) {
  Internal_ElementWiseAdd_InPlace(x, y, B, M, N);
}

void Ext_ElementWiseAdd(int from1, int from2, int to_len, int* to) {
  Internal_ElementWiseAdd(from1, from2, to_len, to);
}

void Ext_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                           const float* const cos, const float* const sin,
                           int B, int Q_M, int K_M, int N, int K) {
  Internal_ApplyRotaryPosEmb(q_tensor, k_tensor, cos, sin, B, Q_M, K_M, N, K);
}

void Ext_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                              const float* const position_ids,
                              int position_ids_M, float* cos, float* sin) {
  Internal_LlamaRotaryEmbedding(inv_freq, inv_freq_M, position_ids,
                                position_ids_M, cos, sin);
}

uint32_t Ext_GenerateCPRNG() {
  return Internal_GenerateCPRNG();
}

uint32_t Ext_GenerateMultKey() {
  return Internal_GenerateMultKey();
}

uint32_t Ext_GenerateAddKey() {
  return Internal_GenerateAddKey();
}

void Ext_Reset() {
  Internal_Reset();
}

// TODO(jpyo0803): Where is its declaration?
void Ext_BookKeeperStore(int loc, float* data, int shape_len, int* shape) {
  Internal_BookKeeperStore(loc, data, shape_len, shape);
}

void Ext_BookKeeperStore_Uint32(int loc, int* data, int shape_len, int* shape) {
  Internal_BookKeeperStore_Uint32(loc, data, shape_len, shape);
}

void Ext_BookKeeperLoad(int loc, float* out, int shape_len, int* shape) {
  Internal_BookKeeperLoad(loc, out, shape_len, shape);
}

void Ext_BookKeeperLoad_Uint32(int loc, int* out, int shape_len, int* shape) {
  Internal_BookKeeperLoad_Uint32(loc, out, shape_len, shape);
}

void Ext_ReplicateTensor(int from, int* to, int to_len) {
  Internal_ReplicateTensor(from, to, to_len);
}

void Ext_ReplicateTensor_Uint32(int from, int* to, int to_len) {
  Internal_ReplicateTensor_Uint32(from, to, to_len);
}

void Ext_GetCprngTensor(int* out, int shape_len, int* shape) {
  Internal_GetCprngTensor(out, shape_len, shape);
}

void Ext_SetEncKeyAndDecKey(int layer_idx, int* src_enc_key_pool,
                            int* src_dec_key, int type) {
  Internal_SetEncKeyAndDecKey(layer_idx, src_enc_key_pool, src_dec_key, type);
}

void Ext_SetLinearWeightScales(int layer_idx, float* scales, int len,
                               int type) {
  Internal_SetLinearWeightScales(layer_idx, scales, len, type);
}

void Ext_EncryptLinearActivation(int layer_idx, int* out, int from, int type) {
  Internal_EncryptLinearActivation(layer_idx, out, from, type);
}

void Ext_DecryptLinearActivation(int layer_idx, int to_len, int* to,
                                 int* enc_tensor, int shape_len, int* shape,
                                 int type) {
  Internal_DecryptLinearActivation(layer_idx, to_len, to, enc_tensor, shape_len,
                                   shape, type);
}

void Ext_SetQKVOutputScales(int layer_idx, float q_output_scale,
                            float k_output_scale, float v_output_scale) {
  Internal_SetQKVOutputScales(layer_idx, q_output_scale, k_output_scale,
                              v_output_scale);
}

void Ext_QuantizeAndShiftQ(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeAndShiftQ(layer_idx, from, to_len, to);
}

void Ext_QuantizeAndShiftK(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeAndShiftK(layer_idx, from, to_len, to);
}

void Ext_UnshiftAndDequantizeQK(int layer_idx, int from, int to_len, int* to) {
  Internal_UnshiftAndDequantizeQK(layer_idx, from, to_len, to);
}

void Ext_QuantizeAndShiftP(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeAndShiftP(layer_idx, from, to_len, to);
}

void Ext_QuantizeAndShiftV(int layer_idx, int from, int to_len, int* to) {
  Internal_QuantizeAndShiftV(layer_idx, from, to_len, to);
}

void Ext_UnshiftAndDequantizePV(int layer_idx, int from, int to_len, int* to) {
  Internal_UnshiftAndDequantizePV(layer_idx, from, to_len, to);
}

void Ext_SetAttentionMask(float* mask, int M, int N) {
  Internal_SetAttentionMask(mask, M, N);
}

void Ext_SetBatchSizeAndTokenLength(int layer_idx, int bsz, int token_length) {
  Internal_SetBatchSizeAndTokenLength(layer_idx, bsz, token_length);
}

void Ext_GenerateSecretKey_QK(int layer_idx) {
  Internal_GenerateSecretKey_QK(layer_idx);
}

void Ext_GenerateDecryptionKey_QK(int layer_idx, int from_x, int from_y) {
  Internal_GenerateDecryptionKey_QK(layer_idx, from_x, from_y);
}

void Ext_EncryptX_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_EncryptX_QK(layer_idx, from, to_len, to);
}

void Ext_EncryptY_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_EncryptY_QK(layer_idx, from, to_len, to);
}

void Ext_Decrypt_QK(int layer_idx, int from, int to_len, int* to) {
  Internal_Decrypt_QK(layer_idx, from, to_len, to);
}

void Ext_GenerateSecretKey_PV(int layer_idx) {
  Internal_GenerateSecretKey_PV(layer_idx);
}

void Ext_GenerateDecryptionKey_PV(int layer_idx, int from_x, int from_y) {
  Internal_GenerateDecryptionKey_PV(layer_idx, from_x, from_y);
}

void Ext_EncryptX_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_EncryptX_PV(layer_idx, from, to_len, to);
}

void Ext_EncryptY_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_EncryptY_PV(layer_idx, from, to_len, to);
}

void Ext_Decrypt_PV(int layer_idx, int from, int to_len, int* to) {
  Internal_Decrypt_PV(layer_idx, from, to_len, to);
}

void Ext_BookKeeperIsAvailable(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable(loc, ret);
}

void Ext_BookKeeperIsAvailable_Uint32(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Uint32(loc, ret);
}
}