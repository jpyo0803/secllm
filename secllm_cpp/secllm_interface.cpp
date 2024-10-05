#include "secllm_interface.h"
#include "secllm.h"
#include "thread_pool.h"

#include <array>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>
#include "time_stamp.h"
#include "types.h"

namespace {
std::unique_ptr<jpyo0803::ThreadPool> thread_pool;
constexpr int kNumThreads = 1;
std::array<std::vector<jpyo0803::TimeStamp>, kNumThreads> time_stamps;

std::array<std::string, 7> proj_type_str = {"Q",    "K",  "V",   "O",
                                            "Gate", "Up", "Down"};
}  // namespace

extern "C" {

void Ext_CreateSecLLM(int hidden_size, int intermediate_size,
                      int max_position_embeddings, int num_attention_heads,
                      int num_hidden_layers, int num_key_value_heads,
                      int enc_key_pool_size) {
  Internal_CreateSecLLM(hidden_size, intermediate_size, max_position_embeddings,
                        num_attention_heads, num_hidden_layers,
                        num_key_value_heads, enc_key_pool_size);

  thread_pool = std::make_unique<jpyo0803::ThreadPool>(kNumThreads);
}

void Ext_Softmax(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "Softmax");
    ts.Start();
    Internal_Softmax(from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_SwiGLU(int layer_idx, int from1, int from2, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "SwiGLU");
    ts.Start();
    Internal_SwiGLU(from1, from2, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

// Dont need threading, it is in the critical node
void Ext_RMSNorm(int layer_idx, int from, int to_len, int* to, int type) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id,
                           "RMSNorm " + std::to_string(type + 1));
    ts.Start();
    Internal_RMSNorm(layer_idx, from, locs, type);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
  // Internal_RMSNorm(layer_idx, from, locs, type);
}

// Dont need threading, it is in the critical node
void Ext_ElementWiseAdd(int layer_idx, int from1, int from2, int to_len,
                        int* to, int type) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id,
                           "Residual Add " + std::to_string(type + 1));
    ts.Start();
    Internal_ElementWiseAdd(from1, from2, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

// Not threaded yet
void Ext_LlamaRotaryEmbedding(const float* const inv_freq, int inv_freq_M,
                              const float* const position_ids,
                              int position_ids_M, float* cos, float* sin) {
  Internal_LlamaRotaryEmbedding(inv_freq, inv_freq_M, position_ids,
                                position_ids_M, cos, sin);
}

// Not threaded yet
void Ext_ApplyRotaryPosEmb(float* q_tensor, float* k_tensor,
                           const float* const cos, const float* const sin,
                           int B, int Q_M, int K_M, int N, int K) {
  Internal_ApplyRotaryPosEmb(q_tensor, k_tensor, cos, sin, B, Q_M, K_M, N, K);
}

void Ext_QuantizeLinearActivation(int layer_idx, int from, int to_len, int* to,
                                  int type) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id,
                           "[" + proj_type_str[type] + "] Quantize input");
    ts.Start();
    Internal_QuantizeLinearActivation(
        layer_idx, from, locs, static_cast<jpyo0803::ProjectionType>(type));
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_EncryptLinearActivation(int layer_idx, int from, int to_len, int* to,
                                 int type) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id,
                           "[" + proj_type_str[type] + "] Encrypt input");
    ts.Start();
    Internal_EncryptLinearActivation(
        layer_idx, from, locs, static_cast<jpyo0803::ProjectionType>(type));
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_DecryptLinearActivation(int layer_idx, int from, int to_len, int* to,
                                 int type) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id,
                           "[" + proj_type_str[type] + " proj] Decrypt result");
    ts.Start();
    Internal_DecryptLinearActivation(
        layer_idx, from, locs, static_cast<jpyo0803::ProjectionType>(type));
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_DequantizeLinearActivation(int layer_idx, int from, int to_len,
                                    int* to, int type) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(
        layer_idx, thread_id,
        "[" + proj_type_str[type] + " proj] Dequantize result");
    ts.Start();
    Internal_DequantizeLinearActivation(
        layer_idx, from, locs, static_cast<jpyo0803::ProjectionType>(type));
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_QuantizeQ_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Quantize Q input");
    ts.Start();
    Internal_QuantizeQ_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_ShiftQ_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Shift Q input");
    ts.Start();
    Internal_ShiftQ_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_QuantizeK_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Quantize K input");
    ts.Start();
    Internal_QuantizeK_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_ShiftK_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Shift K input");
    ts.Start();
    Internal_ShiftK_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_Unshift_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Unshift result");
    ts.Start();
    Internal_Unshift_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_Dequantize_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Dequantize result");
    ts.Start();
    Internal_Dequantize_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_QuantizeP_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Quantize P input");
    ts.Start();
    Internal_QuantizeP_PV(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_ShiftP_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Shift P input");
    ts.Start();
    Internal_ShiftP_PV(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_QuantizeV_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Quantize V input");
    ts.Start();
    Internal_QuantizeV_PV(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_ShiftV_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Shift V input");
    ts.Start();
    Internal_ShiftV_PV(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_Unshift_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Unshift result");
    ts.Start();
    Internal_Unshift_PV(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_Dequantize_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Dequantize result");
    ts.Start();
    Internal_Dequantize_PV(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_GenerateSecretKey_QK(int layer_idx) {
  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Key Generation");
    ts.Start();
    Internal_GenerateSecretKey_QK(layer_idx);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_GenerateDecryptionKey_QK(int layer_idx, int from_x, int from_y) {
  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Dec. Key Generation");
    ts.Start();
    Internal_GenerateDecryptionKey_QK(layer_idx, from_x, from_y);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_EncryptX_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Encrypt Q input");
    ts.Start();
    Internal_EncryptX_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_EncryptY_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Encrypt K input");
    ts.Start();
    Internal_EncryptY_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_Decrypt_QK(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[QK^T] Decrypt result");
    ts.Start();
    Internal_Decrypt_QK(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_GenerateSecretKey_PV(int layer_idx) {
  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Key Generation");
    ts.Start();
    Internal_GenerateSecretKey_PV(layer_idx);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_GenerateDecryptionKey_PV(int layer_idx, int from_x, int from_y) {
  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Dec. Key Generation");
    ts.Start();
    Internal_GenerateDecryptionKey_PV(layer_idx, from_x, from_y);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_EncryptX_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Encrypt P input");
    ts.Start();
    Internal_EncryptX_PV(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_EncryptY_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> to_vec(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Encrypt V input");
    ts.Start();
    Internal_EncryptY_PV(layer_idx, from, to_vec);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

void Ext_Decrypt_PV(int layer_idx, int from, int to_len, int* to) {
  std::vector<int> locs(to, to + to_len);

  thread_pool->enqueue_task([=](int thread_id) {
    jpyo0803::TimeStamp ts(layer_idx, thread_id, "[PV] Decrypt result");
    ts.Start();
    Internal_Decrypt_PV(layer_idx, from, locs);
    ts.End();
    time_stamps[thread_id].push_back(ts);
  });
}

////////////////////////// Functions below do not neet multi-threading //////////////////////////

void Ext_SetQKVOutputScales(int layer_idx, float q_output_scale,
                            float k_output_scale, float v_output_scale) {
  Internal_SetQKVOutputScales(layer_idx, q_output_scale, k_output_scale,
                              v_output_scale);
}

// No threading needed
void Ext_SetAttentionMask(float* mask, int M, int N) {
  Internal_SetAttentionMask(mask, M, N);
}

// No threading needed
void Ext_SetBatchSizeAndTokenLength(int layer_idx, int bsz, int token_length) {
  Internal_SetBatchSizeAndTokenLength(layer_idx, bsz, token_length);
}

// No threading needed
void Ext_BookKeeperIsAvailable_Float(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Float(loc, ret);
}

// No threading needed
void Ext_BookKeeperIsAvailable_Int32(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Int32(loc, ret);
}

// No threading needed
void Ext_BookKeeperIsAvailable_Uint32(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Uint32(loc, ret);
}

// No threading needed
void Ext_BookKeeperIsAvailable_Int8(int loc, bool* ret) {
  Internal_BookKeeperIsAvailable_Int8(loc, ret);
}

// No threading needed
void Ext_QKKeyIsAvailable(int layer_idx, bool* ret) {
  Internal_QKKeyIsAvailable(layer_idx, ret);
}

// No threading needed
void Ext_PVKeyIsAvailable(int layer_idx, bool* ret) {
  Internal_PVKeyIsAvailable(layer_idx, ret);
}

// No threading needed
void Ext_QKDecKeyIsAvailable(int layer_idx, bool* ret) {
  Internal_QKDecKeyIsAvailable(layer_idx, ret);
}

// No threading needed
void Ext_PVDecKeyIsAvailable(int layer_idx, bool* ret) {
  Internal_PVDecKeyIsAvailable(layer_idx, ret);
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

// No threading needed
void Ext_BookKeeperStore_Float(int loc, float* data, int shape_len,
                               int* shape) {
  Internal_BookKeeperStore_Float(loc, data, shape_len, shape);
}

// No threading needed
void Ext_BookKeeperStore_Int32(int loc, int32_t* data, int shape_len,
                               int* shape) {
  Internal_BookKeeperStore_Int32(loc, data, shape_len, shape);
}

// No threading needed
void Ext_BookKeeperStore_Uint32(int loc, uint32_t* data, int shape_len,
                                int* shape) {
  Internal_BookKeeperStore_Uint32(loc, data, shape_len, shape);
}

// No threading needed
void Ext_BookKeeperStore_Int8(int loc, int8_t* data, int shape_len,
                              int* shape) {
  Internal_BookKeeperStore_Int8(loc, data, shape_len, shape);
}

// No threading needed
void Ext_BookKeeperLoad_Float(int loc, float* out, int shape_len, int* shape) {
  Internal_BookKeeperLoad_Float(loc, out, shape_len, shape);
}

// No threading needed
void Ext_BookKeeperLoad_Int32(int loc, int32_t* out, int shape_len,
                              int* shape) {
  Internal_BookKeeperLoad_Int32(loc, out, shape_len, shape);
}

// No threading needed
void Ext_BookKeeperLoad_Uint32(int loc, uint32_t* out, int shape_len,
                               int* shape) {
  Internal_BookKeeperLoad_Uint32(loc, out, shape_len, shape);
}

// No threading needed
void Ext_BookKeeperLoad_Int8(int loc, int8_t* out, int shape_len, int* shape) {
  Internal_BookKeeperLoad_Int8(loc, out, shape_len, shape);
}

// No threading needed
void Ext_ReplicateTensor_Float(int from, int* to, int to_len) {
  Internal_ReplicateTensor_Float(from, to, to_len);
}

// No threading needed
void Ext_ReplicateTensor_Uint32(int from, int* to, int to_len) {
  Internal_ReplicateTensor_Uint32(from, to, to_len);
}

void Ext_GetCprngTensor(int* out, int shape_len, int* shape) {
  Internal_GetCprngTensor(out, shape_len, shape);
}

void Ext_SetEncKeyAndDecKey(int layer_idx, int* src_enc_key_pool,
                            int* src_dec_key, int type) {
  Internal_SetEncKeyAndDecKey(layer_idx, src_enc_key_pool, src_dec_key,
                              static_cast<jpyo0803::ProjectionType>(type));
}

void Ext_SetLinearWeightScales(int layer_idx, float* scales, int len,
                               int type) {
  Internal_SetLinearWeightScales(layer_idx, scales, len,
                                 static_cast<jpyo0803::ProjectionType>(type));
}

void Ext_SetRMSNormWeight(int layer_idx, float* weight, float eps, int type) {
  Internal_SetRMSNormWeight(layer_idx, weight, eps, type);
}

void Ext_PrintTest(int a, int b) {
  Internal_PrintTest(a, b);
}

void Ext_BookKeeperGetShapeLength_Float(int loc, int* out) {
  Internal_BookKeeperGetShapeLength_Float(loc, out);
}

void Ext_BookKeeperGetShapeLength_Int32(int loc, int* out) {
  Internal_BookKeeperGetShapeLength_Int32(loc, out);
}

void Ext_BookKeeperGetShapeLength_Uint32(int loc, int* out) {
  Internal_BookKeeperGetShapeLength_Uint32(loc, out);
}

void Ext_BookKeeperGetShapeLength_Int8(int loc, int* out) {
  Internal_BookKeeperGetShapeLength_Int8(loc, out);
}

void Ext_BookKeeperGetShape_Float(int loc, int* out) {
  Internal_BookKeeperGetShape_Float(loc, out);
}

void Ext_BookKeeperGetShape_Int32(int loc, int* out) {
  Internal_BookKeeperGetShape_Int32(loc, out);
}

void Ext_BookKeeperGetShape_Uint32(int loc, int* out) {
  Internal_BookKeeperGetShape_Uint32(loc, out);
}

void Ext_BookKeeperGetShape_Int8(int loc, int* out) {
  Internal_BookKeeperGetShape_Int8(loc, out);
}

void Ext_Close(const char* output_filename) {
  // shutdown the thread pool
  thread_pool->shutdown();

  // Flush the time stamps
  // open the file named output_filename, do not overwrite
  std::ofstream file(output_filename, std::ios::app);

  // write the time stamps to the file
  for (int i = 0; i < kNumThreads; ++i) {
    for (const auto& ts : time_stamps[i]) {
      file << ts.ToString() << std::endl;
    }
  }
}
}