
#include <cstdio>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>

#include "sgx_urts.h"
#include "Enclave_u.h"

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define TOKEN_FILENAME   "enclave.token"
#define ENCLAVE_FILENAME "secllm_cpp/enclave.signed.so"

using namespace std::chrono;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
        printf("Error: Unexpected error occurred.\n");
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}

thread_local std::chrono::time_point<std::chrono::high_resolution_clock> start;

void ocall_start_clock()
{
	start = std::chrono::high_resolution_clock::now();
}

void ocall_end_clock(const char * str)
{
	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf(str, elapsed.count());
}

double ocall_get_time()
{
    auto now = std::chrono::high_resolution_clock::now();
	return time_point_cast<microseconds>(now).time_since_epoch().count();
}


extern "C"
{

    /*
     * Initialize the enclave
     */
    unsigned long int initialize_enclave(void)
    {

        std::cout << "Initializing Enclave..." << std::endl;

        sgx_enclave_id_t eid = 0;
        sgx_launch_token_t token = {0};
        sgx_status_t ret = SGX_ERROR_UNEXPECTED;
        int updated = 0;

        /* call sgx_create_enclave to initialize an enclave instance */
        /* Debug Support: set 2nd parameter to 1 */
        ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &eid, NULL);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }

        std::cout << "Enclave id: " << eid << std::endl;

        return eid;
    }

    /*
     * Destroy the enclave
     */
    void destroy_enclave(unsigned long int eid)
    {
        std::cout << "Destroying Enclave with id: " << eid << std::endl;
        sgx_destroy_enclave(eid);
    }

    void Ext_CreateSecLLM(unsigned long int eid, int hidden_size, int intermediate_size,
                               int max_position_embeddings, int num_attention_heads,
                               int num_hidden_layers, int num_key_value_heads,
                               int enc_key_pool_size) {
        sgx_status_t ret = ecall_Internal_CreateSecLLM(eid, hidden_size, intermediate_size,
                                                 max_position_embeddings, num_attention_heads,
                                                 num_hidden_layers, num_key_value_heads,
                                                 enc_key_pool_size);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_Softmax_InPlace(unsigned long int eid, float* x, int B, int M, int N, int K) {
        sgx_status_t ret = ecall_Internal_Softmax_InPlace(eid, x, B, M, N, K);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_Softmax(unsigned long int eid, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Softmax(eid, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_SwiGLU_InPlace(unsigned long int eid, float* gate_in, float* up_in, int B, int M, int N) {
        sgx_status_t ret = ecall_Internal_SwiGLU_InPlace(eid, gate_in, up_in, B, M, N);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_SwiGLU(unsigned long int eid, int from1, int from2, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_SwiGLU(eid, from1, from2, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_RMSNorm_InPlace(unsigned long int eid, float* x, float* weight, int B, int M,
                                int N, float eps) {
        sgx_status_t ret = ecall_Internal_RMSNorm_InPlace(eid, x, weight, B, M, N, eps);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_RMSNorm(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to, int type) {
        sgx_status_t ret = ecall_Internal_RMSNorm(eid, layer_idx, fromm, to_len, to, type);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_ElementWiseAdd_InPlace(unsigned long int eid, float* x, float* y, int B, int M, int N) {
        sgx_status_t ret = ecall_Internal_ElementWiseAdd_InPlace(eid, x, y, B, M, N);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_ElementWiseAdd(unsigned long int eid, int from1, int from2, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_ElementWiseAdd(eid, from1, from2, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_ApplyRotaryPosEmb(unsigned long int eid, float* q_tensor, float* k_tensor,
                                    float* cos, float* sin,
                                    int B, int Q_M, int K_M, int N, int K) {
        sgx_status_t ret = ecall_Internal_ApplyRotaryPosEmb(eid, q_tensor, k_tensor, cos, sin, B, Q_M, K_M, N, K);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                    }

    void Ext_LlamaRotaryEmbedding(unsigned long int eid, float* inv_freq, int inv_freq_M,
                                    float* position_ids,
                                    int position_ids_M, float* cos, float* sin) {
        sgx_status_t ret = ecall_Internal_LlamaRotaryEmbedding(eid, inv_freq, inv_freq_M, position_ids, position_ids_M, cos, sin);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                    }

    // uint32_t Ext_GenerateCPRNG(unsigned long int eid) {
    //     uint32_t ret;
    //     sgx_status_t status = ecall_Internal_GenerateCPRNG(eid, &ret);
    //     if (status != SGX_SUCCESS) {
    //         print_error_message(status);
    //         throw status;
    //     }
    //     return ret;
    // }

    // uint32_t Ext_GenerateMultKey(unsigned long int eid) {
    //     uint32_t ret;
    //     sgx_status_t status = ecall_Internal_GenerateMultKey(eid, &ret);
    //     if (status != SGX_SUCCESS) {
    //         print_error_message(status);
    //         throw status;
    //     }
    //     return ret;
    // }

    // uint32_t Ext_GenerateAddKey(unsigned long int eid) {
    //     uint32_t ret;
    //     sgx_status_t status = ecall_Internal_GenerateAddKey(eid, &ret);
    //     if (status != SGX_SUCCESS) {
    //         print_error_message(status);
    //         throw status;
    //     }
    //     return ret;
    // }

    void Ext_Reset(unsigned long int eid) {
        sgx_status_t ret = ecall_Internal_Reset(eid);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_ReplicateTensor_Float(unsigned long int eid, int fromm, int* to, int to_len) {
        sgx_status_t ret = ecall_Internal_ReplicateTensor_Float(eid, fromm, to, to_len);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_ReplicateTensor_Int32(unsigned long int eid, int fromm, int* to, int to_len) {
        sgx_status_t ret = ecall_Internal_ReplicateTensor_Int32(eid, fromm, to, to_len);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_ReplicateTensor_Uint32(unsigned long int eid, int fromm, int* to, int to_len) {
        sgx_status_t ret = ecall_Internal_ReplicateTensor_Uint32(eid, fromm, to, to_len);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_ReplicateTensor_Int8(unsigned long int eid, int fromm, int* to, int to_len) {
        sgx_status_t ret = ecall_Internal_ReplicateTensor_Int8(eid, fromm, to, to_len);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_GetCprngTensor(unsigned long int eid, int* out, int shape_len, int* shape) {
        sgx_status_t ret = ecall_Internal_GetCprngTensor(eid, out, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_SetEncKeyAndDecKey(unsigned long int eid, int layer_idx, int* enc_key_pool, int* dec_key,
                                    int type) // type = projection_type 
    {
        sgx_status_t ret = ecall_Internal_SetEncKeyAndDecKey(eid, layer_idx, enc_key_pool, dec_key, type);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }


    void Ext_SetLinearWeightScales(unsigned long int eid, int layer_idx, float* scales, int len,
                                        int type) {
        sgx_status_t ret = ecall_Internal_SetLinearWeightScales(eid, layer_idx, scales, len, type);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                        }

    void Ext_SetRMSNormWeight(unsigned long int eid, int layer_idx, float* weight, float eps,
                                    int type) {
        sgx_status_t ret = ecall_Internal_SetRMSNormWeight(eid, layer_idx, weight, eps, type);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                    }

    void Ext_QuantizeLinearActivation(unsigned long int eid, int layer_idx, int fromm,
                                        int to_len, int* to,
                                        int type) {
        sgx_status_t ret = ecall_Internal_QuantizeLinearActivation(eid, layer_idx, fromm, to_len, to, type);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                        }

    void Ext_EncryptLinearActivation(unsigned long int eid, int layer_idx, int fromm,
                                            int to_len, int* to,
                                            int type) {
        sgx_status_t ret = ecall_Internal_EncryptLinearActivation(eid, layer_idx, fromm, to_len, to, type);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                            }

    void Ext_DecryptLinearActivation(unsigned long int eid, int layer_idx, int fromm,
                                            int to_len, int* to,
                                            int type) {
        sgx_status_t ret = ecall_Internal_DecryptLinearActivation(eid, layer_idx, fromm, to_len, to, type);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                            }

    void Ext_DequantizeLinearActivation(unsigned long int eid, int layer_idx, int fromm,
                                            int to_len, int* to,
                                            int type) {
        sgx_status_t ret = ecall_Internal_DequantizeLinearActivation(eid, layer_idx, fromm, to_len, to, type);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                            }

    void Ext_SetQKVOutputScales(unsigned long int eid, int layer_idx, float q_output_scale,
                                        float k_output_scale, float v_output_scale) {
        sgx_status_t ret = ecall_Internal_SetQKVOutputScales(eid, layer_idx, q_output_scale, k_output_scale, v_output_scale);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                        }

    void Ext_QuantizeAndShiftQ(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_QuantizeAndShiftQ(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_QuantizeAndShiftK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_QuantizeAndShiftK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_SetAttentionMask(unsigned long int eid, float* mask, int M, int N) {
        sgx_status_t ret = ecall_Internal_SetAttentionMask(eid, mask, M, N);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_SetBatchSizeAndTokenLength(unsigned long int eid, int layer_idx, int bsz, int token_length) {
        sgx_status_t ret = ecall_Internal_SetBatchSizeAndTokenLength(eid, layer_idx, bsz, token_length);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_GenerateSecretKey_QK(unsigned long int eid, int layer_idx) {
        sgx_status_t ret = ecall_Internal_GenerateSecretKey_QK(eid, layer_idx);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_GenerateDecryptionKey_QK(unsigned long int eid, int layer_idx, int from_x, int from_y) {
        sgx_status_t ret = ecall_Internal_GenerateDecryptionKey_QK(eid, layer_idx, from_x, from_y);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_GenerateDecAddBuffer_QK(unsigned long int eid, int layer_idx) {
        sgx_status_t ret = ecall_Internal_GenerateDecAddBuffer_QK(eid, layer_idx);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_GenerateDecMultBuffer_QK(unsigned long int eid, int layer_idx) {
        sgx_status_t ret = ecall_Internal_GenerateDecMultBuffer_QK(eid, layer_idx);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_GenerateUnshiftBuffer_QK(unsigned long int eid, int layer_idx) {
        sgx_status_t ret = ecall_Internal_GenerateUnshiftBuffer_QK(eid, layer_idx);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_QuantizeQ_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_QuantizeQ_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_ShiftQ_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_ShiftQ_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_QuantizeK_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_QuantizeK_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_ShiftK_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_ShiftK_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_EncryptX_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_EncryptX_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_EncryptY_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_EncryptY_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_Decrypt_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Decrypt_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_Unshift_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Unshift_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_Dequantize_QK(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Dequantize_QK(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_GenerateSecretKey_PV(unsigned long int eid, int layer_idx) {
        sgx_status_t ret = ecall_Internal_GenerateSecretKey_PV(eid, layer_idx);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_GenerateDecryptionKey_PV(unsigned long int eid, int layer_idx, int from_x, int from_y) {
        sgx_status_t ret = ecall_Internal_GenerateDecryptionKey_PV(eid, layer_idx, from_x, from_y);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_GenerateDecAddBuffer_PV(unsigned long int eid, int layer_idx) {
        sgx_status_t ret = ecall_Internal_GenerateDecAddBuffer_PV(eid, layer_idx);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_GenerateDecMultBuffer_PV(unsigned long int eid, int layer_idx) {
        sgx_status_t ret = ecall_Internal_GenerateDecMultBuffer_PV(eid, layer_idx);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_GenerateUnshiftBuffer_PV(unsigned long int eid, int layer_idx) {
        sgx_status_t ret = ecall_Internal_GenerateUnshiftBuffer_PV(eid, layer_idx);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_QuantizeP_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_QuantizeP_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_ShiftP_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_ShiftP_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_QuantizeV_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_QuantizeV_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_ShiftV_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_ShiftV_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_EncryptX_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_EncryptX_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_EncryptY_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_EncryptY_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_Decrypt_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Decrypt_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_Unshift_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Unshift_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_Dequantize_PV(unsigned long int eid, int layer_idx, int fromm, int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Dequantize_PV(eid, layer_idx, fromm, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Ext_BookKeeperStore_Float(unsigned long int eid, int loc, float* data, int shape_len,
                                        int* shape) {
        sgx_status_t ret = ecall_Internal_BookKeeperStore_Float(eid, loc, data, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                        }
    void Ext_BookKeeperStore_Int32(unsigned long int eid, int loc, int32_t* data, int shape_len,
                                        int* shape) {
        sgx_status_t ret = ecall_Internal_BookKeeperStore_Int32(eid, loc, data, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                        }
    void Ext_BookKeeperStore_Uint32(unsigned long int eid, int loc, uint32_t* data, int shape_len,
                                        int* shape) {
        sgx_status_t ret = ecall_Internal_BookKeeperStore_Uint32(eid, loc, data, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                        }
    void Ext_BookKeeperStore_Int8(unsigned long int eid, int loc, int8_t* data, int shape_len,
                                        int* shape) {
        sgx_status_t ret = ecall_Internal_BookKeeperStore_Int8(eid, loc, data, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                        }

    void Ext_BookKeeperLoad_Float(unsigned long int eid, int loc, float* out, int shape_len,
                                        int* shape) {
        sgx_status_t ret = ecall_Internal_BookKeeperLoad_Float(eid, loc, out, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                        }
    void Ext_BookKeeperLoad_Int32(unsigned long int eid, int loc, int32_t* out, int shape_len,
                                            int* shape) {
        sgx_status_t ret = ecall_Internal_BookKeeperLoad_Int32(eid, loc, out, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                            }
    void Ext_BookKeeperLoad_Uint32(unsigned long int eid, int loc, uint32_t* out, int shape_len,
                                                int* shape) {
        sgx_status_t ret = ecall_Internal_BookKeeperLoad_Uint32(eid, loc, out, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                                }
    void Ext_BookKeeperLoad_Int8(unsigned long int eid, int loc, int8_t* out, int shape_len,
                                            int* shape) {
        sgx_status_t ret = ecall_Internal_BookKeeperLoad_Int8(eid, loc, out, shape_len, shape);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                            }

    void Ext_BookKeeperIsAvailable_Float(unsigned long int eid, int loc, int* ret) {
        sgx_status_t status = ecall_Internal_BookKeeperIsAvailable_Float(eid, loc, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperIsAvailable_Int32(unsigned long int eid, int loc, int* ret) {
        sgx_status_t status = ecall_Internal_BookKeeperIsAvailable_Int32(eid, loc, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperIsAvailable_Uint32(unsigned long int eid, int loc, int* ret) {
        sgx_status_t status = ecall_Internal_BookKeeperIsAvailable_Uint32(eid, loc, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperIsAvailable_Int8(unsigned long int eid, int loc, int* ret) {
        sgx_status_t status = ecall_Internal_BookKeeperIsAvailable_Int8(eid, loc, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_BookKeeperGetShapeLength_Float(unsigned long int eid, int loc, int* ret) {
        sgx_status_t status = ecall_Internal_BookKeeperGetShapeLength_Float(eid, loc, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperGetShapeLength_Int32(unsigned long int eid, int loc, int* ret) {
        sgx_status_t status = ecall_Internal_BookKeeperGetShapeLength_Int32(eid, loc, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperGetShapeLength_Uint32(unsigned long int eid, int loc, int* ret) {
        sgx_status_t status = ecall_Internal_BookKeeperGetShapeLength_Uint32(eid, loc, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperGetShapeLength_Int8(unsigned long int eid, int loc, int* ret) {
        sgx_status_t status = ecall_Internal_BookKeeperGetShapeLength_Int8(eid, loc, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_BookKeeperGetShape_Float(unsigned long int eid, int loc, int* out) {
        sgx_status_t status = ecall_Internal_BookKeeperGetShape_Float(eid, loc, out);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperGetShape_Int32(unsigned long int eid, int loc, int* out) {
        sgx_status_t status = ecall_Internal_BookKeeperGetShape_Int32(eid, loc, out);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperGetShape_Uint32(unsigned long int eid, int loc, int* out) {
        sgx_status_t status = ecall_Internal_BookKeeperGetShape_Uint32(eid, loc, out);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_BookKeeperGetShape_Int8(unsigned long int eid, int loc, int* out) {
        sgx_status_t status = ecall_Internal_BookKeeperGetShape_Int8(eid, loc, out);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_QKKeyIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_QKKeyIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_QKDecKeyIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_QKDecKeyIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_QKDecAddBufferIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_QKDecAddBufferIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_QKDecMultBufferIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_QKDecMultBufferIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_QKShiftedQIsAvailable(unsigned long int eid, int layer_idx, int* ret){ 
        sgx_status_t status = ecall_Internal_QKShiftedQIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_QKShiftedKIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_QKShiftedKIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_QKUnshiftBufferIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_QKUnshiftBufferIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_PVKeyIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_PVKeyIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_PVDecKeyIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_PVDecKeyIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_PVDecAddBufferIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_PVDecAddBufferIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_PVDecMultBufferIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_PVDecMultBufferIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_PVShiftedPIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_PVShiftedPIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }
    void Ext_PVShiftedVIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_PVShiftedVIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_PVUnshiftBufferIsAvailable(unsigned long int eid, int layer_idx, int* ret) {
        sgx_status_t status = ecall_Internal_PVUnshiftBufferIsAvailable(eid, layer_idx, ret);
        if (status != SGX_SUCCESS) {
            print_error_message(status);
            throw status;
        }
    }

    void Ext_Matmul_CPU_QK(unsigned long int eid, int layer_idx, int q_from, int k_from,
                                int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Matmul_CPU_QK(eid, layer_idx, q_from, k_from, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                }
    void Ext_Matmul_CPU_PV(unsigned long int eid, int layer_idx, int p_from, int v_from,  
                                int to_len, int* to) {
        sgx_status_t ret = ecall_Internal_Matmul_CPU_PV(eid, layer_idx, p_from, v_from, to_len, to);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
                                }

    void Ext_BookKeeperLoadWithoutReset_Float(unsigned long int eid, int loc, float* out) {
        sgx_status_t ret = ecall_Internal_BookKeeperLoadWithoutReset_Float(eid, loc, out);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_BookKeeperLoadWithoutReset_Int32(unsigned long int eid, int loc, int32_t* out) {
        sgx_status_t ret = ecall_Internal_BookKeeperLoadWithoutReset_Int32(eid, loc, out);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_BookKeeperLoadWithoutReset_Uint32(unsigned long int eid, int loc, uint32_t* out) {
        sgx_status_t ret = ecall_Internal_BookKeeperLoadWithoutReset_Uint32(eid, loc, out);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
    void Ext_BookKeeperLoadWithoutReset_Int8(unsigned long int eid, int loc, int8_t* out) {
        sgx_status_t ret = ecall_Internal_BookKeeperLoadWithoutReset_Int8(eid, loc, out);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }
}

/* Application entry */
int main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    try {
        sgx_enclave_id_t eid = initialize_enclave();

        std::cout << "Enclave id: " << eid << std::endl;


        printf("Enter a character to destroy enclave ...\n");
        getchar();

        // Destroy the enclave
        sgx_destroy_enclave(eid);

        printf("Info: Enclave Launcher successfully returned.\n");
        printf("Enter a character before exit ...\n");
        getchar();
        return 0;
    }
    catch (int e)
    {
        printf("Info: Enclave Launch failed!.\n");
        printf("Enter a character before exit ...\n");
        getchar();
        return -1;
    }
}