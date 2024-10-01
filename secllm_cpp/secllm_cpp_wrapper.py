from ctypes import *
import torch
import cupy

SECLLM_LIB_PATH = './secllm_cpp/libsecllm.so'

MAX_NUM_LAYERS = 32
MAX_NUM_OPERATIONS = 92
MAX_NUM_INPUTS = 3


def GetBookKeeperLinearIndex(layer_index, operation_index, input_index):
  # NOTE(jpyo0803): debugging purpose
  assert layer_index < MAX_NUM_LAYERS
  assert operation_index < MAX_NUM_OPERATIONS
  assert input_index < MAX_NUM_INPUTS

  return layer_index * 300 + input_index * 100 + operation_index


class SecLLMCppWrapper:
  def __new__(cls, config, enc_key_pool_size):
    if not hasattr(cls, 'instance'):
      cls._instance = super().__new__(cls)
      cls.lib = cdll.LoadLibrary(SECLLM_LIB_PATH)

      '''
         Pass
         hidden_size,
         intermediate_size,
         max_position_embeddings,
         num_attention_heads,
         num_hidden_layers,
         num_key_value_heads
      '''

      cls.lib.Ext_CreateSecLLM(config.hidden_size, config.intermediate_size, config.max_position_embeddings, config.num_attention_heads, config.num_hidden_layers, config.num_key_value_heads, enc_key_pool_size)

      cls.shape_bookkeeper = [None for _ in range(MAX_NUM_LAYERS * 300)]
      cls.shape_bookkeeper_uint32 = [None for _ in range(MAX_NUM_LAYERS * 300)]

    return cls._instance
  
  def __init__(self, num_hidden_layers, enc_key_pool_size):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True

  @classmethod
  def TransportShape_Float_to_Float(cls, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Transport shape from src to dst
    '''
    src_shape = cls.shape_bookkeeper[src]
    cls.shape_bookkeeper[src] = None
    assert src_shape is not None
    for e in dst:
      cls.shape_bookkeeper[e] = src_shape

  @classmethod
  def TransportShape_Uint32_to_Uint32(cls, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Transport shape from src to dst
    '''
    src_shape = cls.shape_bookkeeper_uint32[src]
    cls.shape_bookkeeper_uint32[src] = None
    assert src_shape is not None
    for e in dst:
      cls.shape_bookkeeper_uint32[e] = src_shape

  @classmethod
  def TransportShape_Float_to_Uint32(cls, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Transport shape from src to dst
    '''
    src_shape = cls.shape_bookkeeper[src]
    cls.shape_bookkeeper[src] = None
    assert src_shape is not None
    for e in dst:
      cls.shape_bookkeeper_uint32[e] = src_shape
  
  @classmethod
  def TransportShape_Uint32_to_Float(cls, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Transport shape from src to dst
    '''
    src_shape = cls.shape_bookkeeper_uint32[src]
    cls.shape_bookkeeper_uint32[src] = None
    assert src_shape is not None
    for e in dst:
      cls.shape_bookkeeper[e] = src_shape

  @classmethod
  def PrintTest(cls, a, b):
    cls.lib.Ext_PrintTest(a, b)
  
  @classmethod
  def Softmax(cls, src : int, dst: list[int]):
    '''
        NOTE(jpyo0803): Softmax
    '''
    cls.TransportShape_Float_to_Float(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Softmax(src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def SwiGLU(cls, gate_in : int, up_in : int, dst : list[int]):
    '''
        NOTE(jpyo0803): SwiGLU
        output will be stored in dst
    '''
    assert cls.shape_bookkeeper[gate_in] is not None
    assert cls.shape_bookkeeper[up_in] is not None
    assert cls.shape_bookkeeper[gate_in] == cls.shape_bookkeeper[up_in]
    cls.TransportShape_Float_to_Float(gate_in, dst)
    cls.shape_bookkeeper[up_in] = None

    dst = torch.tensor(dst, dtype=torch.int32)

    cls.lib.Ext_SwiGLU(gate_in, up_in, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  
  @classmethod
  def RMSNorm(cls, src : int, dst : list[int], weight, eps):
    '''
        NOTE(jpyo0803): RMSNorm
        output will be stored in dst
    '''
    assert cls.shape_bookkeeper[src] is not None
    cls.TransportShape_Float_to_Float(src, dst)
    weight = weight.to(torch.float32) # should happen only once if necessary
    
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_RMSNorm(src, len(dst), cast(dst.data_ptr(), POINTER(c_int)), cast(weight.data_ptr(), POINTER(c_float)), c_float(eps))
  
  @classmethod
  def ElementwiseAdd(cls, src1 : int, src2 : int, dst : list[int]):
    '''
        NOTE(jpyo0803): elementwise add
        output will be stored in dst
    '''
    assert cls.shape_bookkeeper[src1] is not None
    assert cls.shape_bookkeeper[src2] is not None
    assert cls.shape_bookkeeper[src1] == cls.shape_bookkeeper[src2]

    cls.TransportShape_Float_to_Float(src1, dst)
    cls.shape_bookkeeper[src2] = None

    dst = torch.tensor(dst, dtype=torch.int32)

    cls.lib.Ext_ElementWiseAdd(src1, src2, len(dst), cast(dst.data_ptr(), POINTER(c_int)))


  @classmethod
  def ApplyRotaryPosEmb(cls, q, k, cos, sin):
    '''
        NOTE(jpyo0803): in-place apply rotary position embedding
        output will be stored in q
    '''
    assert q.dim() == 4
    assert k.dim() == 4
    q = q.contiguous()
    k = k.contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert cos.is_contiguous()
    assert sin.is_contiguous()

    dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    B, Q_M, N, K = q.shape
    _, K_M, _, _ = k.shape
    cls.lib.Ext_ApplyRotaryPosEmb(cast(q.data_ptr(), POINTER(c_float)), cast(k.data_ptr(), POINTER(c_float)), cast(cos.data_ptr(), POINTER(c_float)), cast(sin.data_ptr(), POINTER(c_float)), B, Q_M, K_M, N, K)
    q_embed = q.to(dtype)
    k_embed = k.to(dtype)
    return q_embed, k_embed

  @classmethod
  def LlamaRotaryEmbedding(cls, inv_freq, position_ids, input_dtype):
    '''
        NOTE(jpyo0803): Llama rotary embedding
    '''
    assert inv_freq.dim() == 1
    assert position_ids.dim() == 2
    assert inv_freq.is_contiguous()
    assert position_ids.is_contiguous()

    inv_freq = inv_freq.to(torch.float32)
    position_ids = position_ids.to(torch.float32)

    cos = torch.zeros(1, position_ids.shape[1], inv_freq.shape[0] * 2, dtype=torch.float32)
    sin = torch.zeros(1, position_ids.shape[1], inv_freq.shape[0] * 2, dtype=torch.float32)

    cls.lib.Ext_LlamaRotaryEmbedding(cast(inv_freq.data_ptr(), POINTER(c_float)), inv_freq.shape[0], cast(position_ids.data_ptr(), POINTER(c_float)), position_ids.shape[1], cast(cos.data_ptr(), POINTER(c_float)), cast(sin.data_ptr(), POINTER(c_float)))
    cos = cos.to(input_dtype)
    sin = sin.to(input_dtype)
    return cos, sin

  @classmethod
  def BookKeeperStore(cls, layer_index, operation_index, input_index, data):
    assert data.is_contiguous()
    assert data.dtype == torch.float32
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    # Convert shape to list
    shape_list = torch.tensor(data.shape, dtype=torch.int32)

    cls.shape_bookkeeper[loc] = data.shape

    cls.lib.Ext_BookKeeperStore(loc, cast(data.data_ptr(), POINTER(c_float)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def BookKeeperIsAvailable(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    ret = c_bool(-1)
    cls.lib.Ext_BookKeeperIsAvailable(loc, byref(ret))
    # convert c_bool to python bool
    return ret.value

  @classmethod
  def BookKeeperIsAvailable_Uint32(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    ret = c_bool(-1)
    cls.lib.Ext_BookKeeperIsAvailable_Uint32(loc, byref(ret))
    # convert c_bool to python bool
    return ret.value

  @classmethod
  def BookKeeperStore_Uint32(cls, layer_index, operation_index, input_index, data, new_shape = None):
    assert data.is_contiguous()
    assert data.dtype == torch.uint32
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    # Convert shape to list
    shape_list = torch.tensor(data.shape, dtype=torch.int32)

    cls.shape_bookkeeper_uint32[loc] = data.shape if new_shape is None else new_shape

    cls.lib.Ext_BookKeeperStore_Uint32(loc, cast(data.data_ptr(), POINTER(c_uint32)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def BookKeeperLoad(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.shape_bookkeeper[loc]
    assert shape is not None
    cls.shape_bookkeeper[loc] = None
    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.float32)

    cls.lib.Ext_BookKeeperLoad(loc, cast(out.data_ptr(), POINTER(c_float)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out

  @classmethod
  def BookKeeperLoad_Uint32(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.shape_bookkeeper_uint32[loc]
    assert shape is not None
    cls.shape_bookkeeper_uint32[loc] = None
    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.uint32)

    cls.lib.Ext_BookKeeperLoad_Uint32(loc, cast(out.data_ptr(), POINTER(c_int)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out

  @classmethod
  def BookKeeperReshape_Uint32(cls, index, new_shape):
    assert cls.shape_bookkeeper_uint32[index] is not None
    cls.shape_bookkeeper_uint32[index] = new_shape

  @classmethod
  def ReplicateTensor(cls, src : int, dst : list[int]):
    # be careful src is in int64
    cls.TransportShape_Float_to_Float(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ReplicateTensor(src, cast(dst.data_ptr(), POINTER(c_int)), len(dst))

  @classmethod
  def ReplicateTensor_Uint32(cls, src : int, dst : list[int]):
    # be careful src is in int64
    cls.TransportShape_Uint32_to_Uint32(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ReplicateTensor_Uint32(src, cast(dst.data_ptr(), POINTER(c_int)), len(dst))

  @classmethod
  def GetCprngTensor(cls, shape):
    out = torch.empty(shape, dtype=torch.int32)
    shape_list = torch.tensor(shape, dtype=torch.int32)
    cls.lib.Ext_GetCprngTensor(cast(out.data_ptr(), POINTER(c_int)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))
    return out
  
  @classmethod
  def SetEncKeyAndDecKey(cls, layer_idx, enc_key_pool, dec_key, type):
    '''
        NOTE(jpyo0803): Set enc_key_pool and precomputed_dec_key
    '''
    assert type <= 6
    cls.lib.Ext_SetEncKeyAndDecKey(layer_idx, cast(enc_key_pool.data_ptr(), POINTER(c_int)), cast(dec_key.data_ptr(), POINTER(c_int)), type)

  @classmethod
  def SetLinearWeightScales(cls, layer_idx, weight_scales, type):
    '''
        NOTE(jpyo0803): Set weight scales
    '''
    assert type <= 6
    assert weight_scales.dtype == torch.float32

    cls.lib.Ext_SetLinearWeightScales(layer_idx, cast(weight_scales.data_ptr(), POINTER(c_float)), weight_scales.shape[0], type)

  @classmethod
  def EncryptLinearActivation(cls, layer_idx, src, type):
    '''
        NOTE(jpyo0803): Encrypt and Project Activation
    '''
    src_shape = cls.shape_bookkeeper[src]
    assert src_shape is not None
    cls.shape_bookkeeper[src] = None

    out = torch.empty(src_shape, dtype=torch.int32)

    cls.lib.Ext_EncryptLinearActivation(layer_idx, cast(out.data_ptr(), POINTER(c_int)), src, type)
    return out

  @classmethod
  def DecryptLinearActivation(cls, layer_idx, dst : list[int], enc_tensor, type):
    '''
        NOTE(jpyo0803): Decrypt Activation
    '''
    assert type <= 6
    for e in dst:
      assert cls.shape_bookkeeper[e] is None
      cls.shape_bookkeeper[e] = enc_tensor.shape
    enc_tensor_shape_list = torch.tensor(enc_tensor.shape, dtype=torch.int32)

    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_DecryptLinearActivation(layer_idx, len(dst), cast(dst.data_ptr(), POINTER(c_int)), cast(enc_tensor.data_ptr(), POINTER(c_int)), len(enc_tensor_shape_list), cast(enc_tensor_shape_list.data_ptr(), POINTER(c_int)), type)

  @classmethod
  def SetQKVOutputScales(cls, layer_idx, q_output_scale, k_output_scale, v_output_scale):
    '''
        NOTE(jpyo0803): Set QKV output scales
    '''

    cls.lib.Ext_SetQKVOutputScales(layer_idx, c_float(q_output_scale), c_float(k_output_scale), c_float(v_output_scale))

  @classmethod
  def QuantizeAndShiftQ(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Quantize and Shift Q
    '''
    cls.TransportShape_Float_to_Uint32(src, dst)
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_QuantizeAndShiftQ(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def QuantizeAndShiftK(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Quantize and Shift K
    '''
    cls.TransportShape_Float_to_Uint32(src, dst)
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_QuantizeAndShiftK(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def QuantizeAndShiftP(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Quantize and Shift P
    '''
    cls.TransportShape_Float_to_Uint32(src, dst)
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_QuantizeAndShiftP(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def QuantizeAndShiftV(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Quantize and Shift V
    '''
    cls.TransportShape_Float_to_Uint32(src, dst)
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_QuantizeAndShiftV(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def UnshiftAndDequantizeQK(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Unshift and Dequantize QK
    '''
    cls.TransportShape_Uint32_to_Float(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_UnshiftAndDequantizeQK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def UnshiftAndDequantizePV(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Unshift and Dequantize PV
    '''
    cls.TransportShape_Uint32_to_Float(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_UnshiftAndDequantizePV(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def SetAttentionMask(cls, attn_mask):
    '''
        NOTE(jpyo0803): Set Attention Mask
    '''

    assert attn_mask.is_contiguous()
    attn_mask = attn_mask.to(torch.float32)
    cls.lib.Ext_SetAttentionMask(cast(attn_mask.data_ptr(), POINTER(c_float)), attn_mask.shape[-2], attn_mask.shape[-1])


  @classmethod
  def SetBatchSizeAndTokenLength(cls, layer_idx, bsz, token_length):
    cls.lib.Ext_SetBatchSizeAndTokenLength(layer_idx, bsz, token_length)

  @classmethod
  def GenerateSecretKey_QK(cls, layer_idx):
    cls.lib.Ext_GenerateSecretKey_QK(layer_idx)

  @classmethod
  def GenerateDecryptionKey_QK(cls, layer_idx, src_x : int, src_y: int):
    assert cls.shape_bookkeeper_uint32[src_x] is not None
    assert cls.shape_bookkeeper_uint32[src_y] is not None
    cls.shape_bookkeeper_uint32[src_x] = None
    cls.shape_bookkeeper_uint32[src_y] = None

    cls.lib.Ext_GenerateDecryptionKey_QK(layer_idx, src_x, src_y)

  @classmethod
  def EncryptX_QK(cls, layer_idx, src, dst : list[int]):
    cls.TransportShape_Uint32_to_Uint32(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_EncryptX_QK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def EncryptY_QK(cls, layer_idx, src, dst : list[int]):
    cls.TransportShape_Uint32_to_Uint32(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_EncryptY_QK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def Decrypt_QK(cls, layer_idx, src, dst : list[int]):
    cls.TransportShape_Uint32_to_Uint32(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Decrypt_QK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def GenerateSecretKey_PV(cls, layer_idx):
    cls.lib.Ext_GenerateSecretKey_PV(layer_idx)

  @classmethod
  def GenerateDecryptionKey_PV(cls, layer_idx, src_x : int, src_y: int):
    assert cls.shape_bookkeeper_uint32[src_x] is not None
    assert cls.shape_bookkeeper_uint32[src_y] is not None
    cls.shape_bookkeeper_uint32[src_x] = None
    cls.shape_bookkeeper_uint32[src_y] = None
    cls.lib.Ext_GenerateDecryptionKey_PV(layer_idx, src_x, src_y)

  @classmethod
  def EncryptX_PV(cls, layer_idx, src, dst : list[int]):
    cls.TransportShape_Uint32_to_Uint32(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_EncryptX_PV(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def EncryptY_PV(cls, layer_idx, src, dst : list[int]):
    cls.TransportShape_Uint32_to_Uint32(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_EncryptY_PV(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def Decrypt_PV(cls, layer_idx, src, dst : list[int]):
    cls.TransportShape_Uint32_to_Uint32(src, dst)
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Decrypt_PV(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def QKKeyIsAvailable(cls, layer_idx):
    ret = c_bool(-1)
    cls.lib.Ext_QKKeyIsAvailable(layer_idx, byref(ret))
    return ret.value

  @classmethod
  def PVKeyIsAvailable(cls, layer_idx):
    ret = c_bool(-1)
    cls.lib.Ext_PVKeyIsAvailable(layer_idx, byref(ret))
    return ret.value
  
  @classmethod
  def QKDecKeyIsAvailable(cls, layer_idx):
    ret = c_bool(-1)
    cls.lib.Ext_QKDecKeyIsAvailable(layer_idx, byref(ret))
    return ret.value
  
  @classmethod
  def PVDecKeyIsAvailable(cls, layer_idx):
    ret = c_bool(-1)
    cls.lib.Ext_PVDecKeyIsAvailable(layer_idx, byref(ret))
    return ret.value

  @classmethod
  def Reset(cls):
    print("Reset internal states")
    cls.lib.Ext_Reset()

if __name__ == '__main__':
    secllm = SecLLM(32)

    t = torch.randn(2, 3, 4)
    print("t: ", t)
    t_shape = t.shape
    print("t shape : ", t_shape)
    print("t sum : ", t.sum())

    secllm.BookKeeperStore(25, 75, 2, t)

    t2 = secllm.BookKeeperLoad(25, 75, 2)

    print("t2: ", t2)
    print("t2 shape : ", t2.shape)
    print("t2 sum : ", t2.sum())

