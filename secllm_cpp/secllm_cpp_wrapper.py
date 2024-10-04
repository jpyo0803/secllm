from ctypes import *
import torch
import cupy
from enum import Enum




SECLLM_LIB_PATH = './secllm_cpp/libsecllm.so'

MAX_NUM_LAYERS = 32
MAX_NUM_OPERATIONS = 99
MAX_NUM_INPUTS = 3


def GetBookKeeperLinearIndex(layer_index, operation_index, input_index):
  # NOTE(jpyo0803): debugging purpose
  assert layer_index < MAX_NUM_LAYERS
  assert operation_index < MAX_NUM_OPERATIONS
  assert input_index < MAX_NUM_INPUTS

  return layer_index * 300 + input_index * 100 + operation_index


class SecLLMCppWrapper:
  class ProjectionType(Enum):
    Q = 0
    K = 1
    V = 2
    O = 3
    Gate = 4
    Up = 5
    Down = 6

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

    return cls._instance
  
  def __init__(self, num_hidden_layers, enc_key_pool_size):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True

  @classmethod
  def __GetShape_Float(cls, loc : int):
    dim = c_int(0)
    cls.lib.Ext_BookKeeperGetShapeLength_Float(loc, byref(dim))

    shape_tensor = torch.empty(dim.value, dtype=torch.int32)
    cls.lib.Ext_BookKeeperGetShape_Float(loc, cast(shape_tensor.data_ptr(), POINTER(c_int)))
    shape_list = shape_tensor.tolist()
    return torch.Size(shape_list)

  @classmethod
  def __GetShape_Int32(cls, loc : int):
    dim = c_int(0)
    cls.lib.Ext_BookKeeperGetShapeLength_Int32(loc, byref(dim))

    shape_tensor = torch.empty(dim.value, dtype=torch.int32)
    cls.lib.Ext_BookKeeperGetShape_Int32(loc, cast(shape_tensor.data_ptr(), POINTER(c_int)))
    shape_list = shape_tensor.tolist()
    return torch.Size(shape_list)

  @classmethod
  def __GetShape_Uint32(cls, loc : int):
    dim = c_int(0)
    cls.lib.Ext_BookKeeperGetShapeLength_Uint32(loc, byref(dim))

    shape_tensor = torch.empty(dim.value, dtype=torch.int32)
    cls.lib.Ext_BookKeeperGetShape_Uint32(loc, cast(shape_tensor.data_ptr(), POINTER(c_int)))
    shape_list = shape_tensor.tolist()
    return torch.Size(shape_list)
  
  @classmethod
  def __GetShape_Int8(cls, loc : int):
    dim = c_int(0)
    cls.lib.Ext_BookKeeperGetShapeLength_Int8(loc, byref(dim))

    shape_tensor = torch.empty(dim.value, dtype=torch.int32)
    cls.lib.Ext_BookKeeperGetShape_Int8(loc, cast(shape_tensor.data_ptr(), POINTER(c_int)))
    shape_list = shape_tensor.tolist()
    return torch.Size(shape_list)

  @classmethod
  def PrintTest(cls, a, b):
    cls.lib.Ext_PrintTest(a, b)
  
  @classmethod
  def Softmax(cls, layer_idx : int, src : int, dst: list[int]):
    '''
        NOTE(jpyo0803): Softmax
    '''
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Softmax(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def SwiGLU(cls, layer_idx : int, gate_in : int, up_in : int, dst : list[int]):
    '''
        NOTE(jpyo0803): SwiGLU
        output will be stored in dst
    '''

    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_SwiGLU(layer_idx, gate_in, up_in, len(dst), cast(dst.data_ptr(), POINTER(c_int)))
  
  @classmethod
  def RMSNorm(cls, layer_idx : int, src : int, dst : list[int], type):
    '''
        NOTE(jpyo0803): RMSNorm
        output will be stored in dst
    '''

    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_RMSNorm(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)), type)
  
  @classmethod
  def ElementwiseAdd(cls, layer_idx : int, src1 : int, src2 : int, dst : list[int], type : int):
    '''
        NOTE(jpyo0803): elementwise add
        output will be stored in dst
    '''
    
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ElementWiseAdd(layer_idx, src1, src2, len(dst), cast(dst.data_ptr(), POINTER(c_int)), type)


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
  def BookKeeperIsAvailable_Float(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    ret = c_bool(-1)
    cls.lib.Ext_BookKeeperIsAvailable_Float(loc, byref(ret))
    # convert c_bool to python bool
    return ret.value

  @classmethod
  def BookKeeperIsAvailable_Int32(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    ret = c_bool(-1)
    cls.lib.Ext_BookKeeperIsAvailable_Int32(loc, byref(ret))
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
  def BookKeeperIsAvailable_Int8(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    ret = c_bool(-1)
    cls.lib.Ext_BookKeeperIsAvailable_Int8(loc, byref(ret))
    # convert c_bool to python bool
    return ret.value
  
  @classmethod
  def BookKeeperStore_Float(cls, layer_index, operation_index, input_index, data):
    assert data.is_contiguous()
    assert data.dtype == torch.float32
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    # Convert shape to list
    shape_list = torch.tensor(data.shape, dtype=torch.int32)
    cls.lib.Ext_BookKeeperStore_Float(loc, cast(data.data_ptr(), POINTER(c_float)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def BookKeeperStore_Int32(cls, layer_index, operation_index, input_index, data):
    assert data.is_contiguous()
    assert data.dtype == torch.int32
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    # Convert shape to list
    shape_list = torch.tensor(data.shape, dtype=torch.int32)
    cls.lib.Ext_BookKeeperStore_Int32(loc, cast(data.data_ptr(), POINTER(c_int)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def BookKeeperStore_Uint32(cls, layer_index, operation_index, input_index, data, new_shape = None):
    assert data.is_contiguous()
    assert data.dtype == torch.uint32
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    # Convert shape to list
    shape_list = torch.tensor(data.shape, dtype=torch.int32)
    cls.lib.Ext_BookKeeperStore_Uint32(loc, cast(data.data_ptr(), POINTER(c_uint32)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def BookKeeperStore_Int8(cls, layer_index, operation_index, input_index, data):
    assert data.is_contiguous()
    assert data.dtype == torch.int8
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    # Convert shape to list
    shape_list = torch.tensor(data.shape, dtype=torch.int32)
    cls.lib.Ext_BookKeeperStore_Int8(loc, cast(data.data_ptr(), POINTER(c_int8)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def BookKeeperLoad_Float(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.__GetShape_Float(loc)

    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.float32)

    cls.lib.Ext_BookKeeperLoad_Float(loc, cast(out.data_ptr(), POINTER(c_float)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out

  @classmethod
  def BookKeeperLoad_Int32(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.__GetShape_Int32(loc)

    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.int32)

    cls.lib.Ext_BookKeeperLoad_Int32(loc, cast(out.data_ptr(), POINTER(c_int)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out

  @classmethod
  def BookKeeperLoad_Uint32(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.__GetShape_Uint32(loc)

    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.uint32)

    cls.lib.Ext_BookKeeperLoad_Uint32(loc, cast(out.data_ptr(), POINTER(c_uint32)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out
  
  @classmethod
  def BookKeeperLoad_Int8(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.__GetShape_Int8(loc)

    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.int8)

    cls.lib.Ext_BookKeeperLoad_Int8(loc, cast(out.data_ptr(), POINTER(c_int8)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out

  @classmethod
  def BroadcastTensor_Float(cls, src : int, dst : list[int]):
    # be careful src is in int64
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ReplicateTensor_Float(src, cast(dst.data_ptr(), POINTER(c_int)), len(dst))

  @classmethod
  def GetCprngTensor(cls, shape):
    out = torch.empty(shape, dtype=torch.int32)
    shape_list = torch.tensor(shape, dtype=torch.int32)
    cls.lib.Ext_GetCprngTensor(cast(out.data_ptr(), POINTER(c_int)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))
    return out
  
  @classmethod
  def SetEncKeyAndDecKey(cls, layer_idx : int, enc_key_pool, dec_key, type : ProjectionType):
    '''
        NOTE(jpyo0803): Set enc_key_pool and precomputed_dec_key
    '''
    assert enc_key_pool.dtype == torch.int32
    assert dec_key.dtype == torch.int32
    cls.lib.Ext_SetEncKeyAndDecKey(layer_idx, cast(enc_key_pool.data_ptr(), POINTER(c_int)), cast(dec_key.data_ptr(), POINTER(c_int)), type.value)

  @classmethod
  def SetLinearWeightScales(cls, layer_idx : int, weight_scales, type : ProjectionType):
    '''
        NOTE(jpyo0803): Set weight scales
    '''
    assert weight_scales.dtype == torch.float32

    cls.lib.Ext_SetLinearWeightScales(layer_idx, cast(weight_scales.data_ptr(), POINTER(c_float)), weight_scales.shape[0], type.value)
  
  @classmethod
  def SetRMSNormWeight(cls, layer_idx : int, weight, eps, type : int):
    '''
        NOTE(jpyo0803): Set RMSNorm weight
    '''
    weight = weight.to(torch.float32)
    assert weight.dtype == torch.float32
    cls.lib.Ext_SetRMSNormWeight(layer_idx, cast(weight.data_ptr(), POINTER(c_float)), c_float(eps), type)

  @classmethod
  def QuantizeLinearActivation(cls, layer_idx: int , src : int, dst : list[int], type : ProjectionType):
    '''
        NOTE(jpyo0803): Quantize Activation
    '''
    dst = torch.tensor(dst, dtype=torch.int32)

    cls.lib.Ext_QuantizeLinearActivation(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)), type.value)

  @classmethod
  def EncryptLinearActivation(cls, layer_idx: int , src : int, dst : list[int], type : ProjectionType):
    '''
        NOTE(jpyo0803): Encrypt and Project Activation
    '''
    dst = torch.tensor(dst, dtype=torch.int32)

    cls.lib.Ext_EncryptLinearActivation(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)), type.value)

  @classmethod
  def DecryptLinearActivation(cls, layer_idx, src : int, dst : list[int], type : ProjectionType):
    '''
        NOTE(jpyo0803): Decrypt Activation
    '''
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_DecryptLinearActivation(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)), type.value)

  @classmethod
  def DequantizeLinearActivation(cls, layer_idx, src : int, dst : list[int], type : ProjectionType):
    '''
        NOTE(jpyo0803): Dequantize Activation
    '''
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_DequantizeLinearActivation(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)), type.value)

  @classmethod
  def SetQKVOutputScales(cls, layer_idx, q_output_scale, k_output_scale, v_output_scale):
    '''
        NOTE(jpyo0803): Set QKV output scales
    '''

    cls.lib.Ext_SetQKVOutputScales(layer_idx, c_float(q_output_scale), c_float(k_output_scale), c_float(v_output_scale))

  @classmethod
  def QuantizeQ_QK(cls, layer_idx : int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Quantize Q
    '''
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_QuantizeQ_QK(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def ShiftQ_QK(cls, layer_idx : int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Shift Q
    '''
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ShiftQ_QK(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def QuantizeK_QK(cls, layer_idx : int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Quantize K
    '''
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_QuantizeK_QK(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def ShiftK_QK(cls, layer_idx : int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Shift K
    '''
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ShiftK_QK(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def QuantizeP_PV(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Quantize P
    '''
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_QuantizeP_PV(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def ShiftP_PV(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Shift P
    '''
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ShiftP_PV(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def QuantizeV_PV(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Quantize V
    '''
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_QuantizeV_PV(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def ShiftV_PV(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Shift V
    '''
    dst_list = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ShiftV_PV(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  @classmethod
  def Unshift_QK(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Unshift QK
    '''
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Unshift_QK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def Dequantize_QK(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Dequantize QK
    '''
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Dequantize_QK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def Unshift_PV(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Unshift PV
    '''
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Unshift_PV(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def Dequantize_PV(cls, layer_idx: int, src : int, dst : list[int]):
    '''
        NOTE(jpyo0803): Dequantize PV
    '''
    
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Dequantize_PV(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

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
    cls.lib.Ext_GenerateDecryptionKey_QK(layer_idx, src_x, src_y)

  @classmethod
  def EncryptX_QK(cls, layer_idx, src, dst : list[int]):
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_EncryptX_QK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def EncryptY_QK(cls, layer_idx, src, dst : list[int]):
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_EncryptY_QK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def Decrypt_QK(cls, layer_idx, src, dst : list[int]):
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_Decrypt_QK(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def GenerateSecretKey_PV(cls, layer_idx):
    cls.lib.Ext_GenerateSecretKey_PV(layer_idx)

  @classmethod
  def GenerateDecryptionKey_PV(cls, layer_idx, src_x : int, src_y: int):
    cls.lib.Ext_GenerateDecryptionKey_PV(layer_idx, src_x, src_y)

  @classmethod
  def EncryptX_PV(cls, layer_idx, src, dst : list[int]):
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_EncryptX_PV(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def EncryptY_PV(cls, layer_idx, src, dst : list[int]):
    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_EncryptY_PV(layer_idx, src, len(dst), cast(dst.data_ptr(), POINTER(c_int)))

  @classmethod
  def Decrypt_PV(cls, layer_idx, src, dst : list[int]):
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

  @classmethod
  def Close(cls, output_file_name : str):
    # pass string to C++
    cls.lib.Ext_Close(c_char_p(output_file_name.encode('utf-8'))) 

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

