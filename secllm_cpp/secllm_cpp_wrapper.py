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
  def PrintTest(cls, a, b):
    cls.lib.Ext_PrintTest(a, b)


  @classmethod
  def Softmax_InPlace(cls, x):
    '''
        NOTE(jpyo0803): in-place Softmax
    '''
    assert x.dim() == 4
    assert x.is_contiguous()
    dtype = x.dtype
    x = x.to(torch.float32)
    B, M, N, K = x.shape
    cls.lib.Ext_Softmax_InPlace(cast(x.data_ptr(), POINTER(c_float)), B, M, N, K)
    x = x.to(dtype)
    return x
  
  def Softmax(cls, src : int, dst: int):
    '''
        NOTE(jpyo0803): Softmax
    '''
    src_shape = cls.shape_bookkeeper[src]
    assert src_shape is not None
    cls.shape_bookkeeper[dst] = src_shape
    cls.shape_bookkeeper[src] = None

    cls.lib.Ext_Softmax(src, dst)

  def SwiGLU_InPlace(cls, gate_in, up_in):
    '''
        NOTE(jpyo0803): in-place SwiGLU_InPlace
        output will be stored in gate_in
    '''

    assert gate_in.dim() == 3
    assert gate_in.size() == up_in.size()
    assert gate_in.is_contiguous()
    assert up_in.is_contiguous()

    dtype = gate_in.dtype
    gate_in = gate_in.to(torch.float32)
    up_in = up_in.to(torch.float32)
    B, M, N = gate_in.shape
    cls.lib.Ext_SwiGLU_InPlace(cast(gate_in.data_ptr(), POINTER(c_float)), cast(up_in.data_ptr(), POINTER(c_float)), B, M, N)
    gate_in = gate_in.to(dtype)
    return gate_in

  def SwiGLU(cls, gate_in : int, up_in : int, dst : int):
    '''
        NOTE(jpyo0803): SwiGLU
        output will be stored in dst
    '''

    gate_in_shape = cls.shape_bookkeeper[gate_in]
    up_in_shape = cls.shape_bookkeeper[up_in]
    assert gate_in_shape is not None
    assert up_in_shape is not None
    assert gate_in_shape == up_in_shape
    cls.shape_bookkeeper[dst] = gate_in_shape
    cls.shape_bookkeeper[gate_in] = None
    cls.shape_bookkeeper[up_in] = None

    cls.lib.Ext_SwiGLU(gate_in, up_in, dst)

  def RMSNorm_InPlace(cls, x, weight, eps):
    '''
        NOTE(jpyo0803): in-place RMSNorm
        output will be stored in x
    '''

    assert x.dim() == 3
    assert x.size()[2] == weight.size()[0] # last dimension must match
    assert x.is_contiguous()
    # assert weight.is_contiguous()

    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32) # should happen only once if necessary
    B, M, N = x.shape
    cls.lib.Ext_RMSNorm_InPlace(cast(x.data_ptr(), POINTER(c_float)), cast(weight.data_ptr(), POINTER(c_float)), B, M, N, c_float(eps))
    x = x.to(dtype)
    return x
  
  def RMSNorm(cls, src : int, dst: int, weight, eps):
    '''
        NOTE(jpyo0803): RMSNorm
        output will be stored in dst
    '''

    src_shape = cls.shape_bookkeeper[src]
    assert src_shape is not None
    cls.shape_bookkeeper[dst] = src_shape
    cls.shape_bookkeeper[src] = None

    weight = weight.to(torch.float32) # should happen only once if necessary
    cls.lib.Ext_RMSNorm(src, dst, cast(weight.data_ptr(), POINTER(c_float)), c_float(eps))

  def ElementwiseAdd_InPlace(cls, x, y):
    '''
        NOTE(jpyo0803): in-place elementwise add
        output will be stored in x
    '''
    assert x.dim() == 3
    assert x.size() == y.size()
    assert x.is_contiguous()
    assert y.is_contiguous()

    dtype = x.dtype
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    B, M, N = x.shape
    cls.lib.Ext_ElementWiseAdd_InPlace(cast(x.data_ptr(), POINTER(c_float)), cast(y.data_ptr(), POINTER(c_float)), B, M, N)
    x = x.to(dtype)
    return x
  
  def ElementwiseAdd(cls, src1 : int, src2 : int, dst : int):
    '''
        NOTE(jpyo0803): elementwise add
        output will be stored in dst
    '''
    src1_shape = cls.shape_bookkeeper[src1]
    src2_shape = cls.shape_bookkeeper[src2]
    assert src1_shape is not None
    assert src2_shape is not None
    assert src1_shape == src2_shape
    cls.shape_bookkeeper[dst] = src1_shape
    cls.shape_bookkeeper[src1] = None
    cls.shape_bookkeeper[src2] = None

    cls.lib.Ext_ElementWiseAdd(src1, src2, dst)

  def ElementwiseSubtract(cls, src1 : int, src2 : int, dst : int):
    '''
        NOTE(jpyo0803): elementwise subtract
        output will be stored in dst
    '''
    src1_shape = cls.shape_bookkeeper[src1]
    src2_shape = cls.shape_bookkeeper[src2]
    assert src1_shape is not None
    assert src2_shape is not None
    assert src1_shape == src2_shape
    cls.shape_bookkeeper[dst] = src1_shape
    cls.shape_bookkeeper[src1] = None
    cls.shape_bookkeeper[src2] = None

    cls.lib.Ext_ElementWiseSubtract(src1, src2, dst)

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



  def BookKeeperStore(cls, layer_index, operation_index, input_index, data):
    assert data.is_contiguous()
    assert data.dtype == torch.float32
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    # Convert shape to list
    shape_list = torch.tensor(data.shape, dtype=torch.int32)

    cls.shape_bookkeeper[loc] = data.shape

    cls.lib.Ext_BookKeeperStore(loc, cast(data.data_ptr(), POINTER(c_float)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))

  def BookKeeperStore_Uint32(cls, layer_index, operation_index, input_index, data, new_shape = None):
    assert data.is_contiguous()
    assert data.dtype == torch.uint32
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    # Convert shape to list
    shape_list = torch.tensor(data.shape, dtype=torch.int32)

    cls.shape_bookkeeper_uint32[loc] = data.shape if new_shape is None else new_shape

    cls.lib.Ext_BookKeeperStore_Uint32(loc, cast(data.data_ptr(), POINTER(c_uint32)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))

  def BookKeeperLoad(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.shape_bookkeeper[loc]
    assert shape is not None
    cls.shape_bookkeeper[loc] = None
    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.float32)

    cls.lib.Ext_BookKeeperLoad(loc, cast(out.data_ptr(), POINTER(c_float)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out

  def BookKeeperLoad_Uint32(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.shape_bookkeeper_uint32[loc]
    assert shape is not None
    cls.shape_bookkeeper_uint32[loc] = None
    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.uint32)

    cls.lib.Ext_BookKeeperLoad_Uint32(loc, cast(out.data_ptr(), POINTER(c_int)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out

  def BookKeeperReshape_Uint32(cls, index, new_shape):
    assert cls.shape_bookkeeper_uint32[index] is not None
    cls.shape_bookkeeper_uint32[index] = new_shape

  def ReplicateTensor(cls, src : int, dst : list):
    # be careful src is in int64
    src_shape = cls.shape_bookkeeper[src]
    cls.shape_bookkeeper[src] = None
    for e in dst:
      cls.shape_bookkeeper[e] = src_shape

    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ReplicateTensor(src, cast(dst.data_ptr(), POINTER(c_int)), len(dst))

  def ReplicateTensor_Uint32(cls, src : int, dst : list):
    # be careful src is in int64
    src_shape = cls.shape_bookkeeper_uint32[src]
    cls.shape_bookkeeper_uint32[src] = None
    for e in dst:
      cls.shape_bookkeeper_uint32[e] = src_shape

    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ReplicateTensor_Uint32(src, cast(dst.data_ptr(), POINTER(c_int)), len(dst))

  def GetCprngTensor(cls, shape):
    out = torch.empty(shape, dtype=torch.int32)
    shape_list = torch.tensor(shape, dtype=torch.int32)
    cls.lib.Ext_GetCprngTensor(cast(out.data_ptr(), POINTER(c_int)), len(shape_list), cast(shape_list.data_ptr(), POINTER(c_int)))
    return out
  
  def SetEncKeyAndDecKey(cls, layer_idx, enc_key_pool, dec_key, type):
    '''
        NOTE(jpyo0803): Set enc_key_pool and precomputed_dec_key
    '''
    assert type <= 6
  
    cls.lib.Ext_SetEncKeyAndDecKey(layer_idx, cast(enc_key_pool.data_ptr(), POINTER(c_int)), cast(dec_key.data_ptr(), POINTER(c_int)), type)

  def SetLinearWeightScales(cls, layer_idx, weight_scales, type):
    '''
        NOTE(jpyo0803): Set weight scales
    '''
    assert type <= 6
    assert weight_scales.dtype == torch.float32

    cls.lib.Ext_SetLinearWeightScales(layer_idx, cast(weight_scales.data_ptr(), POINTER(c_float)), weight_scales.shape[0], type)

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

  def DecryptLinearActivation(cls, layer_idx, dst, enc_tensor, type):
    '''
        NOTE(jpyo0803): Decrypt Activation
    '''
    assert type <= 6
    cls.shape_bookkeeper[dst] = enc_tensor.shape
    enc_tensor_shape_list = torch.tensor(enc_tensor.shape, dtype=torch.int32)

    cls.lib.Ext_DecryptLinearActivation(layer_idx, dst, cast(enc_tensor.data_ptr(), POINTER(c_int)), len(enc_tensor_shape_list), cast(enc_tensor_shape_list.data_ptr(), POINTER(c_int)), type)

  def SetQKVOutputScales(cls, layer_idx, q_output_scale, k_output_scale, v_output_scale):
    '''
        NOTE(jpyo0803): Set QKV output scales
    '''

    cls.lib.Ext_SetQKVOutputScales(layer_idx, c_float(q_output_scale), c_float(k_output_scale), c_float(v_output_scale))

  def QuantizeAndShiftQ(cls, layer_idx: int, src : int, dst : list):
    '''
        NOTE(jpyo0803): Quantize and Shift Q
    '''
    src_shape = cls.shape_bookkeeper[src]
    assert src_shape is not None
    cls.shape_bookkeeper[src] = None
    for e in dst:
      cls.shape_bookkeeper_uint32[e] = src_shape

    dst_list = torch.tensor(dst, dtype=torch.int32)

    cls.lib.Ext_QuantizeAndShiftQ(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  def QuantizeAndShiftK(cls, layer_idx: int, src : int, dst : list):
    '''
        NOTE(jpyo0803): Quantize and Shift K
    '''
    src_shape = cls.shape_bookkeeper[src]
    assert src_shape is not None
    cls.shape_bookkeeper[src] = None
    for e in dst:
      cls.shape_bookkeeper_uint32[e] = src_shape

    dst_list = torch.tensor(dst, dtype=torch.int32)

    cls.lib.Ext_QuantizeAndShiftK(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  def QuantizeAndShiftP(cls, layer_idx: int, src : int, dst : list):
    '''
        NOTE(jpyo0803): Quantize and Shift P
    '''
    src_shape = cls.shape_bookkeeper[src]
    assert src_shape is not None
    cls.shape_bookkeeper[src] = None
    for e in dst:
      cls.shape_bookkeeper_uint32[e] = src_shape

    dst_list = torch.tensor(dst, dtype=torch.int32)

    cls.lib.Ext_QuantizeAndShiftP(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  def QuantizeAndShiftV(cls, layer_idx: int, src : int, dst : list):
    '''
        NOTE(jpyo0803): Quantize and Shift V
    '''
    src_shape = cls.shape_bookkeeper[src]
    assert src_shape is not None
    cls.shape_bookkeeper[src] = None
    for e in dst:
      cls.shape_bookkeeper_uint32[e] = src_shape

    dst_list = torch.tensor(dst, dtype=torch.int32)

    cls.lib.Ext_QuantizeAndShiftV(layer_idx, src, len(dst_list), cast(dst_list.data_ptr(), POINTER(c_int)))

  def UnshiftAndDequantizeQK(cls, layer_idx: int, src : int, dst : int):
    '''
        NOTE(jpyo0803): Unshift and Dequantize QK
    '''
    src_shape = cls.shape_bookkeeper_uint32[src]
    assert src_shape is not None
    cls.shape_bookkeeper[dst] = src_shape
    cls.shape_bookkeeper_uint32[src] = None

    cls.lib.Ext_UnshiftAndDequantizeQK(layer_idx, src, dst)

  def UnshiftAndDequantizePV(cls, layer_idx: int, src : int, dst : int):
    '''
        NOTE(jpyo0803): Unshift and Dequantize PV
    '''
    src_shape = cls.shape_bookkeeper_uint32[src]
    assert src_shape is not None
    cls.shape_bookkeeper[dst] = src_shape
    cls.shape_bookkeeper_uint32[src] = None

    cls.lib.Ext_UnshiftAndDequantizePV(layer_idx, src, dst)


  def SetAttentionMask(cls, attn_mask):
    '''
        NOTE(jpyo0803): Set Attention Mask
    '''
    assert attn_mask.is_contiguous()
    attn_mask = attn_mask.to(torch.float32)
    cls.lib.Ext_SetAttentionMask(cast(attn_mask.data_ptr(), POINTER(c_float)), attn_mask.shape[-2], attn_mask.shape[-1])

  def Reset(cls):
    print("decoder reset")
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

