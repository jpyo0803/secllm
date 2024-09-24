from ctypes import *
import torch

SECLLM_LIB_PATH = './secllm_cpp/libsecllm.so'

MAX_NUM_LAYERS = 32
MAX_NUM_OPERATIONS = 91
MAX_NUM_INPUTS = 3

def GetBookKeeperLinearIndex(layer_index, operation_index, input_index):
  # NOTE(jpyo0803): debugging purpose
  assert layer_index < MAX_NUM_LAYERS
  assert operation_index < MAX_NUM_OPERATIONS
  assert input_index < MAX_NUM_INPUTS

  return layer_index * 300 + input_index * 100 + operation_index

class SecLLMCppWrapper:
  def __new__(cls, num_hidden_layers):
    if not hasattr(cls, 'instance'):
      cls._instance = super().__new__(cls)
      cls.lib = cdll.LoadLibrary(SECLLM_LIB_PATH)

      cls.lib.Ext_CreateSecLLM(num_hidden_layers)

      cls.shape_bookkeeper = [None for _ in range(MAX_NUM_LAYERS * 300)]
    return cls._instance
  
  def __init__(self, num_hidden_layers):
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

  def SwiGLU(cls, gate_in, up_in):
    '''
        NOTE(jpyo0803): in-place SwiGLU
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
    cls.lib.Ext_SwiGLU(cast(gate_in.data_ptr(), POINTER(c_float)), cast(up_in.data_ptr(), POINTER(c_float)), B, M, N)
    gate_in = gate_in.to(dtype)
    return gate_in

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

  def ElementwiseAdd(cls, x, y):
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
    cls.lib.Ext_ElementwiseAdd(cast(x.data_ptr(), POINTER(c_float)), cast(y.data_ptr(), POINTER(c_float)), B, M, N)
    x = x.to(dtype)
    return x
  
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

  def BookKeeperLoad(cls, layer_index, operation_index, input_index):
    loc = GetBookKeeperLinearIndex(layer_index, operation_index, input_index)
    
    shape = cls.shape_bookkeeper[loc]
    assert shape is not None
    cls.shape_bookkeeper[loc] = None
    shape_list = torch.tensor(shape, dtype=torch.int32)
    out = torch.empty(shape, dtype=torch.float32)

    cls.lib.Ext_BookKeeperLoad(loc, cast(out.data_ptr(), POINTER(c_float)), len(shape), cast(shape_list.data_ptr(), POINTER(c_int)))

    return out
  
  def ReplicateTensor(cls, src : int, dst : list):
    # be careful src is in int64
    for e in dst:
      cls.shape_bookkeeper[e] = cls.shape_bookkeeper[src]
    cls.shape_bookkeeper[src] = None

    dst = torch.tensor(dst, dtype=torch.int32)
    cls.lib.Ext_ReplicateTensor(src, cast(dst.data_ptr(), POINTER(c_int)), len(dst))


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

