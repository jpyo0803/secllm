from ctypes import *
import torch

SECLLM_LIB_PATH = './secllm/libsecllm.so'

class SecLLM:
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls._instance = super().__new__(cls)
      cls.lib = cdll.LoadLibrary(SECLLM_LIB_PATH)

      cls.lib.CreateSecLLM()
    return cls._instance
  
  def __init__(self):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True

  @classmethod
  def PrintHelloFromCpp(cls):
    cls.lib.PrintHelloFromCpp()

  @classmethod
  def SecLLMTestPrint(cls):
    cls.lib.SecLLMTestPrint()

  @classmethod
  def Softmax(cls, x):
    '''
        NOTE(jpyo0803): in-place Softmax
    '''
    assert x.dim() == 4
    assert x.is_contiguous()
    dtype = x.dtype
    x = x.to(torch.float32)
    B, M, N, K = x.shape
    cls.lib.Softmax(cast(x.data_ptr(), POINTER(c_float)), B, M, N, K)
    x = x.to(dtype)
    return x

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
    cls.lib.SwiGLU(cast(gate_in.data_ptr(), POINTER(c_float)), cast(up_in.data_ptr(), POINTER(c_float)), B, M, N)
    gate_in = gate_in.to(dtype)
    return gate_in

  def RMSNorm(cls, x, weight, eps):
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
    cls.lib.RMSNorm(cast(x.data_ptr(), POINTER(c_float)), cast(weight.data_ptr(), POINTER(c_float)), B, M, N, c_float(eps))
    x = x.to(dtype)
    return x

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
    cls.lib.ElementwiseAdd(cast(x.data_ptr(), POINTER(c_float)), cast(y.data_ptr(), POINTER(c_float)), B, M, N)
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
    cls.lib.ApplyRotaryPosEmb(cast(q.data_ptr(), POINTER(c_float)), cast(k.data_ptr(), POINTER(c_float)), cast(cos.data_ptr(), POINTER(c_float)), cast(sin.data_ptr(), POINTER(c_float)), B, Q_M, K_M, N, K)
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

    cls.lib.LlamaRotaryEmbedding(cast(inv_freq.data_ptr(), POINTER(c_float)), inv_freq.shape[0], cast(position_ids.data_ptr(), POINTER(c_float)), position_ids.shape[1], cast(cos.data_ptr(), POINTER(c_float)), cast(sin.data_ptr(), POINTER(c_float)))
    cos = cos.to(input_dtype)
    sin = sin.to(input_dtype)
    return cos, sin


if __name__ == '__main__':
    secllm = SecLLM()
  #  secllm.PrintHelloFromCpp()

    obj = secllm.CreateSecLLM()
    secllm.SecLLMTestPrint(obj)