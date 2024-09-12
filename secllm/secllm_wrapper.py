from ctypes import *
import torch

SECLLM_LIB_PATH = './secllm/secllm.so'

class SecLLM:
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls._instance = super().__new__(cls)
      cls.lib = cdll.LoadLibrary(SECLLM_LIB_PATH)
    return cls._instance
  
  def __init__(self):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True

  @classmethod
  def PrintHelloFromCpp(cls):
    cls.lib.PrintHelloFromCpp()

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

if __name__ == '__main__':
    secllm = SecLLM()
    secllm.PrintHelloFromCpp()