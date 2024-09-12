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

  def SiLU(cls, x):
    '''
        NOTE(jpyo0803): in-place SiLU
    '''
    assert x.dim() == 3
    assert x.is_contiguous()
    dtype = x.dtype
    x = x.to(torch.float32)
    B, M, N = x.shape
    cls.lib.SiLU(cast(x.data_ptr(), POINTER(c_float)), B, M, N)
    x = x.to(dtype)
    return x

if __name__ == '__main__':
    secllm = SecLLM()
    secllm.PrintHelloFromCpp()