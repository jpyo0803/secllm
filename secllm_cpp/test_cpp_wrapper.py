from ctypes import *

import torch

TEST_LIB_PATH = "./secllm_cpp/libtest.so"

class TestCppWrapper:
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super().__new__(cls)
      cls.lib = cdll.LoadLibrary(TEST_LIB_PATH)
    return cls.instance
  
  def __init__(self):
    cls = type(self)
    if not hasattr(cls, 'instance'):
      cls.__init = True

  @classmethod
  def Matmul_CPU_Eigen(cls, x : torch.tensor, y : torch.tensor):
    # assert x.dim() == 4
    # assert y.dim() == 4
    # assert x.dtype == torch.int8
    # assert y.dtype == torch.int8
    # assert x.is_contiguous()
    # assert y.is_contiguous()

    X_B, X_M, X_K, X_N = x.size()
    Y_B, Y_M, Y_K, Y_N = y.size()
    # assert X_N == Y_N
    # assert X_B == Y_B
    # assert X_M == Y_M

    out = torch.empty(X_B, X_M, X_K, Y_K, dtype=torch.int32)
    cls.lib.Test_Matmul_Eigen(cast(out.data_ptr(), POINTER(c_int32)), cast(x.data_ptr(), POINTER(c_int8)), cast(y.data_ptr(), POINTER(c_int8)), c_int(X_B * X_M), c_int(X_K), c_int(Y_K), c_int(X_N))
    return out
  
  @classmethod
  def Matmul_CPU_Naive(cls, x : torch.tensor, y : torch.tensor):
    # assert x.dim() == 4
    # assert y.dim() == 4
    # assert x.dtype == torch.int8
    # assert y.dtype == torch.int8
    # assert x.is_contiguous()
    # assert y.is_contiguous()

    X_B, X_M, X_K, X_N = x.size()
    Y_B, Y_M, Y_K, Y_N = y.size()
    # assert X_N == Y_N
    # assert X_B == Y_B
    # assert X_M == Y_M

    out = torch.empty(X_B, X_M, X_K, Y_K, dtype=torch.int32)
    cls.lib.Test_Matmul_Naive(cast(out.data_ptr(), POINTER(c_int32)), cast(x.data_ptr(), POINTER(c_int8)), cast(y.data_ptr(), POINTER(c_int8)), c_int(X_B * X_M), c_int(X_K), c_int(Y_K), c_int(X_N))
    return out
  @classmethod
  def GetTimeStamp_Monotonic(cls):
    cls.lib.Test_GetTimeStamp_Monotonic()
