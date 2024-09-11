from ctypes import *

SECLLM_LIB_PATH = './secllm/secllm.so'

print("secllm.py: Loading SecLLM library from", SECLLM_LIB_PATH)

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

if __name__ == '__main__':
    secllm = SecLLM()
    secllm.PrintHelloFromCpp()