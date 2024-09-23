from secllm_cpp.secllm_cpp_wrapper import SecLLMCppWrapper

class SecLLM:
  def __new__(cls, config):
    if not hasattr(cls, 'instance'):
      cls._instance = super(SecLLM, cls).__new__(cls)

      cls.config = config

      cls.secllm_cpp_wrapper = SecLLMCppWrapper(cls.config.num_hidden_layers)
    return cls._instance
  
  def __init__(self, config):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True

  