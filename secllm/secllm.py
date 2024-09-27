
from secllm_cpp.secllm_cpp_wrapper import SecLLMCppWrapper
from secllm.task_scheduler import TaskScheduler

class SecLLM:
  def __new__(cls, model_info):
    if not hasattr(cls, 'instance'):
      cls._instance = super(SecLLM, cls).__new__(cls)

      cls._config = model_info.config

      cls._enc_key_pool_size = 128

      cls._secllm_cpp_wrapper = SecLLMCppWrapper(cls._config, cls._enc_key_pool_size)

      # Load list from file from 'dependency_graph.txt'
      cls._graph = {}
      with open('dependency_graph.txt', 'r') as f:
        for line in f:
          key, value = line.strip().split(':')
          cls._graph[int(key)] = eval(value)

      model_info.tensor_buffer = [None for _ in range(32 * 300)]

      cls._model_info = model_info
      
      cls._task_scheduler = TaskScheduler(cls._graph, cls._secllm_cpp_wrapper, cls._model_info)

    return cls._instance
  
  def __init__(self, model_info):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True
