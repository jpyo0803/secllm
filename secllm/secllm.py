
from secllm_cpp.secllm_cpp_wrapper import SecLLMCppWrapper
from secllm.task_scheduler import TaskScheduler
from secllm.thread_pool import ThreadPool

NUM_WORKERS = 8

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
      
      cls._thread_pool = ThreadPool(NUM_WORKERS)

      cls._task_scheduler = TaskScheduler(cls._graph, cls._secllm_cpp_wrapper, cls._model_info, cls._thread_pool)


    return cls._instance
  
  def __init__(self, model_info):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True

  @classmethod
  def close(cls):
    cls._thread_pool.shutdown()