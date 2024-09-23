
from secllm_cpp.secllm_cpp_wrapper import SecLLMCppWrapper
from secllm.task_scheduler import TaskScheduler

class SecLLM:
  def __new__(cls, config):
    if not hasattr(cls, 'instance'):
      cls._instance = super(SecLLM, cls).__new__(cls)

      cls.config = config

      cls.secllm_cpp_wrapper = SecLLMCppWrapper(cls.config.num_hidden_layers)

      # Load list from file from 'dependency_graph.txt'
      cls.graph = {}
      with open('dependency_graph.txt', 'r') as f:
        for line in f:
          key, value = line.strip().split(':')
          cls.graph[int(key)] = eval(value)

      cls.task_scheduler = TaskScheduler(cls.graph, cls.secllm_cpp_wrapper)
      cls.task_scheduler()
      assert False

    return cls._instance
  
  def __init__(self, config):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True
