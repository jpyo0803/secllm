
from secllm_cpp.secllm_cpp_wrapper import SecLLMCppWrapper
from secllm.task_scheduler import TaskScheduler
from secllm.thread_pool import ThreadPool
from secllm.time_collector import TimeCollector

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
      
      cls._time_collector = TimeCollector(NUM_WORKERS)
      cls._thread_pool = ThreadPool(NUM_WORKERS)

      cls._task_scheduler = TaskScheduler(cls._graph, cls._secllm_cpp_wrapper, cls._model_info, cls._thread_pool, cls._time_collector)
    return cls._instance
  
  def __init__(self, model_info):
    cls = type(self)
    if not hasattr(cls, '__init'):
      cls.__init = True

  @classmethod
  def close(cls):
    cls._thread_pool.shutdown()

    output_file_name = "collected_timestamp.txt"

    # Output the collected timestamp to a file
    with open(output_file_name, 'w') as f:
      f.write("(py or cpp) layer_idx worker_id op start end\n")
      for worker_id in range(NUM_WORKERS):
        for time_stamp in cls._time_collector.time_stamps[worker_id]:
          f.write(f"{'py'} {time_stamp.layer_idx} {time_stamp.worker_id} {time_stamp.op} {time_stamp.start} {time_stamp.end}\n")

    cls._secllm_cpp_wrapper.Close(output_file_name)
    # Collect timestamp from C++