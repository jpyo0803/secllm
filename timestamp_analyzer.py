
TIMESTAMP_FILE = 'collected_timestamp.txt'

class ProcessedTimestamp:
  def __init__(self, py_or_cpp, layer_idx, worker_id, op, start, end):
    self.py_or_cpp = py_or_cpp
    self.layer_idx = layer_idx
    self.worker_id = worker_id
    self.op = op
    self.start = start
    self.end = end
    self.dt = end - start


def ReadInData():
  processed_timestamps = []

  # Read the collected timestamp from a file
  with open(TIMESTAMP_FILE, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
      # space separated values
      values = line.strip().split()
      # first value is 'py' or 'cpp'
      # second value is layer_idx
      # third value is worker_id
      # second to the last value is start timestamp
      # last value is end timestamp
      # other values are concatenated to form the operation name

      py_or_cpp = values[0]
      layer_idx = int(values[1])
      worker_id = int(values[2])
      op = ' '.join(values[3:-2])
      start = int(values[-2])
      end = int(values[-1])
      processed_timestamps.append(ProcessedTimestamp(py_or_cpp, layer_idx, worker_id, op, start, end))
  return processed_timestamps

def GroupByCategory(data):
  # Group the data by category
  categories = {}
  for d in data:
    if d.op not in categories:
      categories[d.op] = []
    categories[d.op].append(d)
  return categories

def ComputeAverageByCategory(data_by_op):
  # Compute the average time taken for each category
  avg_time_by_op = {}
  for op, values in data_by_op.items():
    total_time = 0
    for v in values:
      total_time += v.dt
    avg_time_by_op[op] = total_time / len(values)
  return avg_time_by_op

def ComputePercentageByCategory(data_by_op, sort_increasing_order=False):
  # Compute the percentage of time taken for each category
  total_time = 0
  for op, values in data_by_op.items():
    for v in values:
      total_time += v.dt
  percentage_by_op = {}
  for op, values in data_by_op.items():
    total_time_op = 0
    for v in values:
      total_time_op += v.dt
    percentage_by_op[op] = total_time_op / total_time

  if sort_increasing_order:
    percentage_by_op = dict(sorted(percentage_by_op.items(), key=lambda item: item[1], reverse=True))
  else:
    percentage_by_op = dict(sorted(percentage_by_op.items(), key=lambda item: item[1]))

  return percentage_by_op

if __name__ == '__main__':
  data = ReadInData()

  data_by_op = GroupByCategory(data)
  
  avg_time_by_op = ComputeAverageByCategory(data_by_op)

  percentage_by_op = ComputePercentageByCategory(data_by_op)
  # sort by percentage, so that with the highest percentage is at the bottom

  for op, percentage in percentage_by_op.items():
    print(f'{op}: {percentage * 100:.2f} %')