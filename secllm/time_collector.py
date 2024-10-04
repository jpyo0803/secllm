import time

class TimeStamp:
    def __init__(self, layer_idx : int, worker_id, op : str):
        self.layer_idx = layer_idx
        self.worker_id = worker_id
        self.op = op

    def Start(self):
        self.start = time.monotonic_ns()

    def End(self):
        self.end = time.monotonic_ns()

    def get_timestamp(self):
        return self.timestamp

class TimeCollector:
    def __init__(self, num_worker : int):
        self.num_worker = num_worker
        self.time_stamps = [[] for _ in range(num_worker)]

    def Insert(self, worker_id : int, time_stamp : TimeStamp):
        return self.time_stamps[worker_id].append(time_stamp)