import GPUtil
from threading import Thread
import time
import psutil

class MemoryMonitor(Thread):
    def __init__(self, delay = 0.1):
        super(MemoryMonitor, self).__init__()
        self.stopped = False
        self.delay = delay

        self.max_cpu_mem_used = 0
        self.max_gpu_mem_used = 0

        self.start()
    
    def run(self):
        process = psutil.Process()

        while not self.stopped:
            try:
                self.max_cpu_mem_used = max(self.max_cpu_mem_used, process.memory_info().rss / 1024 / 1024)
            except psutil.NoSuchProcess:
                break

            # GPUtil.showUtilization()
            self.max_gpu_mem_used = max(self.max_gpu_mem_used, GPUtil.getGPUs()[0].memoryUsed)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        time.sleep(1) # Allow the thread time to stop

    def get_max_mem_used(self):
        assert self.stopped, "The monitor thread is still running"
        return (self.max_cpu_mem_used , self.max_gpu_mem_used)


if __name__ == '__main__':
    import torch

    memory_monitor = MemoryMonitor(0.1)

    a = torch.randn(256, 1024, 1024, dtype=torch.float16)
    b = torch.randn(512, 1024, 1024, dtype=torch.float16, device='cuda')

    time.sleep(1)

    memory_monitor.stop()

    peak_cpu_mem_used, peak_gpu_mem_used = memory_monitor.get_max_mem_used()
    print(f"Peak CPU memory usage: {peak_cpu_mem_used:0.3f} MB")
    print(f"Peak GPU memory usage: {peak_gpu_mem_used:0.3f} MB")