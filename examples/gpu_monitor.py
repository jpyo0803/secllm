import GPUtil
from threading import Thread
import time

class GpuMonitor(Thread):
    def __init__(self, delay = 0.1):
        super(GpuMonitor, self).__init__()
        self.stopped = False
        self.delay = delay

        self.max_mem_used = 0

        self.start()
    
    def run(self):
        while not self.stopped:
            # GPUtil.showUtilization()
            self.max_mem_used = max(self.max_mem_used, GPUtil.getGPUs()[0].memoryUsed)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        time.sleep(self.delay * 5) # Allow the thread time to stop

    def get_max_mem_used(self):
        assert self.stopped, "The monitor thread is still running"
        return self.max_mem_used
