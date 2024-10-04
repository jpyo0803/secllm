import threading
from queue import Queue
import time

class ThreadPool:
    def __init__(self, num_threads = 1):
        self.tasks = Queue()
        self.num_threads = num_threads
        self.workers = []
        self.stop = False

        # Create worker threads
        for worker_id in range(num_threads):
            worker = threading.Thread(target=self.worker, args=(worker_id,))
            worker.start()
            self.workers.append(worker)

    def enqueue_task(self, task):
        """
        Add a task to the queue. The task is expected to be a callable object (like a function or lambda).
        """
        self.tasks.put(task)

    def worker(self, worker_id):
        """
        Worker thread function that processes tasks.
        """
        while not self.stop:
            try:
                task = self.tasks.get(timeout=1)  # Timeout to check if stop is True
                # task.print_info()
                task(worker_id)  # Execute the task
                self.tasks.task_done()
            except Exception as e:
                pass

    def shutdown(self):
        """
        Stop all worker threads and wait for them to finish.
        """
        self.stop = True
        for worker in self.workers:
            worker.join()
        print("Shutdown ThreadPool")

# Test the thread pool
if __name__ == "__main__":
    def task1():
        print(f"Task 1 executed by {threading.current_thread().name}")
        time.sleep(1)

    def task2():
        print(f"Task 2 executed by {threading.current_thread().name}")
        time.sleep(2)

    # Create a thread pool with 4 threads
    pool = ThreadPool(4)

    # Enqueue tasks
    for _ in range(5):
        pool.enqueue_task(task1)
        pool.enqueue_task(task2)

    # Wait for a while before shutting down
    time.sleep(5)
    pool.shutdown()
    print("Thread pool shutdown.")