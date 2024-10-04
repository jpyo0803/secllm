#include "thread_pool.h"

namespace jpyo0803 {

ThreadPool::ThreadPool(int num_threads)
    : stop(false), num_threads_(num_threads) {
  for (int i = 0; i < num_threads_; ++i) {
    workers.emplace_back([this, i] { worker(i); });
  }
}

ThreadPool::~ThreadPool() {
  shutdown();
}

void ThreadPool::enqueue_task(std::function<void(int)> task) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    tasks.emplace(task);
  }

  condition.notify_one();
}

void ThreadPool::worker(int thread_id) {
  while (true) {
    std::function<void(int)> task;

    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      condition.wait(lock, [this] { return stop || !tasks.empty(); });

      if (stop && tasks.empty()) {
        return;
      }

      task = std::move(tasks.front());
      tasks.pop();
    }

    task(thread_id);
  }
}

void ThreadPool::shutdown() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }

  condition.notify_all();

  for (std::thread& worker : workers) {
    worker.join();
  }
}

}  // namespace jpyo0803