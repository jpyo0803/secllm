#ifndef SECLLM_CPP_THREAD_POOL_H
#define SECLLM_CPP_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace jpyo0803 {

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);

  ~ThreadPool();

  void enqueue_task(std::function<void()> task);

  void shutdown();

 private:
  void worker();

  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;

  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop;

  int num_threads_;
};

}  // namespace jpyo0803

#endif  // SECLLM_CPP_THREAD_POOL_H