#ifndef SECLLM_CPP_TIME_STAMP_H
#define SECLLM_CPP_TIME_STAMP_H

#include <stdint.h>
#include <chrono>
#include <iostream>
#include <string>

namespace jpyo0803 {
struct TimeStamp {
  TimeStamp(int layer_idx, int worker_id, std::string op)
      : layer_idx(layer_idx), worker_id(worker_id), op(op) {}

  void Start() {
    ts_start = std::chrono::steady_clock::now().time_since_epoch().count();
  }

  void End() {
    ts_end = std::chrono::steady_clock::now().time_since_epoch().count();
  }

  std::string ToString() const {
    // only values with space separated
    return "cpp " + std::to_string(layer_idx) + " " +
           std::to_string(worker_id) + " " + op + " " +
           std::to_string(ts_start) + " " + std::to_string(ts_end);
  }

  void Print() {
    std::cout << "Layer: " << layer_idx << ", Worker: " << worker_id
              << ", Operation: " << op << ", Start Time: " << ts_start
              << ", End Time: " << ts_end << std::endl;
  }

  int64_t ts_start;
  int64_t ts_end;

  int layer_idx;
  int worker_id;

  std::string op;
};
}  // namespace jpyo0803

#endif  // SECLLM_CPP_TIME_STAMP_H