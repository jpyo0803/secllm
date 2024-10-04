#ifndef SECLLM_CPP_TIME_STAMP_H
#define SECLLM_CPP_TIME_STAMP_H

#include <stdint.h>
#include <string>

namespace jpyo0803 {
struct TimeStamp {
  TimeStamp(int layer_idx, int worker_id, std::string op)
      : layer_idx(layer_idx), worker_id(worker_id), op(op) {}

  int64_t ts_start;
  int64_t ts_end;

  int layer_idx;
  int worker_id;

  std::string op;
};
}  // namespace jpyo0803

#endif  // SECLLM_CPP_TIME_STAMP_H