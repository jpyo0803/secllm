#ifndef SECLLM_CPP_TYPES_H
#define SECLLM_CPP_TYPES_H

namespace jpyo0803 {
enum class ProjectionType {
  kQ = 0,
  kK = 1,
  kV = 2,
  kO = 3,
  kGate = 4,
  kUp = 5,
  kDown = 6,
};
}

#endif  // SECLLM_CPP_TYPES_H