#ifndef SECLLM_MACRO_H
#define SECLLM_MACRO_H

#include <cstdlib>  // For std::abort()
#include <iostream>

// Custom assert that always checks, even in release mode
#define ASSERT_ALWAYS(cond, msg)                                               \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "Assertion failed: " << msg << " in file " << __FILE__      \
                << ", function " << __func__ << ", line " << __LINE__ << '\n'; \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#endif  // SECLLM_MACRO_H