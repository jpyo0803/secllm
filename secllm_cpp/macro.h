#ifndef SECLLM_MACRO_H
#define SECLLM_MACRO_H

#include <cstdlib>  // For std::abort()

#if SGX_ENABLE == 0
#include <iostream>
#endif

#define DEBUG_PRINT 0
#define CHECK_SANITY 1
#define INTERNAL_TIME_MEASURE 0

#if SGX_ENABLE == 0
#define ASSERT_ALWAYS(cond, msg)                                               \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "Assertion failed: " << msg << " in file " << __FILE__      \
                << ", function " << __func__ << ", line " << __LINE__ << '\n'; \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#else
#define ASSERT_ALWAYS(cond, msg)                                               \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#endif

#endif  // SECLLM_MACRO_H