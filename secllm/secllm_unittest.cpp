#include "secllm.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>

TEST(GenerateMultKeyTest, RelativelyPrimeTest) {
    EXPECT_EQ(1, 1);
    constexpr int num_tests = 10000;
    constexpr uint64_t mod = 1ULL << 32;
    for (int i = 0; i < num_tests; ++i) {
        uint32_t key = GenerateMultKey();
        EXPECT_EQ(std::gcd(static_cast<uint64_t>(key), mod), 1);
    }
}



int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);


  int ret = RUN_ALL_TESTS();
  std::cout << "Unittest " << (ret == 0 ? "passed" : "failed") << std::endl;

  return 0;
}