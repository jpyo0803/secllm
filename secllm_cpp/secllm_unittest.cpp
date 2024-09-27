#include "secllm.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <numeric>
#include "tensor.h"

#include "book_keeper.h"

#include "func_utils.h"

using namespace std;
using namespace jpyo0803;

TEST(GenerateMultKeyTest, RelativelyPrimeTest) {
  EXPECT_EQ(1, 1);
  constexpr int num_tests = 10000;
  constexpr uint64_t mod = 1ULL << 32;
  for (int i = 0; i < num_tests; ++i) {
    uint32_t key = GenerateMultKey();
    EXPECT_EQ(std::gcd(static_cast<uint64_t>(key), mod), 1);
  }
}

TEST(TensorTest, TensorTest) {
  vector<int> shape{2, 3, 4};
  jpyo0803::Tensor<float> tensor(shape);

  auto ret_shape = tensor.shape();
  for (int i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(ret_shape[i], shape[i]);
  }

  vector<int> shape2 = {3, 4, 5};
  vector<int> data2(shape2[0] * shape2[1] * shape2[2]);
  for (int i = 0; i < data2.size(); ++i) {
    data2[i] = i;
  }
  std::shared_ptr<jpyo0803::Tensor<int>> tensor2 =
      std::make_shared<jpyo0803::Tensor<int>>(shape2, data2);
  EXPECT_EQ(tensor2.use_count(), 1);
  auto tensor3 = tensor2;
  EXPECT_EQ(tensor2.use_count(), 2);
  EXPECT_EQ(tensor3.use_count(), 2);
  auto tensor4 = tensor3;
  EXPECT_EQ(tensor2.use_count(), 3);
  EXPECT_EQ(tensor3.use_count(), 3);
  EXPECT_EQ(tensor4.use_count(), 3);
  auto tensor5 = tensor2;
  EXPECT_EQ(tensor2.use_count(), 4);
  EXPECT_EQ(tensor3.use_count(), 4);
  EXPECT_EQ(tensor4.use_count(), 4);
  EXPECT_EQ(tensor5.use_count(), 4);
  auto tensor6 = tensor3;
  EXPECT_EQ(tensor2.use_count(), 5);
  EXPECT_EQ(tensor3.use_count(), 5);
  EXPECT_EQ(tensor4.use_count(), 5);
  EXPECT_EQ(tensor5.use_count(), 5);
  EXPECT_EQ(tensor6.use_count(), 5);

  EXPECT_TRUE(*tensor2 == *tensor6);
  EXPECT_EQ(data2.size(), tensor5->num_elements());

  tensor3.reset();
  EXPECT_EQ(tensor2.use_count(), 4);
  EXPECT_EQ(tensor4.use_count(), 4);
  EXPECT_EQ(tensor5.use_count(), 4);
  EXPECT_EQ(tensor6.use_count(), 4);
  tensor4.reset();
  EXPECT_EQ(tensor2.use_count(), 3);
  EXPECT_EQ(tensor5.use_count(), 3);
  EXPECT_EQ(tensor6.use_count(), 3);
  tensor5.reset();
  EXPECT_EQ(tensor2.use_count(), 2);
  EXPECT_EQ(tensor6.use_count(), 2);
  tensor6.reset();
  EXPECT_EQ(tensor2.use_count(), 1);
}

TEST(BookKeeperTest, BookKeeperTest) {
  BookKeeper<Tensor<int>> book_keeper(10);

  auto tensor1 = std::make_shared<Tensor<int>>(std::vector<int>{2, 3, 4});

  book_keeper.Keep({0, 1, 2, 9}, tensor1);
  EXPECT_EQ(tensor1.use_count(), 0);

  auto ret_tensor1 = book_keeper.Retrieve(0);
  EXPECT_EQ(ret_tensor1.use_count(), 4);
  {
    auto ret_tensor2 = book_keeper.Retrieve(1);
    auto ret_tensor3 = book_keeper.Retrieve(2);
  }
  EXPECT_EQ(ret_tensor1.use_count(), 2);

  {
    auto ret_tensor4 = book_keeper.Retrieve(9);
    EXPECT_EQ(ret_tensor1.use_count(), 2);
  }

  EXPECT_EQ(ret_tensor1.use_count(), 1);

  ret_tensor1.reset();

  EXPECT_EQ(ret_tensor1.use_count(), 0);
}

TEST(DynamicQuantizeActivationPerTokenAbsmaxTest,
     DynamicQuantizeActivationPerTokenAbsmaxTest) {
  size_t B = 2, M = 3, N = 4;
  std::vector<float> t = {-1.0, 0.5,  2.0,  -3.0, 4.5,  -6.0, 1.0,  -2.0,
                          7.0,  8.0,  -9.0, 10.0, -2.5, 3.0,  -1.5, 1.0,
                          6.0,  -7.5, 8.0,  -9.5, 1.5,  2.5,  3.5,  -4.0};

  auto [q_act, max_vals] =
      jpyo0803::DynamicQuantizeActivationPerTokenAbsmax(t, B, M, N);

  std::vector<int8_t> expected_q_act = {
      -42,  21,  85,  -127, 95, -127, 21,  -42,  89, 102, -114, 127,
      -106, 127, -64, 42,   80, -100, 107, -127, 48, 79,  111,  -127};

  std::vector<float> expected_max_vals = {0.023622, 0.0472441, 0.0787402,
                                          0.023622, 0.0748032, 0.0314961};

  for (size_t i = 0; i < q_act.size(); ++i) {
    EXPECT_EQ(q_act[i], expected_q_act[i]);
  }
  for (size_t i = 0; i < max_vals.size(); ++i) {
    EXPECT_NEAR(max_vals[i], expected_max_vals[i], 1e-6);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int ret = RUN_ALL_TESTS();
  std::cout << "Unittest " << (ret == 0 ? "passed" : "failed") << std::endl;

  return 0;
}