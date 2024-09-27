#ifndef SECLLM_TENSOR_H
#define SECLLM_TENSOR_H

#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace jpyo0803 {

template <typename T>
class Tensor {
 public:
  Tensor(const std::vector<int>& shape)
      : shape_(shape),
        data_(std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<int>())) {}

  Tensor(const std::vector<int>& shape, const std::vector<T>& data)
      : shape_(shape), data_(data) {
    if (data.size() != std::accumulate(shape.begin(), shape.end(), 1,
                                       std::multiplies<int>())) {
      throw std::runtime_error("Data size does not match the shape.");
    }
  }

  Tensor(const Tensor& other) : shape_(other.shape_), data_(other.data_) {}

  bool operator==(const Tensor& other) const {
    if (shape_ != other.shape_) {
      return false;
    }

    for (int i = 0; i < data_.size(); ++i) {
      if (data_[i] != other.data_[i]) {
        return false;
      }
    }

    return true;
  }

  std::vector<int> shape() const { return shape_; }

  int num_elements() const { return data_.size(); }

  std::vector<T>& data() { return data_; }

  T& operator[](int idx) { return data_[idx]; }

  void PrintDataFirstAndLast10() const {
    std::cout << "data: [";
    for (int i = 0; i < 10; ++i) {
      std::cout << data_[i];
      if (i != 9) {
        std::cout << ", ";
      }
    }
    std::cout << ", ..., ";
    for (int i = data_.size() - 10; i < data_.size(); ++i) {
      std::cout << data_[i];
      if (i != data_.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  void PrintData() const {
    std::cout << "data: [";
    for (int i = 0; i < data_.size(); ++i) {
      std::cout << data_[i];
      if (i != data_.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  void PrintShape() const {
    std::cout << "shape: [";
    for (int i = 0; i < shape_.size(); ++i) {
      std::cout << shape_[i];
      if (i != shape_.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

 private:
  const std::vector<int> shape_;
  std::vector<T> data_;
};

}  // namespace jpyo0803

#endif  // SECLLM_TENSOR_H