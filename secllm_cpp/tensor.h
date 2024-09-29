#ifndef SECLLM_TENSOR_H
#define SECLLM_TENSOR_H

#include <iomanip>  // For setting precision
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

  // New print function
  void PrintAsTorchStyle(int precision = 4, int threshold = 50) const {
    std::cout << "tensor(";
    if (data_.size() > threshold) {
      PrintMultiDimTorchStyle(0, {}, true,
                              precision);  // Start recursion with empty indices
    } else {
      PrintMultiDimTorchStyle(0, {}, false,
                              precision);  // No truncation if below threshold
    }
    std::cout << ")" << std::endl;
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

  double GetMean() const {
    double sum = 0;
    for (int i = 0; i < data_.size(); ++i) {
      sum += data_[i];
    }
    return sum / data_.size();
  }

  double PosDepSum() const {
    double sum = 0;
    for (int i = 0; i < data_.size(); ++i) {
      sum += data_[i] * (i + 1);
    }
    return sum;
  }

  void PrintCharacteristics() const {
    int dim = shape_.size();
    if (dim > 4) {
      std::cout << "Dim > 4 is not supported." << std::endl;
    } else {
      std::cout << "Dim: " << dim << std::endl;

      if (dim == 4) {
        int B = shape_[0];
        int M = shape_[1];
        int K = shape_[2];
        int N = shape_[3];

        for (int b = 0; b < B; ++b) {
          for (int m = 0; m < M; ++m) {
            std::cout << "[" << b << ", " << m
                      << ", 0, 0] = " << data_[b * M * K * N + m * K * N]
                      << ", [" << b << ", " << m << ", " << K - 1 << ", "
                      << N - 1
                      << "] = " << data_[b * M * K * N + m * K * N + K * N - 1]
                      << std::endl;
          }
        }
      } else if (dim == 3) {
        int B = shape_[0];
        int M = shape_[1];
        int N = shape_[2];

        for (int b = 0; b < B; ++b) {
          std::cout << "[" << b << ", 0, 0] = " << data_[b * M * N] << ", ["
                    << b << ", " << M - 1 << ", " << N - 1
                    << "] = " << data_[b * M * N + M * N - 1] << std::endl;
        }
      } else if (dim == 2) {
        int M = shape_[0];
        int N = shape_[1];

        std::cout << "[0, 0] = " << data_[0] << ", [" << M - 1 << ", " << N - 1
                  << "] = " << data_[M * N - 1] << std::endl;
      } else if (dim == 1) {
        std::cout << "[0] = " << data_[0] << ", [" << shape_[0] - 1
                  << "] = " << data_[shape_[0] - 1] << std::endl;
      }
    }
  }

  // Member function to transpose the tensor
  Tensor<T> Transpose(int dim1, int dim2) {
    if (dim1 >= shape_.size() || dim2 >= shape_.size()) {
      throw std::runtime_error("Invalid dimensions for transpose.");
    }

    std::vector<int> new_shape = shape_;
    std::swap(new_shape[dim1], new_shape[dim2]);

    Tensor<T> result(new_shape);

    // Transpose the data by swapping the specified dimensions
    std::vector<int> strides = CalculateStrides(shape_);
    std::vector<int> new_strides = CalculateStrides(new_shape);

    for (int i = 0; i < data_.size(); ++i) {
      std::vector<int> old_indices = UnravelIndex(i, strides);
      std::swap(old_indices[dim1], old_indices[dim2]);
      int new_idx = RavelIndex(old_indices, new_strides);
      result.data_[new_idx] = data_[i];
    }

    return result;
  }

  // Member function to reshape the tensor
  Tensor<T> Reshape(const std::vector<int>& new_shape) {
    int total_elements = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                         std::multiplies<int>());
    if (total_elements != num_elements()) {
      throw std::runtime_error(
          "New shape must have the same number of elements as the original.");
    }

    return Tensor<T>(new_shape, data_);
  }

 private:
  const std::vector<int> shape_;
  std::vector<T> data_;

 private:
  // Helper to calculate the flat index based on the multi-dimensional indices
  int CalculateIndex(const std::vector<int>& indices) const {
    int index = 0;
    int stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
      index += indices[i] * stride;
      stride *= shape_[i];
    }
    return index;
  }

  // Main print function handling each dimension
  void PrintMultiDimTorchStyle(int dim, std::vector<int> indices, bool truncate,
                               int precision) const {
    if (dim == shape_.size() - 1) {  // Last dimension (base case)
      std::cout << "[";
      int limit = (truncate && shape_[dim] > 3)
                      ? 3
                      : shape_[dim];  // Truncate if dimension > 3
      // Print first 3 elements
      for (int i = 0; i < limit; ++i) {
        std::vector<int> new_indices = indices;
        new_indices.push_back(i);
        int idx = CalculateIndex(new_indices);
        if (idx >= data_.size())
          throw std::runtime_error(
              "Index out of bounds in Tensor data access.");

        std::cout << std::setw(precision + 6) << std::setprecision(precision)
                  << std::scientific << data_[idx];
        if (i != limit - 1) {
          std::cout << ", ";
        }
      }
      // Truncate and print the last 3 elements if dimension is large
      if (truncate && shape_[dim] > 3) {
        std::cout << ", ..., ";
        for (int i = shape_[dim] - 3; i < shape_[dim]; ++i) {
          std::vector<int> new_indices = indices;
          new_indices.push_back(i);
          int idx = CalculateIndex(new_indices);
          std::cout << std::setw(precision + 6) << std::setprecision(precision)
                    << std::scientific << data_[idx];
          if (i != shape_[dim] - 1) {
            std::cout << ", ";
          }
        }
      }
      std::cout << "]";
    } else {  // Higher dimensions
      std::cout << "[";
      int limit = (truncate && shape_[dim] > 3)
                      ? 3
                      : shape_[dim];  // Truncate if dimension > 3
      // Print the first 3 slices
      for (int i = 0; i < limit; ++i) {
        std::vector<int> new_indices = indices;
        new_indices.push_back(i);
        PrintMultiDimTorchStyle(dim + 1, new_indices, truncate, precision);
        if (i != limit - 1) {
          std::cout << ",\n" << std::string(dim + 1, ' ');
        }
      }
      // Truncate and print the last 3 slices if dimension is large
      if (truncate && shape_[dim] > 3) {
        std::cout << ",\n" << std::string(dim + 1, ' ') << "...,\n";
        for (int i = shape_[dim] - 3; i < shape_[dim]; ++i) {
          std::vector<int> new_indices = indices;
          new_indices.push_back(i);
          PrintMultiDimTorchStyle(dim + 1, new_indices, truncate, precision);
          if (i != shape_[dim] - 1) {
            std::cout << ",\n" << std::string(dim + 1, ' ');
          }
        }
      }
      std::cout << "]";
    }
  }

  // Helper function to calculate strides for a given shape
  std::vector<int> CalculateStrides(const std::vector<int>& shape) const {
    std::vector<int> strides(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  // Helper function to convert a flat index to multidimensional indices
  std::vector<int> UnravelIndex(int idx,
                                const std::vector<int>& strides) const {
    std::vector<int> indices(strides.size());
    for (int i = 0; i < strides.size(); ++i) {
      indices[i] = (idx / strides[i]) % shape_[i];
    }
    return indices;
  }

  // Helper function to convert multidimensional indices to a flat index
  int RavelIndex(const std::vector<int>& indices,
                 const std::vector<int>& strides) const {
    int idx = 0;
    for (int i = 0; i < indices.size(); ++i) {
      idx += indices[i] * strides[i];
    }
    return idx;
  }
};

}  // namespace jpyo0803

#endif  // SECLLM_TENSOR_H