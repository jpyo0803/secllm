#ifndef SECLLM_BOOK_KEEPER_H
#define SECLLM_BOOK_KEEPER_H

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jpyo0803 {

template <typename T>
class BookKeeper {
 public:
  BookKeeper(int size) : dict_(std::vector<std::shared_ptr<T>>(size)) {}

  int size() const { return dict_.size(); }

  void Keep(const std::vector<int>& locs, std::shared_ptr<T>& obj) {
    for (auto loc : locs) {
      if (loc < 0 || loc >= dict_.size()) {
        std::string msg = "Invalid location: " + std::to_string(loc);
        throw std::out_of_range(msg);
      }
      dict_[loc] = obj;
    }
    obj.reset();
    // NOTE(jpyo0803): Notice
  }

  std::shared_ptr<T> Retrieve(int loc) {
    if (loc < 0 || loc >= dict_.size()) {
      std::string msg = "Invalid location: " + std::to_string(loc);
      throw std::out_of_range(msg);
    }
    if (dict_[loc] == nullptr) {
      std::string msg = "No object at the location: " + std::to_string(loc);
      throw std::runtime_error(msg);
    }
    std::shared_ptr<T> ret = dict_[loc];
    dict_[loc].reset();
    return ret;
  }

 private:
  std::vector<std::shared_ptr<T>> dict_;
};
}  // namespace jpyo0803

#endif  // SECLLM_BOOK_KEEPER_H