#ifndef SECLLM_BOOK_KEEPER_H
#define SECLLM_BOOK_KEEPER_H

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "macro.h"

namespace jpyo0803 {

template <typename T>
class BookKeeper {
 public:
  BookKeeper(int size) : dict_(std::vector<std::shared_ptr<T>>(size)) {}

  int size() const { return dict_.size(); }

  void Keep(const std::vector<int>& locs, std::shared_ptr<T>& obj) {
    for (auto loc : locs) {
      ASSERT_ALWAYS(loc >= 0 && loc < dict_.size(), "Invalid location, Keep");
      ASSERT_ALWAYS(dict_[loc] == nullptr, "Location already occupied");

      dict_[loc] = obj;
    }
    obj.reset();
  }

  std::shared_ptr<T> Retrieve(int loc) {
    ASSERT_ALWAYS(loc >= 0 && loc < dict_.size(), "Invalid location, Retrieve");
    ASSERT_ALWAYS(dict_[loc] != nullptr, "No object at the location");

    std::shared_ptr<T> ret = dict_[loc];
    dict_[loc].reset();
    return ret;
  }

  bool IsAvailable(int loc) {
    ASSERT_ALWAYS(loc >= 0 && loc < dict_.size(),
                  "Invalid location, IsAvailable");
    return dict_[loc] != nullptr;
  }

 private:
  std::vector<std::shared_ptr<T>> dict_;
};
}  // namespace jpyo0803

#endif  // SECLLM_BOOK_KEEPER_H