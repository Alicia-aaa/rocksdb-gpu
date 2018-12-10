
#pragma once

#include <string>
#include <vector>
#include <functional>

namespace ruda {

  const int RUDA_ERR = -1;
  const int RUDA_OK = 0;

  enum Operator {
    EQ, LESS, GREATER, LESS_EQ, GREATER_EQ,
  };

  template <typename T>
  class Comparator;

  int test(int a);

  template <typename T>
  int sstFilter(const std::vector<T>& values,
                const Comparator<T>& cond,
                std::vector<bool>& results);
}
