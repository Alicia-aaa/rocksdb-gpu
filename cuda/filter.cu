#include "filter.h"

#include <stdio.h>
#include <vector>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <algorithm>

namespace ruda {

  template <typename T>
  class Comparator {
    public:
      Comparator(Operator op, T pivot) {
        this->_op = op;
        this->_pivot = pivot;
      }

    __host__ __device__
    bool operator()(const T& target) const {
      printf("Print on Comparator\n");
      return false;
      // switch (this->_op) {
      //   case EQ:
      //     return std::equal_to<T>()(target, pivot);
      //   case LESS:
      //     return std::less<T>()(target, pivot);
      //   case GREATER:
      //     return std::greater<T>()(target, pivot);
      //   case LESS_EQ:
      //     return std::less_equal<T>()(target, pivot);
      //   case GREATER_EQ:
      //     return std::greater_equal<T>()(target, pivot);
      //   default:
      //     throw new runtime_error("[RUDA][COMPARATOR] Invalid Operator Type");
      // }
    }

    private:
      Operator _op;
      T _pivot;
  };

  int test(int a) {
    return a + 3;
  }

  template <typename T>
  int sstFilter(const std::vector<T>& values,
                const Comparator<T>& cond,
                std::vector<bool>& results) {
    printf("Print on sstFilter\n");
    thrust::device_vector<T> d_values(values);
    thrust::device_vector<int> d_results(values.size());

    thrust::copy_if(d_values.begin(), d_values.end(), d_results.begin(), cond());
    thrust::copy(d_values.begin(), d_values.end(), results.begin());

    return ruda::RUDA_OK;
  }

  // Workaround(totoro): Template implementation needs to notify to Compiler.
  // https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
  template int sstFilter<int>(const std::vector<int>& values,
                              const Comparator<int>& cond,
                              std::vector<bool>& results);
  template int sstFilter<float>(const std::vector<float>& values,
                                const Comparator<float>& cond,
                                std::vector<bool>& results);
  template int sstFilter<std::string>(const std::vector<std::string>& values,
                                      const Comparator<std::string>& cond,
                                      std::vector<bool>& results);
}

