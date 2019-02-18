
#include "simple_filter.h"

// 256 / 4 = 64

namespace avx_filter {

int _avxIntCompare() {
  
}

int _avxSimpleIntFilter(const std::vector<int> &source,
                        const FilterContext ctx,
                        std::vector<int> &results) {
  #pragma omp simd
  #pragma vector aligned
  for (int i = 0; i < source.size(); ++i) {
    results[i] = ctx(source[i]);
  }
  return AVX_OK;
}

int avxSimpleIntFilter(const std::vector<int> &source,
                       const FilterContext ctx,
                       std::vector<int> &results) {
  results.resize(source.size());
  return _avxSimpleIntFilter(source, ctx, results);
}

}  // namespace avx_filter
