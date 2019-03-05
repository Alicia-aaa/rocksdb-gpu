
#include "accelerator/avx/filter.h"

#include <immintrin.h>
#include <vector>

namespace avx {

int simpleIntFilter(std::vector<long> &source, accelerator::FilterContext ctx,
                    std::vector<long> &results) {
  results.resize(source.size());
  uint64_t pivot = static_cast<uint64_t>(ctx._pivot);
  int size = (int) source.size();
  // Round up size to next multiple of 8
  int roundedSize = (size + 7) & ~7UL;

  __m256i pivots = _mm256_set_epi32(
      pivot, pivot, pivot, pivot, pivot, pivot, pivot, pivot);
  __m256i mask = _mm256_cmpeq_epi32(pivots, pivots);  // 0xffffffff mask
  for (int i = 0; i < roundedSize; i += 8) {
    __m256i sources = _mm256_set_epi32(
        source[i], source[i+1], source[i+2], source[i+3], source[i+4],
        source[i+5], source[i+6], source[i+7]);
    __m256i result;
    switch (ctx._op) {
      case accelerator::EQ: {
        result = _mm256_cmpeq_epi32(sources, pivots);
        break;
      }
      case accelerator::GREATER: {
        result = _mm256_cmpgt_epi32(sources, pivots);
        break;
      }
      case accelerator::LESS: {
        result = _mm256_cmpgt_epi32(sources, pivots);
        result = _mm256_xor_si256(result, mask);
        break;
      }
      case accelerator::GREATER_EQ: {
        __m256i greater = _mm256_cmpgt_epi32(sources, pivots);
        __m256i eq = _mm256_cmpeq_epi32(sources, pivots);
        result = _mm256_or_si256(greater, eq);
        break;
      }
      case accelerator::LESS_EQ: {
        __m256i greater = _mm256_cmpgt_epi32(sources, pivots);
        __m256i less = _mm256_xor_si256(greater, mask);
        __m256i eq = _mm256_cmpeq_epi32(sources, pivots);
        result = _mm256_or_si256(less, eq);
        break;
      }
      default:
        return accelerator::ACC_ERR;
    }

    unsigned result_mask = _mm256_movemask_epi8(result);
    unsigned compare_mask = 0xf0000000;
    int limit = (i + 7) < size ? 8 : size & 7;
    for (int j = 0; j < limit; ++j) {
      if (result_mask & compare_mask) {
        results[i + j] = 1;
      } else {
        results[i + j] = 0;
      }
      compare_mask = compare_mask >> 4;
    }
  }

  return accelerator::ACC_OK;
}

}  // namespace avx
