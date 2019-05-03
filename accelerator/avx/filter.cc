
#include "accelerator/avx/filter.h"
#include "accelerator/util.h"

#include <immintrin.h>
#include <vector>

namespace avx {

int _simpleIntNativeFilter(std::vector<long> &source,
                           accelerator::FilterContext ctx,
                           std::vector<long> &results) {
  if (!ctx.isValidOp()) {
    return accelerator::ACC_ERR;
  }

  for (size_t i = 0; i < source.size(); ++i) {
    results[i] = ctx(source[i]);
  }
  return accelerator::ACC_OK;
}

int simpleIntFilter(std::vector<long> &source, accelerator::FilterContext ctx,
                    std::vector<long> &results) {
  results.resize(source.size());
  uint64_t pivot = static_cast<uint64_t>(ctx._pivot);
  int size = (int) source.size();

  // If total source size is under 8, just run native filter.
  if (size < 8) {
    return _simpleIntNativeFilter(source, ctx, results);
  }

  // Round up size to lower multiple of 8
  int rounded_size = (size / 8) * 8;
  int remain_size = size % 8;

  printf("[AVX][simpleIntFilter] Origin: %d, Round: %d, Remain: %d\n",
      size, rounded_size, remain_size);

  __m256i pivots = _mm256_set_epi32(
      pivot, pivot, pivot, pivot, pivot, pivot, pivot, pivot);
  __m256i mask = _mm256_cmpeq_epi32(pivots, pivots);  // 0xffffffff mask
  for (int i = 0; i < rounded_size; i += 8) {
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

  // Process remain data by native loop...
  for (int i = 0; i < remain_size; ++i) {
    int idx = rounded_size + i;
    results[i] = ctx(source[idx]);
  }

  return accelerator::ACC_OK;
}

int _recordNativeFilter(std::vector<rocksdb::Slice> &raw_records,
                           const rocksdb::SlicewithSchema &schema_key,
                           std::vector<rocksdb::PinnableSlice> &results) {
  if (!schema_key.context.isValidOp()) {
    return accelerator::ACC_ERR;
  }

  for (auto &raw_record : raw_records) {
    long col_value = accelerator::convertRecord(schema_key, raw_record.data_);
    if (schema_key.context(col_value) == 1) {
      results.emplace_back(std::move(rocksdb::PinnableSlice(
          raw_record.data_, raw_record.size_)));
    }
  }
  return accelerator::ACC_OK;
}

int recordFilter(std::vector<rocksdb::Slice> &raw_records,
                 const rocksdb::SlicewithSchema &schema_key,
                 std::vector<rocksdb::PinnableSlice> &results) {
  // printf("[AVX][recordIntFilter] START raw_record_size: %lu\n", raw_records.size());
  uint64_t pivot = static_cast<uint64_t>(schema_key.context._pivot);
  int size = (int) raw_records.size();

  // If operator is INVALID, put all records to results.
  if (schema_key.context._op == accelerator::INVALID) {
    for (auto &raw_record : raw_records) {
      results.emplace_back(rocksdb::PinnableSlice(
          raw_record.data_, raw_record.size_));
    }
    return accelerator::ACC_OK;
  }

  // If total raw_records size is under 8, just run native filter.
  if (size < 8) {
    return _recordNativeFilter(raw_records, schema_key, results);
  }

  // Round up size to lower multiple of 8
  int rounded_size = (size / 8) * 8;
  int remain_size = size % 8;

//  printf("[AVX][recordIntFilter] Origin: %d, Round: %d, Remain: %d\n",
//      size, rounded_size, remain_size);

  // printf("[AVX][recordIntFilter] Break Point 1\n");
  __m256i pivots = _mm256_set_epi32(
      pivot, pivot, pivot, pivot, pivot, pivot, pivot, pivot);
  __m256i mask = _mm256_cmpeq_epi32(pivots, pivots);  // 0xffffffff mask
  // printf("[AVX][recordIntFilter] Break Point 2\n");
  for (int i = 0; i < rounded_size; i += 8) {
    // printf("[AVX][recordIntFilter] Break Point 3-1\n");
    // printf("[AVX][recordIntFilter] Size: %lu, Raw Record: ", raw_records[i].size_);
    __m256i sources = _mm256_set_epi32(
        accelerator::convertRecord(schema_key, raw_records[i].data_),
        accelerator::convertRecord(schema_key, raw_records[i+1].data_),
        accelerator::convertRecord(schema_key, raw_records[i+2].data_),
        accelerator::convertRecord(schema_key, raw_records[i+3].data_),
        accelerator::convertRecord(schema_key, raw_records[i+4].data_),
        accelerator::convertRecord(schema_key, raw_records[i+5].data_),
        accelerator::convertRecord(schema_key, raw_records[i+6].data_),
        accelerator::convertRecord(schema_key, raw_records[i+7].data_));
    // printf("[AVX][recordIntFilter] Break Point 3-2\n");
    __m256i result;
    switch (schema_key.context._op) {
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
        printf("[AVX][recordFilter] INVALID\n");
        return accelerator::ACC_ERR;
    }
    // printf("[AVX][recordIntFilter] Break Point 3-3\n");

    unsigned result_mask = _mm256_movemask_epi8(result);
    unsigned compare_mask = 0xf0000000;
    int limit = (i + 7) < size ? 8 : size & 7;
    // printf("[AVX][recordIntFilter] Break Point 3-4\n");
    for (int j = 0; j < limit; ++j) {
      if (result_mask & compare_mask) {
        results.emplace_back(
            std::move(rocksdb::PinnableSlice(
                raw_records[i + j].data_,
                raw_records[i + j].size_)));
      }
      compare_mask = compare_mask >> 4;
    }
    // printf("[AVX][recordIntFilter] Break Point 3-5\n");
  }

  // Process remain records by native loop...
  for (int i = 0; i < remain_size; ++i) {
    int idx = rounded_size + i;
    long col_value = accelerator::convertRecord(
        schema_key, raw_records[idx].data_);
    if (schema_key.context(col_value) == 1) {
      results.emplace_back(std::move(rocksdb::PinnableSlice(
          raw_records[idx].data_, raw_records[idx].size_)));
    }
  }

  // printf("[AVX][recordIntFilter] END\n");
  return accelerator::ACC_OK;
}

}  // namespace avx
