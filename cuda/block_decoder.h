
/*
 * block_decoder.cuh
 *
 *  Created on: Jan 23, 2019
 *      Author: totoro
 */

#pragma once

#include <cstdint>
#include <cstddef>

#include "cuda/filter.h"

namespace ruda {

// Note(totoro): This class is copied from "rocksdb/Slice.h".
class RudaSlice {
 public:
  // Create an empty slice.
  __host__ __device__
  RudaSlice() : heap_data_(nullptr), data_(nullptr), size_(0) { }

  // Create a slice that refers to d[0,n-1].
  __host__ __device__
  RudaSlice(char* d, size_t n) : heap_data_(d), data_(nullptr), size_(n) { }

  // Create a slice that refers to s[0,strlen(s)-1]
  /* implicit */
  __host__ __device__
  RudaSlice(char* s) : heap_data_(s), data_(nullptr) {
    size_ = (s == nullptr) ? 0 : strlen(s);
  }

  char* heapData() const { return heap_data_; }

  // Return a pointer to the beginning of the referenced data
  __host__ __device__
  char* data() const { return data_; }

  // Return the length (in bytes) of the referenced data
  __host__ __device__
  size_t size() const { return size_; }

  // Return true iff the length of the referenced data is zero
  __host__ __device__
  bool empty() const { return size_ == 0; }

  // Return the ith byte in the referenced data.
  // REQUIRES: n < size()
  __host__ __device__
  char operator[](size_t n) const {
    // assert(n < size());
    return data_[n];
  }

  __host__ __device__
  void setData(char* data) { data_ = data; }

  __host__ __device__
  void populateDataFromHeap() {
    memcpy(data_, heap_data_, size_);
  }

  // Change this slice to refer to an empty array
  __host__ __device__
  void clear() { data_ = nullptr; size_ = 0; }

 // private: make these public for rocksdbjni access
  char* heap_data_;
  char* data_;
  size_t size_;

  // Intentionally copyable
};

__device__
void DecodeSubDataBlocks(// Parameters
                         const char *cached_data,
                         const uint64_t cached_data_size,
                         const uint64_t start_idx, const uint64_t end_idx,
                         ConditionContext *ctx,
                         // Results
                         unsigned long long int *results_idx, RudaSlice *results_keys,
                         RudaSlice *results_values);

}  // namespace ruda
