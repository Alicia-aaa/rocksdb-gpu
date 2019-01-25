
/*
 * block_decoder.cuh
 *
 *  Created on: Jan 23, 2019
 *      Author: totoro
 */

#pragma once

#include <cstdint>

namespace ruda {

// Note(totoro): This class is copied from "rocksdb/Slice.h".
class RudaSlice {
 public:
  // Create an empty slice.
  __host__ __device__
  RudaSlice() : data_(nullptr), size_(0) { }

  // Create a slice that refers to d[0,n-1].
  __host__ __device__
  RudaSlice(char* d, size_t n) : data_(d), size_(n) { }

  // Create a slice that refers to s[0,strlen(s)-1]
  /* implicit */
  __host__ __device__
  RudaSlice(char* s) : data_(s) {
    size_ = (s == nullptr) ? 0 : strlen(s);
  }

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

  // Change this slice to refer to an empty array
  __host__ __device__
  void clear() { data_ = nullptr; size_ = 0; }

 // private: make these public for rocksdbjni access
  char* data_;
  size_t size_;

  // Intentionally copyable
};

__device__
void DecodeSubDataBlocks(// Parameters
                         const char *cached_data,
                         const uint64_t cached_data_size,
                         const uint64_t start_idx, const uint64_t end_idx,
                         // Results
                         int *results_idx, RudaSlice *results_keys,
                         RudaSlice *results_values);

}  // namespace ruda
