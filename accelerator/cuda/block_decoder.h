
/*
 * block_decoder.cuh
 *
 *  Created on: Jan 23, 2019
 *      Author: totoro
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <iostream>

#include "accelerator/common.h"
#include "rocksdb/slice.h"

const int kMaxRudaSliceDataSize = 64;

namespace ruda {

// TODO(totoro): Not used... need to remove this class...
enum class RudaSliceMode {
  HEAP, STACK
};

// Note(totoro): This class is copied from "rocksdb/Slice.h".
// TODO(totoro): Not used... need to remove this class...
class RudaSlice {
 public:
  // Create an empty slice.
  __host__ __device__
  RudaSlice()
    : mode_(RudaSliceMode::STACK), heap_data_(nullptr), data_(nullptr),
      size_(0) { }

  // Create a slice that refers to d[0,n-1].
  __host__ __device__
  RudaSlice(char* d, size_t n)
    : mode_(RudaSliceMode::HEAP), heap_data_(d), data_(nullptr), size_(n) { }

  // Create a slice that refers to s[0,strlen(s)-1]
  /* implicit */
  __host__ __device__
  RudaSlice(char* s)
    : mode_(RudaSliceMode::HEAP), heap_data_(s), data_(nullptr) {
    size_ = (s == nullptr) ? 0 : strlen(s);
  }

  // Return a pointer to the beginning of the referenced data
  __host__ __device__
  char* data() const { return data_; }

  __host__ __device__
  char* heapData() const {
    if (mode_ != RudaSliceMode::HEAP) {
      return nullptr;
    }
    return heap_data_;
  }

  __host__ __device__
  const char* stackData() const {
    if (mode_ != RudaSliceMode::STACK) {
      return "";
    }
    return stack_data_;
  }

  // Return the length (in bytes) of the referenced data
  __host__ __device__
  size_t size() const { return size_; }

  __host__ __device__
  void copyToStack(const char* s, size_t n) {
    if (mode_ != RudaSliceMode::STACK) {
      return;
    }
    memcpy(stack_data_, s, sizeof(char) * n);
    size_ = n;
  }

  // Return true iff the length of the referenced data is zero
  __host__ __device__
  bool empty() const { return size_ == 0; }

  // Return the ith byte in the referenced data.
  // REQUIRES: n < size()
  __host__ __device__
  char operator[](size_t n) const {
    // assert(n < size());
    if (mode_ == RudaSliceMode::HEAP) {
      return data_[n];
    } else {
      return stack_data_[n];
    }
  }

  __host__ __device__
  void setData(char* data) { data_ = data; }

  __host__ __device__
  void populateDataFromHeap() {
    if (mode_ != RudaSliceMode::HEAP) {
      return;
    }
    memcpy(data_, heap_data_, size_);
  }

  // Change this slice to refer to an empty array
  __host__ __device__
  void clear() {
    mode_ = RudaSliceMode::STACK;
    heap_data_ = nullptr;
    data_ = nullptr;
    memset(stack_data_, 0, sizeof(char) * kMaxRudaSliceDataSize);
    size_ = 0;
  }

 // private: make these public for rocksdbjni access
  RudaSliceMode mode_;

  char* heap_data_;
  char stack_data_[kMaxRudaSliceDataSize];
  char* data_;
  size_t size_;

  // Intentionally copyable
};

// TODO(totoro): Not used... need to remove this class...
class RudaKVPair {
 public:
  __host__ __device__
  RudaKVPair() : key_(RudaSlice()), value_(RudaSlice()) {}

  __host__ __device__
  void clear() {
    key_.clear();
    value_.clear();
  }

  __host__ __device__
  RudaSlice* key() { return &key_; }

  __host__ __device__
  RudaSlice* value() { return &value_; }

 private:
  RudaSlice key_;   /* 128 byte */
  RudaSlice value_; /* 128 byte */
};

class RudaIndexEntry {
 public:
  __host__ __device__
  RudaIndexEntry() : used(false), start_(0), end_(0) {}

  __host__ __device__
  RudaIndexEntry(size_t start, size_t end) : used(true), start_(start), end_(end) {}

  bool used;
  size_t start_;
  size_t end_;
};

class RudaKVIndexPair {
 public:
  __host__ __device__
  RudaKVIndexPair()
      : key_index_(RudaIndexEntry()), value_index_(RudaIndexEntry()) {}

  __host__ __device__
  RudaKVIndexPair(size_t key_start, size_t key_end, size_t value_start,
                  size_t value_end)
      : key_index_(RudaIndexEntry(key_start, key_end)),
        value_index_(RudaIndexEntry(value_start, value_end)) {}

  __host__ __device__
  RudaKVIndexPair(size_t value_start, size_t value_end)
      : value_index_(RudaIndexEntry(value_start, value_end)) {}

  __device__
  void pinKeyBuf(char *key_buf, size_t key_buf_size) {
    if (dh_key != nullptr) {
      delete[] dh_key;
    }
    dh_key = new char[key_buf_size];
    key_size = key_buf_size;

    memcpy(dh_key, key_buf, sizeof(char) * key_buf_size);
  }

  __device__
  void releaseKeyBuf() {
    if (dh_key != nullptr) {
      delete[] dh_key;
    }
  }

  RudaIndexEntry key_index_;
  RudaIndexEntry value_index_;

  // Store key in cuda device heap memory
  char *dh_key = nullptr;
  size_t key_size;
};

// GPU-accessable Schema
// Copy class of SlicewithSchema
class RudaSchema {
 public:
  cudaError_t populateToCuda(const rocksdb::SlicewithSchema &schema) {
    this->size = schema.size_;
    this->target_idx = schema.target_idx;
    this->ctx = schema.context;
    this->field_type_size = schema.field_type.size();
    this->field_length_size = schema.field_length.size();
    this->field_skip_size = schema.field_skip.size();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "[ERROR][RudaSchema][populateToCuda] Pre-error before calling" << std::endl;
      return err;
    }

    cudaMalloc((void **) &data, sizeof(char) * size);
    cudaMalloc((void **) &field_type, sizeof(uint) * field_type_size);
    cudaMalloc((void **) &field_length, sizeof(uint) * field_length_size);
    cudaMalloc((void **) &field_skip, sizeof(uint) * field_skip_size);

    cudaMemcpy(
        data, schema.data_, sizeof(char) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(
        field_type, &schema.field_type[0], sizeof(uint) * field_type_size,
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        field_length, &schema.field_length[0],
        sizeof(uint) * field_length_size,
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        field_skip, &schema.field_skip[0],
        sizeof(uint) * field_skip_size,
        cudaMemcpyHostToDevice);

    return cudaGetLastError();
  }

  cudaError_t clear() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "[ERROR][RudaSchema][clear] Pre-error before calling" << std::endl;
      return err;
    }
    cudaFree(data);
    cudaFree(field_type);
    cudaFree(field_length);
    return cudaGetLastError();
  }

  char *data;
  size_t size;
  accelerator::FilterContext ctx;
  int target_idx;
  uint *field_type;
  size_t field_type_size;
  uint *field_length;
  size_t field_length_size;
  uint *field_skip;
  size_t field_skip_size;
};

__host__ __device__
uint64_t DecodeFixed64(const char* ptr);

__device__
void DecodeNFilterSubDataBlocks(// Parameters
                                const char *cached_data,
                                const uint64_t cached_data_size,
                                const uint64_t block_offset,
                                const uint64_t start_idx,
                                const uint64_t end_idx,
                                accelerator::FilterContext *ctx,
                                // Results
                                unsigned long long int *results_idx,
                                RudaKVIndexPair *results);

__device__
void CachedDecodeNFilterOnSchema(// Parameters
                                 const char *cached_data,
                                 const uint64_t cached_data_size,
                                 const uint64_t block_offset,
                                 const uint64_t start_idx,
                                 const uint64_t end_idx,
                                 RudaSchema *schema,
                                 // Results
                                 unsigned long long int *results_idx,
                                 RudaKVIndexPair *results);

__device__
void DecodeNFilterOnSchema(// Parameters
                           const char *non_cached_data,
                           const uint64_t cached_data_size,
                           const uint64_t block_offset,
                           const uint64_t start_idx,
                           const uint64_t end_idx,
                           RudaSchema *schema,
                           // Results
                           unsigned long long int *results_idx,
                           RudaKVIndexPair *results);
}  // namespace ruda
