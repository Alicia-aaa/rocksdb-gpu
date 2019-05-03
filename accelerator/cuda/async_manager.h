/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   async_manager.h
 * Author: wonki
 *
 * Created on March 29, 2019, 1:48 PM
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include "rocksdb/slice.h"

namespace ruda {
class RudaSchema;
class RudaKVIndexPair;
    
struct RudaAsyncBlockContext {
  cudaStream_t * stream = nullptr;
  cudaEvent_t kernel_finish_event;
  
  // Cuda Kernel Parameters
  const size_t kSize = 0;             // Total seek indices count
  const int kBlockSize = 0;
  const int kGridSize = 0;
  const size_t kMaxResultsCount = 0;  // Total count of filtered Key-Value pairs
  size_t kApproxResultsCount = 0;

  // Cuda Kernel Parameters - stream
  const int kStreamCount = 0;
  const int kStreamSize = 0;
  const int kGridSizePerStream = 0;

  // Cuda Results - Device
  RudaKVIndexPair *d_results;                  // Filtered KV pairs
  unsigned long long int *d_results_idx;  // Atomic increment counter index

  // Cuda Results - Host
  // Total results count copied from 'd_results_idx' after kernel call...
  RudaKVIndexPair *h_results;
  unsigned long long int *h_results_count;

  // Max cached datablocks size on same gpu block (For using SharedMemory)
  size_t kMaxCacheSize = 0;

  // Allocated offsets on gpu
  size_t seek_start_offset, seek_size, datablocks_start_offset, datablocks_size;

  uint64_t *d_gpu_block_seek_starts;
  uint64_t *gpu_block_seek_starts;

  // Log
  size_t total_gpu_used_memory = 0;
  
  std::mutex * mt_;
  uint64_t * kComplete_;

  RudaAsyncBlockContext(const size_t total_size, const int block_size,
                         const int grid_size, const size_t max_results_count,
                         const int stream_count, const int stream_size,
                         const int grid_size_per_stream);

  void cudaMallocGpuBlockSeekStarts();

  size_t calculateGpuBlockSeekStarts(std::vector<char> &datablocks,
                                     std::vector<uint64_t> &seek_indices,
                                     size_t start, size_t size);

  void initParams(std::vector<char> &datablocks,
                  std::vector<uint64_t> &seek_indices,
                  size_t start, size_t size, size_t start_datablocks,
                  size_t size_datablocks, std::mutex *mt);

  void populateToCuda(std::vector<char> &datablocks,
                      std::vector<uint64_t> &seek_indices,
                      char *d_datablocks, uint64_t *d_seek_indices);

  void populateToCuda_d_results_idx();
  void populateToCuda_d_datablocks(std::vector<char> &datablocks,
                                   char *d_datablocks);

  void populateToCuda_d_seek_indices(std::vector<uint64_t> &seek_indices,
                                     uint64_t *d_seek_indices);

  void populateToCuda_d_gpu_block_seek_starts();

  void executeKernel(// Kernel Parameter
                     size_t kTotalDataSize,
                     // Sources
                     char *d_datablocks, uint64_t *d_seek_indices,
                     RudaSchema *d_schema);

  void copyFromCuda(uint64_t * kComplete);

  void freeCudaObjects();
  
  void populateStream(cudaStream_t * allocated_stream);

  void initializeStream();

  void destroyStream();

  void clear();

};
    
struct RudaAsyncManager {
  // Cuda Kernel Parameters
  // IMPORTANT: Kernel Parameters never be changed except in constructor.
  size_t kSize = 0;             // Total seek indices count
  int kBlockSize = 0;
  int kGridSize = 0;
  size_t kMaxResultsCount = 0;  // Total count of filtered Key-Value pairs

  // Cuda Kernel Parameters - stream
  int kStreamCount = 0;
  int kApproxStreamSize = 0;
  int kApproxGridSizePerStream = 0;

  // Streams
  std::vector<RudaAsyncBlockContext> stream_ctxs;
  std::mutex *ml_;

  std::vector<char> * h_datablocks;
  std::vector<uint64_t> * h_seek_indices;
  // Parameters
  char *d_datablocks;
  uint64_t *d_seek_indices;

  RudaSchema *d_schema_;
  uint64_t kComplete = 0;
  int join_idx_ = 0;

  // Log
  size_t total_gpu_used_memory = 0;
  
  cudaStream_t * cuda_stream_;
  
  RudaAsyncManager(RudaSchema * d_schema, std::mutex * ml, cudaStream_t * cuda_stream);

  void calStreamContext(const size_t total_size, const int block_size,
                         const size_t stream_count,
                         const size_t max_results_count);

  void registerPinnedMemory(std::vector<char> &datablocks,
                            std::vector<uint64_t> &seek_indices);

  void unregisterPinnedMemory(std::vector<char> &datablocks,
                              std::vector<uint64_t> &seek_indices);
  
  void initParams(std::vector<char> &datablocks,
                  std::vector<uint64_t> &seek_indices, int join_idx);

  void populateToCuda(std::vector<char> &datablocks,
                      std::vector<uint64_t> &seek_indices);

  void executeKernels(size_t kTotalDataSize, int join_idx);

  void copyFromCuda();

  void _translatePairsToSlices(RudaAsyncBlockContext &ctx,
                               std::vector<char> &datablocks,
                               std::vector<rocksdb::Slice> &sub_values);

  void translatePairsToSlices(std::vector<char> &datablocks,
                              std::vector<rocksdb::PinnableSlice> &values);
  void log();

  void clear();
  void release();
};

}  // namespace ruda
