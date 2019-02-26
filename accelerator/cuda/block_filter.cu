
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "accelerator/cuda/block_decoder.h"
#include "accelerator/cuda/filter.h"
#include "rocksdb/slice.h"
#include "table/format.h"

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

namespace ruda {
namespace kernel {

// Kernels
__global__
void rudaIntBlockFilterKernel(// Parameters (ReadOnly)
                              size_t kSize, size_t dataSize,
                              size_t resultsCount, char *data,
                              uint64_t *seek_indices,
                              accelerator::FilterContext *ctx,
                              uint64_t *block_seek_start_indices,
                              // Variables
                              unsigned long long int *results_idx,
                              // Results
                              RudaKVIndexPair *results);
__global__
void rudaPopulateSlicesFromHeap(size_t kSize, RudaSlice *sources);

}  // namespace kernel

struct RudaBlockFilterContext {
  // Parameters
  char *d_datablocks;
  uint64_t *d_seek_indices;
  accelerator::FilterContext *d_cond_ctx;
  uint64_t *d_block_seek_start_indices;

  // Results - Device
  unsigned long long int *d_results_idx;   // Atomic incrementer index
  RudaKVIndexPair *d_results;    // Filtered KV pairs

  // Results - Host
  // Total results count copied from 'd_results_idx' after kernel call.
  unsigned long long int h_results_count;
  RudaKVIndexPair *h_results;    // Filtered KV pairs

  // Cuda Kernel Parameters
  const size_t kSize = 0;
  const int kBlockSize = 0;
  const int kGridSize = 0;
  const size_t kMaxResultsCount = 0;
  size_t kMaxCacheSize = 0;

  RudaBlockFilterContext(const size_t total_size, const int block_size,
                         const size_t max_results_count)
      : kSize(total_size), kBlockSize(block_size),
        kGridSize(ceil((float) total_size / (float) block_size)),
        kMaxResultsCount(max_results_count) {}

  size_t calculateBlockSeekIndices(const std::vector<uint64_t> &seek_indices) {
    uint64_t block_seek_start_indices[kGridSize];

    for (size_t i = 0; i < kGridSize; ++i) {
      size_t i_thread = i * kBlockSize;
      block_seek_start_indices[i] = seek_indices[i_thread];
    }

    size_t max_cache_size = 0;
    for (size_t i = 0; i < kGridSize; ++i) {
      if (i == kGridSize - 1) {
        break;
      }
      size_t cache_size =
          block_seek_start_indices[i + 1] - block_seek_start_indices[i];
      if (cache_size > max_cache_size) {
        max_cache_size = cache_size;
      }
    }

    cudaCheckError(cudaMemcpy(
        d_block_seek_start_indices, block_seek_start_indices,
        sizeof(uint64_t) * kGridSize, cudaMemcpyHostToDevice));
    return sizeof(char) * max_cache_size;
  }

  void populateParametersToCuda(const std::vector<char> &datablocks,
                                const std::vector<uint64_t> &seek_indices,
                                const accelerator::FilterContext &cond_ctx) {
    // Register as a Pinned Memory
    // cudaHostRegister(&datablocks[0], sizeof(char) * datablocks.size());
    // cudaHostRegister(&seek_indices[0], sizeof(uint64_t) * seek_indices.size());
    // cudaHostRegister(&cond_ctx, sizeof(accelerator::FilterContext));

    // Cuda Parameters
    cudaCheckError(cudaMalloc(
        (void **) &d_datablocks, sizeof(char) * datablocks.size()));
    cudaCheckError(cudaMalloc(
        (void **) &d_seek_indices, sizeof(uint64_t) * kSize));
    cudaCheckError(cudaMalloc(
        (void **) &d_cond_ctx, sizeof(accelerator::FilterContext)));
    cudaCheckError(cudaMalloc(
        (void **) &d_block_seek_start_indices, sizeof(uint64_t) * kGridSize));
    kMaxCacheSize = calculateBlockSeekIndices(seek_indices);

    // Cuda Results
    cudaCheckError(cudaMalloc(
        (void **) &d_results_idx, sizeof(unsigned long long int)));
    cudaCheckError(cudaMalloc(
        (void **) &d_results, sizeof(RudaKVIndexPair) * kMaxResultsCount));

    cudaCheckError(cudaMemcpy(
        d_datablocks, &datablocks[0], sizeof(char) * datablocks.size(),
        cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(
        d_seek_indices, &seek_indices[0], sizeof(uint64_t) * kSize,
        cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(
        d_cond_ctx, &cond_ctx, sizeof(accelerator::FilterContext),
        cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(
        d_results_idx, 0, sizeof(unsigned long long int)));

    // Unregister as a Pinned Memory
    // cudaHostUnregister(&datablocks[0]);
    // cudaHostUnregister(&seek_indices[0]);
    // cudaHostUnregister(&cond_ctx);
  }

  void populateResultsFromCuda() {
    // Copy to host results
    cudaCheckError(cudaMemcpy(
        &h_results_count, d_results_idx, sizeof(unsigned long long int),
        cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMallocHost(
        (void **) &h_results, sizeof(RudaKVIndexPair) * h_results_count));
    cudaCheckError(cudaMemcpy(
        h_results, d_results, sizeof(RudaKVIndexPair) * h_results_count,
        cudaMemcpyDeviceToHost));
  }

  void copyToFinalResults(const std::vector<char> &datablocks,
                          std::vector<rocksdb::Slice> &keys,
                          std::vector<rocksdb::Slice> &values) {
    // Copy to results
    for (size_t i = 0; i < h_results_count; ++i) {
      RudaKVIndexPair &entry = h_results[i];
      size_t key_size = entry.key_index_.end_ - entry.key_index_.start_;
      size_t value_size = entry.value_index_.end_ - entry.value_index_.start_;
      char *key = new char[key_size];
      char *value = new char[value_size];
      memcpy(
          key, &datablocks[0] + entry.key_index_.start_,
          sizeof(char) * key_size);
      memcpy(
          value, &datablocks[0] + entry.value_index_.end_,
          sizeof(char) * value_size);
      keys.emplace_back(key, key_size);
      values.emplace_back(value, value_size);
    }
  }

  void freeParametersFromCuda() {
    cudaCheckError(cudaFree(d_datablocks));
    cudaCheckError(cudaFree(d_seek_indices));
    cudaCheckError(cudaFree(d_cond_ctx));
    cudaCheckError(cudaFree(d_block_seek_start_indices));
  }

  void freeResultsFromCuda() {
    cudaCheckError(cudaFree(d_results_idx));
    cudaCheckError(cudaFree(d_results));
    cudaCheckError(cudaFreeHost(h_results));
  }

  void freeAllFromCuda() {
    freeParametersFromCuda();
    freeResultsFromCuda();
  }
};

__global__
void kernel::rudaIntBlockFilterKernel(// Parameters (ReadOnly)
                                      size_t kSize, size_t dataSize,
                                      size_t resultsCount, char *data,
                                      uint64_t *seek_indices,
                                      accelerator::FilterContext *ctx,
                                      uint64_t *block_seek_start_indices,
                                      // Variables
                                      unsigned long long int *results_idx,
                                      // Results
                                      RudaKVIndexPair *results) {
  uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

  // Overflow kernel ptr case.
  if (i >= kSize) {
    return;
  }

  // Shared variables.
  // Caches data used from threads in single block.
  extern __shared__ char cached_data[];

  uint64_t block_seek_start_index = block_seek_start_indices[blockIdx.x];
  uint64_t start = seek_indices[i] - block_seek_start_index;
  uint64_t end = 0;
  if (i == (kSize - 1)) {
    // Last seek index case. 'end' must be end of data.
    end = dataSize - block_seek_start_index;
  } else {
    // 'end' must be next seek index.
    end = seek_indices[i + 1] - block_seek_start_index;
  }

  for (size_t j = start; j < end; ++j) {
    size_t data_idx = block_seek_start_index + j;
    cached_data[j] = data[data_idx];
  }

  __syncthreads();

  size_t size = end - start;
  DecodeNFilterSubDataBlocks(
      // Parameters
      cached_data, size, block_seek_start_index, start, end, ctx,
      // Results
      results_idx, results);
}

__global__
void kernel::rudaPopulateSlicesFromHeap(size_t kSize, RudaSlice *sources) {
  uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= kSize || sources[i].data() == nullptr) {
    return;
  }

  sources[i].populateDataFromHeap();
  delete sources[i].heap_data_;
}

int sstIntBlockFilter(const std::vector<char> &datablocks,
                      const std::vector<uint64_t> &seek_indices,
                      const accelerator::FilterContext context,
                      const size_t max_results_count,
                      std::vector<rocksdb::Slice> &keys,
                      std::vector<rocksdb::Slice> &values) {
  RudaBlockFilterContext block_context(
      seek_indices.size() /* kSize */,
      64 /* kBlockSize */,
      max_results_count);

  // Copy & Initializes variables from host to device.
  block_context.populateParametersToCuda(datablocks, seek_indices, context);

  std::cout
      << "[BlockContext]" << std::endl
      << "kSize: " << block_context.kSize << std::endl
      << "kGridSize: " << block_context.kGridSize << std::endl
      << "kBlockSize: " << block_context.kBlockSize << std::endl
      << "kMaxCacheSize: " << block_context.kMaxCacheSize << std::endl
      << "DataSize: " << datablocks.size() << std::endl
      << "Max Results Count: " << block_context.kMaxResultsCount << std::endl;

  // cudaCheckError(cudaDeviceSetLimit(
  //     cudaLimitMallocHeapSize, 100 * sizeof(char) * datablocks.size()));

  // Call kernel.
  kernel::rudaIntBlockFilterKernel<<<block_context.kGridSize,
                                     block_context.kBlockSize,
                                     block_context.kMaxCacheSize>>>(
      // Kernel Parameters
      block_context.kSize, datablocks.size(),
      block_context.kMaxResultsCount, block_context.d_datablocks,
      block_context.d_seek_indices, block_context.d_cond_ctx,
      block_context.d_block_seek_start_indices,
      // Kernel Variables
      block_context.d_results_idx,
      // Kernel Results
      block_context.d_results);

  block_context.populateResultsFromCuda();
  block_context.copyToFinalResults(datablocks, keys, values);

  std::cout << "Total Results Count: " << block_context.h_results_count
      << std::endl;

  // Free device variables.
  block_context.freeAllFromCuda();

  return accelerator::ACC_OK;
}

}  // namespace ruda
