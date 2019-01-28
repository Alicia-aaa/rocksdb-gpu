

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "cuda/block_decoder.h"
#include "cuda/filter.h"
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
                                uint64_t *seek_indices, ConditionContext *ctx,
                                uint64_t *block_seek_start_indices,
                                // Variables
                                unsigned long long int *results_idx,
                                // Results
                                ruda::RudaSlice * results_keys,
                                ruda::RudaSlice * results_values);
  __global__
  void rudaPopulateSlicesFromHeap(size_t kSize, RudaSlice *sources);
  __global__
  void rudaIntFilterKernel(ConditionContext *context, int *values,
                          int *results);
}  // namespace kernel

struct RudaIntTransformator {
  ConditionContext _context;

  RudaIntTransformator(ConditionContext context) {
    this->_context = context;
  }

  __host__ __device__
  int operator()(const int target) const {
    switch (this->_context._op) {
      case EQ:
        return target == this->_context._pivot ? 1 : 0;
      case LESS:
        return target < this->_context._pivot ? 1 : 0;
      case GREATER:
        return target > this->_context._pivot ? 1 : 0;
      case LESS_EQ:
        return target <= this->_context._pivot ? 1 : 0;
      case GREATER_EQ:
        return target >= this->_context._pivot ? 1 : 0;
      default:
        return 0;
    }
  }
};

struct RudaBlockFilterContext {
  // Parameters
  char *d_datablocks;
  uint64_t *d_seek_indices;
  ConditionContext *d_cond_context;
  uint64_t *d_block_seek_start_indices;

  // Results - Device
  unsigned long long int *d_results_idx;   // Atomic incrementer index
  RudaSlice *d_results_keys;   // Filtered keys
  RudaSlice *d_results_values; // Filtered values

  // Results - Host
  // Total results count copied from 'd_results_idx' after kernel call.
  unsigned long long int h_results_count;
  RudaSlice *h_results_keys;   // Filtered keys
  RudaSlice *h_results_values; // Filtered values

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

  size_t CalculateBlockSeekIndices(const std::vector<uint64_t> &seek_indices) {
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
                                const ConditionContext &cond_context) {
    // Cuda Parameters
    cudaCheckError(cudaMalloc(
        (void **) &d_datablocks, sizeof(char) * datablocks.size()));
    cudaCheckError(cudaMalloc(
        (void **) &d_seek_indices, sizeof(uint64_t) * kSize));
    cudaCheckError(cudaMalloc(
        (void **) &d_cond_context, sizeof(ConditionContext)));
    cudaCheckError(cudaMalloc(
        (void **) &d_block_seek_start_indices, sizeof(uint64_t) * kGridSize));
    kMaxCacheSize = CalculateBlockSeekIndices(seek_indices);

    // Cuda Results
    cudaCheckError(cudaMalloc(
        (void **) &d_results_idx, sizeof(unsigned long long int)));
    cudaCheckError(cudaMalloc(
        (void **) &d_results_keys, sizeof(RudaSlice) * kMaxResultsCount));
    cudaCheckError(cudaMalloc(
        (void **) &d_results_values, sizeof(RudaSlice) * kMaxResultsCount));

    cudaCheckError(cudaMemcpy(
        d_datablocks, &datablocks[0], sizeof(char) * datablocks.size(),
        cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(
        d_seek_indices, &seek_indices[0], sizeof(uint64_t) * kSize,
        cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(
        d_cond_context, &cond_context, sizeof(ConditionContext),
        cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(
        d_results_idx, 0, sizeof(unsigned long long int)));
  }

  void populateResultsFromCuda() {
    // Copy to host results
    cudaCheckError(cudaMemcpy(
        &h_results_count, d_results_idx, sizeof(unsigned long long int),
        cudaMemcpyDeviceToHost));
    h_results_keys = new RudaSlice[h_results_count];
    h_results_values = new RudaSlice[h_results_count];
    cudaCheckError(cudaMemcpy(
        h_results_keys, d_results_keys, sizeof(RudaSlice) * h_results_count,
        cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(
        h_results_values, d_results_values, sizeof(RudaSlice) * h_results_count,
        cudaMemcpyDeviceToHost));

    // Populates results from cuda heap space.
    for (size_t i = 0; i < h_results_count; ++i) {
      cudaCheckError(cudaMalloc(
          (void **) &h_results_keys[i].data_,
          sizeof(char) * h_results_keys[i].size()));
      cudaCheckError(cudaMalloc(
          (void **) &h_results_values[i].data_,
          sizeof(char) * h_results_values[i].size()));
    }

    cudaCheckError(cudaFree(d_results_keys));
    cudaCheckError(cudaFree(d_results_values));
    cudaCheckError(cudaMalloc(
        (void **) &d_results_keys, sizeof(RudaSlice) * h_results_count));
    cudaCheckError(cudaMalloc(
        (void **) &d_results_values, sizeof(RudaSlice) * h_results_count));

    cudaCheckError(cudaMemcpy(
        d_results_keys, h_results_keys, sizeof(RudaSlice) * h_results_count,
        cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(
        d_results_values, h_results_values, sizeof(RudaSlice) * h_results_count,
        cudaMemcpyHostToDevice));

    size_t kResultsGridSize = ceil(
        (float) h_results_count / (float) kBlockSize);
    kernel::rudaPopulateSlicesFromHeap<<<kResultsGridSize, kBlockSize>>> (
        h_results_count, d_results_keys);
    kernel::rudaPopulateSlicesFromHeap<<<kResultsGridSize, kBlockSize>>> (
        h_results_count, d_results_values);

    cudaCheckError(cudaMemcpy(
        h_results_keys, d_results_keys, sizeof(RudaSlice) * h_results_count,
        cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(
        h_results_values, d_results_values, sizeof(RudaSlice) * h_results_count,
        cudaMemcpyDeviceToHost));
  }

  void copyToFinalResults(std::vector<rocksdb::Slice> &keys,
                          std::vector<rocksdb::Slice> &values) {
    // Copy to results
    for (size_t i = 0; i < h_results_count; ++i) {
      size_t key_size = h_results_keys[i].size();
      size_t value_size = h_results_values[i].size();
      char *key = new char[key_size];
      char *value = new char[value_size];
      cudaCheckError(cudaMemcpy(
          key, h_results_keys[i].data(), key_size, cudaMemcpyDeviceToHost));
      cudaCheckError(cudaMemcpy(
          value, h_results_values[i].data(), value_size,
          cudaMemcpyDeviceToHost));
      keys.emplace_back(rocksdb::Slice(key, h_results_keys[i].size()));
      values.emplace_back(rocksdb::Slice(value, h_results_keys[i].size()));
    }
  }

  void freeParametersFromCuda() {
    cudaCheckError(cudaFree(d_datablocks));
    cudaCheckError(cudaFree(d_seek_indices));
    cudaCheckError(cudaFree(d_cond_context));
    cudaCheckError(cudaFree(d_block_seek_start_indices));
  }

  void freeResultsFromCuda() {
    cudaCheckError(cudaFree(d_results_idx));
    cudaCheckError(cudaFree(d_results_keys));
    cudaCheckError(cudaFree(d_results_values));

    // Free 2d cuda array
    for (size_t i = 0; i < h_results_count; ++i) {
      if (h_results_keys[i].size() != 0) {
        cudaCheckError(cudaFree(h_results_keys[i].data()));
      }
      if (h_results_values[i].size() != 0) {
        cudaCheckError(cudaFree(h_results_values[i].data()));
      }
    }

    delete[] h_results_keys;
    delete[] h_results_values;
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
                                      ConditionContext *ctx,
                                      uint64_t *block_seek_start_indices,
                                      // Variables
                                      unsigned long long int *results_idx,
                                      // Results
                                      ruda::RudaSlice * results_keys,
                                      ruda::RudaSlice * results_values) {
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
  DecodeSubDataBlocks(
      // Parameters
      cached_data, size, start, end, ctx,
      // Results
      results_idx, results_keys, results_values);
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

__global__
void kernel::rudaIntFilterKernel(ConditionContext *context, int *values,
                         int *results) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  switch (context->_op) {
    case EQ:
      results[index] = values[index] == context->_pivot ? 1 : 0;
      break;
    case LESS:
      results[index] = values[index] < context->_pivot ? 1 : 0;
      break;
    case GREATER:
      results[index] = values[index] > context->_pivot ? 1 : 0;
      break;
    case LESS_EQ:
      results[index] = values[index] <= context->_pivot ? 1 : 0;
      break;
    case GREATER_EQ:
      results[index] = values[index] >= context->_pivot ? 1 : 0;
      break;
    default:
      break;
  }
}

int sstIntFilter(const std::vector<int> &values,
                 const ConditionContext context,
                 std::vector<int> &results) {
  rocksdb::BlockContents block;
  // std::cout << "[RUDA][sstIntFilter] Start" << std::endl;
  results.resize(values.size());

  // std::cout << "[sstIntFilter] Inputs" << std::endl;
  // std::cout << "[sstIntFilter] Inputs - values" << std::endl;
  // for (int i = 0; i < values.size(); ++i) {
    // std::cout << values[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "[sstIntFilter] Inputs - context: " << context.toString()
      // << std::endl;

  thrust::device_vector<int> d_values(values);
  thrust::device_vector<int> d_results(values.size());

  RudaIntTransformator rudaTrans(context);
  thrust::transform(d_values.begin(), d_values.end(), d_results.begin(),
                    rudaTrans);

  // std::cout << "[sstIntFilter] Results" << std::endl;
  // std::cout << "[sstIntFilter] Results - d_results" << std::endl;
  // for (int i = 0; i < d_results.size(); ++i) {
    // std::cout << d_results[i] << " ";
  // }
  // std::cout << std::endl;

  thrust::copy(d_results.begin(), d_results.end(), results.begin());
  // std::cout << "[sstIntFilter] Results - results" << std::endl;
  // for (int i = 0; i < results.size(); ++i) {
    // std::cout << results[i] << " ";
  // }
  // std::cout << std::endl;

  return ruda::RUDA_OK;
}

int sstIntNativeFilter(const std::vector<int> &values,
                       const ConditionContext context,
                       std::vector<int> &results) {
  int *d_values, *d_results;
  int *h_results;
  ConditionContext *d_context;
  const int kSize = values.size();
  const int kBlockSize = 256;
  const float kGridSize = ceil((float) kSize / (float) kBlockSize);

  h_results = (int *) malloc(sizeof(int) * kSize);

  cudaMalloc((void **) &d_values, sizeof(int) * kSize);
  cudaMalloc((void **) &d_context, sizeof(ConditionContext));
  cudaMalloc((void **) &d_results, sizeof(int) * kSize);

  cudaMemcpy(d_values, &values[0], sizeof(int) * kSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_context, &context, sizeof(ConditionContext),
             cudaMemcpyHostToDevice);

  kernel::rudaIntFilterKernel<<<kGridSize, kBlockSize>>>(
      d_context, d_values, d_results);

  cudaMemcpy(h_results, d_results, sizeof(int) * kSize, cudaMemcpyDeviceToHost);

  cudaFree(d_values);
  cudaFree(d_context);
  cudaFree(d_results);

  results.assign(h_results, h_results + kSize);
  free(h_results);

  return ruda::RUDA_OK;
}

int sstIntBlockFilter(const std::vector<char> &datablocks,
                      const std::vector<uint64_t> &seek_indices,
                      const ConditionContext context,
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

  cudaCheckError(cudaDeviceSetLimit(
      cudaLimitMallocHeapSize, 100 * sizeof(char) * datablocks.size()));

  // Call kernel.
  kernel::rudaIntBlockFilterKernel<<<block_context.kGridSize,
                                     block_context.kBlockSize,
                                     block_context.kMaxCacheSize>>>(
      // Kernel Parameters
      block_context.kSize, datablocks.size(),
      block_context.kMaxResultsCount, block_context.d_datablocks,
      block_context.d_seek_indices, block_context.d_cond_context,
      block_context.d_block_seek_start_indices,
      // Kernel Variables
      block_context.d_results_idx,
      // Kernel Results
      block_context.d_results_keys, block_context.d_results_values);

  block_context.populateResultsFromCuda();
  block_context.copyToFinalResults(keys, values);

  std::cout << "[BlockContext::Result]" << std::endl
      << "Total Results Count: " << block_context.h_results_count << std::endl;

  // Free device variables.
  block_context.freeAllFromCuda();

  return ruda::RUDA_OK;
}

}  // namespace ruda
