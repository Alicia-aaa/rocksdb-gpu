

#include <algorithm>
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

namespace ruda {

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

  // Results
  int *d_results_idx;   // Atomic incrementer index
  RudaSlice *d_results_keys;   // Filtered keys
  RudaSlice *d_results_values; // Filtered values

  // Test
  char *d_data;
  RudaSlice *d_subdata;
  uint64_t *d_starts;
  uint64_t *d_ends;

  // Cuda Kernel Parameters
  const size_t kSize = 0;
  const int kBlockSize = 0;
  const int kGridSize = 0;
  const size_t kResultsCount = 0;
  size_t kMaxCacheSize = 0;

  RudaBlockFilterContext(const size_t total_size, const int block_size,
                         const size_t results_count)
      : kSize(total_size), kBlockSize(block_size),
        kGridSize(ceil((float) total_size / (float) block_size)),
        kResultsCount(results_count) {}

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

    cudaMemcpy(
        d_block_seek_start_indices, block_seek_start_indices,
        sizeof(uint64_t) * kGridSize, cudaMemcpyHostToDevice);
    return sizeof(char) * max_cache_size;
  }

  void populateParametersToCuda(const std::vector<char> &datablocks,
                                const std::vector<uint64_t> &seek_indices,
                                const ConditionContext &cond_context) {
    // Cuda Parameters
    cudaMalloc((void **) &d_datablocks, sizeof(char) * datablocks.size());
    cudaMalloc((void **) &d_seek_indices, sizeof(uint64_t) * kSize);
    cudaMalloc((void **) &d_cond_context, sizeof(ConditionContext));
    cudaMalloc(
        (void **) &d_block_seek_start_indices, sizeof(uint64_t) * kGridSize);
    kMaxCacheSize = CalculateBlockSeekIndices(seek_indices);

    // Cuda Results
    cudaMalloc((void **) &d_results_idx, sizeof(int));
    cudaMalloc((void **) &d_results_keys, sizeof(RudaSlice) * kResultsCount);
    cudaMalloc((void **) &d_results_values, sizeof(RudaSlice) * kResultsCount);

    // Test
    cudaMalloc((void **) &d_data, sizeof(char) * datablocks.size());
    cudaMalloc((void **) &d_subdata, sizeof(RudaSlice) * kSize);
    cudaMalloc((void **) &d_starts, sizeof(uint64_t) * kSize);
    cudaMalloc((void **) &d_ends, sizeof(uint64_t) * kSize);

    cudaMemcpy(
        d_datablocks, &datablocks[0], sizeof(char) * datablocks.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_seek_indices, &seek_indices[0], sizeof(uint64_t) * kSize,
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_cond_context, &cond_context, sizeof(ConditionContext),
        cudaMemcpyHostToDevice);
    cudaMemset(d_results_idx, 0, sizeof(int));
  }

  void freeParametersFromCuda() {
    cudaFree(d_datablocks);
    cudaFree(d_seek_indices);
    cudaFree(d_cond_context);
    cudaFree(d_block_seek_start_indices);
  }

  void freeResultsFromCuda() {
    cudaFree(d_results_idx);

    // Free 2d cuda array
    RudaSlice *h_results_keys = new RudaSlice[kResultsCount];
    RudaSlice *h_results_values = new RudaSlice[kResultsCount];
    cudaMemcpy(
        h_results_keys, d_results_keys, sizeof(RudaSlice) * kResultsCount,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        h_results_values, d_results_values, sizeof(RudaSlice) * kResultsCount,
        cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < kResultsCount; ++i) {
      if (h_results_keys[i].size() != 0) {
        cudaFree(h_results_keys[i].data());
      }
      if (h_results_values[i].size() != 0) {
        cudaFree(h_results_values[i].data());
      }
    }

    RudaSlice *h_subdata = new RudaSlice[kSize];
    cudaMemcpy(
      h_subdata, d_subdata, sizeof(RudaSlice) * kSize,
      cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < kSize; ++i) {
      if (h_subdata[i].size() != 0) {
        cudaFree(h_subdata[i].data());
      }
    }

    cudaFree(d_results_keys);
    cudaFree(d_results_values);
    cudaFree(d_subdata);
    cudaFree(d_starts);
    cudaFree(d_ends);

    delete[] h_results_keys;
    delete[] h_results_values;
    delete[] h_subdata;
  }

  void freeAllFromCuda() {
    freeParametersFromCuda();
    freeResultsFromCuda();
  }
};

__global__
void rudaIntBlockFilterKernel(// Parameters (ReadOnly)
                              size_t kSize, size_t dataSize,
                              size_t resultsCount, char *data,
                              uint64_t *seek_indices, ConditionContext *ctx,
                              uint64_t *block_seek_start_indices,
                              // Variables
                              int *results_idx,
                              // Results
                              ruda::RudaSlice * /*results_keys*/,
                              ruda::RudaSlice * /*results_values*/,
                              // Test
                              char *result_data,
                              ruda::RudaSlice *subdata,
                              uint64_t *starts,
                              uint64_t *ends) {
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
    result_data[data_idx] = data[data_idx];
  }

  starts[i] = start + block_seek_start_index;
  ends[i] = end + block_seek_start_index;

  __syncthreads();

  size_t size = end - start;
  char *subdata_thread = new char[size];
  for (size_t j = 0; j < size; ++j) {
    subdata_thread[j] = cached_data[j + start];
  }
  subdata[i] = RudaSlice(subdata_thread, size);

  // DecodeSubDataBlocks(
  //     // Parameters
  //     cached_data, cached_data_size, start, end,
  //     // Results
  //     results_idx, results_keys, results_values);
}

__global__
void rudaIntFilterKernel(ConditionContext *context, int *values,
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

  rudaIntFilterKernel<<<kGridSize, kBlockSize>>>(
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
                      const size_t results_count,
                      std::vector<rocksdb::Slice> &keys,
                      std::vector<rocksdb::Slice> &values) {
  RudaBlockFilterContext block_context(
      seek_indices.size() /* kSize */,
      4 /* kBlockSize */,
      results_count);

  // Copy & Initializes variables from host to device.
  block_context.populateParametersToCuda(datablocks, seek_indices, context);

  std::cout
      << "[BlockContext]" << std::endl
      << "kSize: " << block_context.kSize << std::endl
      << "kGridSize: " << block_context.kGridSize << std::endl
      << "kBlockSize: " << block_context.kBlockSize << std::endl
      << "kMaxCacheSize: " << block_context.kMaxCacheSize << std::endl
      << "DataSize: " << datablocks.size() << std::endl
      << "Results Count: " << block_context.kResultsCount << std::endl;

  // Call kernel.
  rudaIntBlockFilterKernel<<<block_context.kGridSize,
                             block_context.kBlockSize,
                             block_context.kMaxCacheSize>>>(
      // Kernel Parameters
      block_context.kSize, datablocks.size(),
      block_context.kResultsCount, block_context.d_datablocks,
      block_context.d_seek_indices, block_context.d_cond_context,
      block_context.d_block_seek_start_indices,
      // Kernel Variables
      block_context.d_results_idx,
      // Kernel Results
      block_context.d_results_keys, block_context.d_results_values,
      block_context.d_data,
      block_context.d_subdata, block_context.d_starts, block_context.d_ends);

  // Copy to host results
  // int h_results_idx;
  // cudaMemcpy(
  //     &h_results_idx, block_context.d_results_idx, sizeof(int),
  //     cudaMemcpyDeviceToHost);
  // RudaSlice *h_results_keys = new RudaSlice[h_results_idx];
  // RudaSlice *h_results_values = new RudaSlice[h_results_idx];
  // cudaMemcpy(
  //     h_results_keys, block_context.d_results_keys,
  //     sizeof(RudaSlice) * h_results_idx, cudaMemcpyDeviceToHost);
  // cudaMemcpy(
  //     h_results_values, block_context.d_results_values,
  //     sizeof(RudaSlice) * h_results_idx, cudaMemcpyDeviceToHost);

  // Test
  char h_data[datablocks.size()];
  cudaMemcpy(
      h_data, block_context.d_data, sizeof(char) * datablocks.size(),
      cudaMemcpyDeviceToHost);
  std::cout << "<<<<<<<<<<< Data comparison" << std::endl;
  std::cout << "datablocks" << std::endl;
  for (size_t i = 0; i < datablocks.size(); ++i) {
    std::cout << datablocks[i];
  }
  std::cout << std::endl;
  std::cout << "h_data" << std::endl;
  for (size_t i = 0; i < datablocks.size(); ++i) {
    std::cout << h_data[i];
  }
  std::cout << std::endl << std::endl;

  RudaSlice h_subdata[block_context.kSize];
  cudaMemcpy(
      h_subdata, block_context.d_subdata,
      sizeof(RudaSlice) * block_context.kSize, cudaMemcpyDeviceToHost);
  uint64_t h_starts[block_context.kSize];
  cudaMemcpy(
      h_starts, block_context.d_starts, sizeof(uint64_t) * block_context.kSize,
      cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < block_context.kSize; ++i) {
    size_t size = h_subdata[i].size();
    char subdata[size];
    char original[size];
    cudaMemcpy(
        subdata, h_subdata[i].data(), size, cudaMemcpyDeviceToHost);
    memcpy(original, &datablocks[0] + h_starts[i], size);
    std::cout << "Subdata[Size: " << size << ", Pointer: " << (void *) h_subdata[i].data() << "]" << std::endl;
    for (size_t j = 0; j < size; ++j) {
      std::cout << subdata[j];
    }
    std::cout << std::endl;
    std::cout << "Original[Size: " << size << "]" << std::endl;
    for (size_t j = 0; j < size; ++j) {
      std::cout << original[j];
    }
    std::cout << std::endl;
    std::cout << "Cmp result: " << strcmp(subdata, original) << std::endl;
  }

  uint64_t h_ends[block_context.kSize];
  cudaMemcpy(
      h_ends, block_context.d_ends, sizeof(uint64_t) * block_context.kSize,
      cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < block_context.kSize; ++i) {
    std::cout << "Index[" << i
        << "], Start[" << h_starts[i]
        << "], End[" << h_ends[i]
        << "]" << std::endl;
  }

  // for (size_t i = 0; i < h_results_idx; ++i) {
  //   size_t key_size = h_results_keys[i].size();
  //   size_t value_size = h_results_values[i].size();
  //   char *key = new char[key_size];
  //   char *value = new char[value_size];
  //   cudaMemcpy(
  //       key, h_results_keys[i].data(), key_size, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(
  //       value, h_results_values[i].data(), value_size, cudaMemcpyDeviceToHost);
  //   std::cout << "Key[Size: " << key_size << ", ";
  //   for (size_t j = 0; j < key_size; ++j) {
  //     std::cout << key[j];
  //   }
  //   std::cout << "] Value[Size: " << value_size << ", ";
  //   for (size_t j = 0; j < value_size; ++j) {
  //     std::cout << value[j];
  //   }
  //   std::cout << "]" << std::endl;
  //   keys.emplace_back(rocksdb::Slice(key, h_results_keys[i].size()));
  //   values.emplace_back(rocksdb::Slice(value, h_results_keys[i].size()));
  // }

  // Free device variables.
  block_context.freeAllFromCuda();

  return ruda::RUDA_OK;
}

}  // namespace ruda
