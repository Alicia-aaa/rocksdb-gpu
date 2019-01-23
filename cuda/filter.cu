

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "filter.h"
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

  // Results
  char *d_results;
  // rocksdb::Slice *d_results;
  size_t *d_results_count;

  // Cuda Kernel Parameters
  const int kSize;
  const int kBlockSize;
  const float kGridSize;

  RudaBlockFilterContext() : kSize(0), kBlockSize(0), kGridSize(0) {}

  RudaBlockFilterContext(const int total_size, const int block_size)
      : kSize(total_size), kBlockSize(block_size),
        kGridSize(ceil((float) total_size / (float) block_size)) {}

  void populateParametersToCuda(const std::vector<char> &datablocks,
                                const std::vector<uint64_t> &seek_indices,
                                const ConditionContext &cond_context) {
    // Cuda Parameters
    cudaMalloc((void **) &d_datablocks, sizeof(char) * datablocks.size());
    cudaMalloc((void **) &d_seek_indices, sizeof(uint64_t) * kSize);
    cudaMalloc((void **) &d_cond_context, sizeof(ConditionContext));

    // Cuda Results
    cudaMalloc((void **) &d_results_count, sizeof(size_t));
    // Note(totoro): 'd_results' will be created on Kernel.

    cudaMemcpy(
        d_datablocks, &datablocks[0], sizeof(char) * datablocks.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_seek_indices, &seek_indices[0], sizeof(uint64_t) * kSize,
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_cond_context, &cond_context, sizeof(ConditionContext),
        cudaMemcpyHostToDevice);
  }

  void freeParametersFromCuda() {
    cudaFree(d_datablocks);
    cudaFree(d_seek_indices);
    cudaFree(d_cond_context);
  }

  void freeResultsFromCuda() {
    cudaFree(d_results_count);
    cudaFree(d_results);
  }

  void freeAllFromCuda() {
    freeParametersFromCuda();
    freeResultsFromCuda();
  }
};

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

__global__
void rudaIntBlockFilterKernel(// Parameters
                              size_t kSize, size_t dataSize,
                              char *data, uint64_t *seek_indices,
                              ConditionContext *ctx,
                              // Results
                              char ** /* results */,
                              size_t * /* results_count */) {
  uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

  // Overflow kernel ptr case.
  if (i >= kSize) {
    return;
  }

  // Shared variables.
  // Caches data used from threads in single block.
  __shared__ char *cached_data;
  __shared__ char *cached_data_end;
  if (threadIdx.x == 0) {
    cached_data = data + seek_indices[i];
  }

  if (
    threadIdx.x == (blockDim.x - 1) ||
    (
      blockIdx.x == ((kSize / blockDim.x)) &&
      threadIdx.x == ((kSize % blockDim.x) - 1)
    )
  ) {
    if (i == kSize - 1) {
      // Last seek index case. 'end' must be end of data.
      cached_data_end = data + dataSize;
    } else {
      // 'end' must be next seek index.
      cached_data_end = data + seek_indices[i + 1];
    }
  }
  __syncthreads();

  size_t cached_data_length = static_cast<size_t>(
      cached_data_end - cached_data);
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
                      std::vector<rocksdb::Slice> &results) {
  RudaBlockFilterContext block_context(
      seek_indices.size() /* kSize */,
      4 /* kBlockSize */);

  // Copy & Initializes variables from host to device.
  block_context.populateParametersToCuda(datablocks, seek_indices, context);

  // Call kernel.
  rudaIntBlockFilterKernel<<<block_context.kGridSize,
                             block_context.kBlockSize>>>(
      // Kernel Parameters
      static_cast<size_t>(block_context.kSize), datablocks.size(),
      block_context.d_datablocks, block_context.d_seek_indices,
      block_context.d_cond_context,
      // Kernel Results
      &block_context.d_results, block_context.d_results_count);

  // Free device variables.
  block_context.freeAllFromCuda();

  return ruda::RUDA_OK;
}

}  // namespace ruda
