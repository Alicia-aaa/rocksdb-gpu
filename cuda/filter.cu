#include "filter.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <algorithm>

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
  cudaFree(d_results);

  results.assign(h_results, h_results + kSize);
  free(h_results);

  return ruda::RUDA_OK;
}

}  // namespace ruda
