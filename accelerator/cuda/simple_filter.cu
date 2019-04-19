
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "accelerator/cuda/filter.h"

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

__global__
void rudaIntFilterKernel(accelerator::FilterContext *context, int *values,
                         int *results);

}  // namespace kernel

struct RudaIntFunctor {
  accelerator::FilterContext _context;

  RudaIntFunctor(accelerator::FilterContext context) {
    this->_context = context;
  }

  __host__ __device__
  int operator()(const long target) const {
    switch (this->_context._op) {
      case accelerator::EQ:
        return target == this->_context._pivot ? 1 : 0;
      case accelerator::LESS:
        return target < this->_context._pivot ? 1 : 0;
      case accelerator::GREATER:
        return target > this->_context._pivot ? 1 : 0;
      case accelerator::LESS_EQ:
        return target <= this->_context._pivot ? 1 : 0;
      case accelerator::GREATER_EQ:
        return target >= this->_context._pivot ? 1 : 0;
      default:
        return 0;
    }
  }
};

__global__
void kernel::rudaIntFilterKernel(accelerator::FilterContext *context, int *values,
                                 int *results) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  switch (context->_op) {
    case accelerator::EQ:
      results[index] = values[index] == context->_pivot ? 1 : 0;
      break;
    case accelerator::LESS:
      results[index] = values[index] < context->_pivot ? 1 : 0;
      break;
    case accelerator::GREATER:
      results[index] = values[index] > context->_pivot ? 1 : 0;
      break;
    case accelerator::LESS_EQ:
      results[index] = values[index] <= context->_pivot ? 1 : 0;
      break;
    case accelerator::GREATER_EQ:
      results[index] = values[index] >= context->_pivot ? 1 : 0;
      break;
    default:
      break;
  }
}

int gpuWarmingUp() {
  // Warming up
  // Note(totoro): Because, there is a warming up latency on gpu when
  // gpu-related function called(ex. set up gpu driver). So, we ignore this
  // latency by just firing meaningless malloc function.
  void *warming_up;
  cudaCheckError(cudaMalloc(&warming_up, 0));
  cudaCheckError(cudaFree(warming_up));

  return accelerator::ACC_OK;
}

int sstThrustFilter(const std::vector<long> &values,
                    const accelerator::FilterContext context,
                    std::vector<long> &results) {
  // std::cout << "[RUDA][sstThrustFilter] Start" << std::endl;
  results.resize(values.size());

  // std::cout << "[sstThrustFilter] Inputs" << std::endl;
  // std::cout << "[sstThrustFilter] Inputs - values" << std::endl;
  // for (int i = 0; i < values.size(); ++i) {
  // std::cout << values[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "[sstThrustFilter] Inputs - context: " << context.toString()
  // << std::endl;

  thrust::device_vector<long> d_values(values);
  thrust::device_vector<long> d_results(values.size());

  RudaIntFunctor rudaFunc(context);
  thrust::transform(d_values.begin(), d_values.end(), d_results.begin(),
                    rudaFunc);

  // std::cout << "[sstThrustFilter] Results" << std::endl;
  // std::cout << "[sstThrustFilter] Results - d_results" << std::endl;
  // for (int i = 0; i < d_results.size(); ++i) {
  // std::cout << d_results[i] << " ";
  // }
  // std::cout << std::endl;

  thrust::copy(d_results.begin(), d_results.end(), results.begin());
  // std::cout << "[sstThrustFilter] Results - results" << std::endl;
  // for (int i = 0; i < results.size(); ++i) {
  // std::cout << results[i] << " ";
  // }
  // std::cout << std::endl;

  return accelerator::ACC_OK;
}

int sstIntNativeFilter(const std::vector<int> &values,
                       const accelerator::FilterContext context,
                       std::vector<int> &results) {
  int *d_values, *d_results;
  int *h_results;
  accelerator::FilterContext *d_context;
  const int kSize = values.size();
  const int kBlockSize = 256;
  const float kGridSize = ceil((float) kSize / (float) kBlockSize);

  h_results = (int *) malloc(sizeof(int) * kSize);

  cudaMalloc((void **) &d_values, sizeof(int) * kSize);
  cudaMalloc((void **) &d_context, sizeof(accelerator::FilterContext));
  cudaMalloc((void **) &d_results, sizeof(int) * kSize);

  cudaMemcpy(d_values, &values[0], sizeof(int) * kSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_context, &context, sizeof(accelerator::FilterContext),
             cudaMemcpyHostToDevice);

  kernel::rudaIntFilterKernel<<<kGridSize, kBlockSize>>>(
      d_context, d_values, d_results);

  cudaMemcpy(h_results, d_results, sizeof(int) * kSize, cudaMemcpyDeviceToHost);

  cudaFree(d_values);
  cudaFree(d_context);
  cudaFree(d_results);

  results.assign(h_results, h_results + kSize);
  free(h_results);

  return accelerator::ACC_OK;
}

}  // namespace ruda
