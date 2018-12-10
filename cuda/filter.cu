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

  struct CudaIntComparator {
    IntComparator _rawComp;

    CudaIntComparator(IntComparator rawComp) {
      this->_rawComp = rawComp;
    }

    __host__ __device__
    bool operator()(const int target) {
      switch (this->_rawComp._op) {
        case EQ:
          return target == this->_rawComp._pivot;
        case LESS:
          return target < this->_rawComp._pivot;
        case GREATER:
          return target > this->_rawComp._pivot;
        case LESS_EQ:
          return target <= this->_rawComp._pivot;
        case GREATER_EQ:
          return target >= this->_rawComp._pivot;
        default:
          return false;
      }
    }

    std::string toString() {
      return this->_rawComp.toString();
    }
  };

  int sstIntFilter(const std::vector<int>& values,
                   const IntComparator rawComp,
                   std::vector<bool>& results) {
    std::cout << "[RUDA][sstIntFilter] Start" << std::endl;
    results.resize(values.size());

    std::cout << "[sstIntFilter] Inputs" << std::endl;
    std::cout << "[sstIntFilter] Inputs - values" << std::endl;
    for (int i = 0; i < values.size(); ++i) {
      std::cout << values[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "[sstIntFilter] Inputs - rawComp: " << rawComp.toString()
        << std::endl;

    thrust::device_vector<int> d_values(values);
    thrust::device_vector<int> d_results(values.size());

    CudaIntComparator cudaComp(rawComp);

    std::cout << "[sstIntFilter] Devices" << std::endl;
    std::cout << "[sstIntFilter] Devices - d_values" << std::endl;
    for (int i = 0; i < d_values.size(); ++i) {
      std::cout << d_values[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "[sstIntFilter] cudaComp: " << cudaComp.toString() << std::endl;

    thrust::copy_if(d_values.begin(), d_values.end(), d_results.begin(),
                    cudaComp);
    cudaDeviceSynchronize();

    std::cout << "[sstIntFilter] Results" << std::endl;
    std::cout << "[sstIntFilter] Results - d_results" << std::endl;
    for (int i = 0; i < d_results.size(); ++i) {
      std::cout << d_results[i] << " ";
    }
    std::cout << std::endl;

    thrust::copy(d_results.begin(), d_results.end(), results.begin());
    std::cout << "[sstIntFilter] Results - results" << std::endl;
    for (int i = 0; i < results.size(); ++i) {
      std::cout << results[i] << " ";
    }
    std::cout << std::endl;

    return ruda::RUDA_OK;
  }
}

