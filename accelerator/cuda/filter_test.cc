
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "accelerator/common.h"
#include "accelerator/cuda/filter.h"

int createRandomNumbers(std::vector<int> &values, const uint64_t kCount,
                        const int kMin, const int kMax) {
  std::random_device rd;
  std::mt19937 generator(rd());

  std::uniform_int_distribution<> uintdist(kMin, kMax);

  std::generate_n(
    std::back_inserter(values),
    kCount,
    [&uintdist, &generator]() {
      return uintdist(generator);
    }
  );

  return accelerator::ACC_OK;
}

void runFilter(accelerator::FilterContext &ctx, std::vector<int> &values,
               std::vector<int> &results) {
  results.clear();

  std::cout << "[FILTER_TEST] Run SST Filter" << std::endl;
  ruda::sstThrustFilter(values, ctx, results);

  // std::cout << "[FILTER_TEST] Results" << std::endl;
  // for (int result : results) {
  //   std::cout << result << " ";
  // }
  // std::cout << std::endl;
}

void runNativeFilter(accelerator::FilterContext &ctx, std::vector<int> &values,
                     std::vector<int> &results) {
  results.clear();

  std::cout << "[FILTER_TEST] Run SST Native Filter" << std::endl;
  ruda::sstIntNativeFilter(values, ctx, results);

  // std::cout << "[FILTER_TEST] Results" << std::endl;
  // for (int result : results) {
  //   std::cout << result << " ";
  // }
  // std::cout << std::endl;
}

void runCpuFilter(accelerator::FilterContext &ctx, std::vector<int> &values,
                  std::vector<int> &results) {
  results.clear();
  std::transform(
    values.begin(), values.end(), std::back_inserter(results),
    [&ctx](int value) -> int {
      switch (ctx._op) {
        case ruda::EQ:
          return value == ctx._pivot ? 1 : 0;
        case ruda::LESS:
          return value < ctx._pivot ? 1 : 0;
        case ruda::GREATER:
          return value > ctx._pivot ? 1 : 0;
        case ruda::LESS_EQ:
          return value <= ctx._pivot ? 1 : 0;
        case ruda::GREATER_EQ:
          return value >= ctx._pivot ? 1 : 0;
        default:
          return -1;
      }
    }
  );
}

int main() {
  std::cout << "[FILTER_TEST] Starts" << std::endl;
  accelerator::FilterContext ctx = {
    ruda::LESS_EQ, 50,
  };

  const uint64_t kCount = 1000000000;
  const int kMin = 0;
  const int kMax = 100;

  std::vector<int> values;
  std::vector<int> results;

  createRandomNumbers(values, kCount, kMin, kMax);
  std::cout << "[FILTER_TEST] Test Numbers" << std::endl;
  std::cout << values.size() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  runNativeFilter(ctx, values, results);
  // runFilter(ctx, values, results);
  // runCpuFilter(ctx, values, results);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = end - start;
  long long mic_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  std::cout << "[FILTER_TEST] Elapsed Time: " << mic_elapsed << "us"
      << std::endl;

  return 0;
}
