#include <immintrin.h>
#include <cpuid.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "accelerator/common.h"
#include "accelerator/avx/filter.h"

int createRandomNumbers(std::vector<int> &source, const uint64_t kCount,
                        const int kMin, const int kMax) {
  std::random_device rd;
  std::mt19937 generator(rd());

  std::uniform_int_distribution<> uintdist(kMin, kMax);

  std::generate_n(
    std::back_inserter(source),
    kCount,
    [&uintdist, &generator]() {
      return uintdist(generator);
    }
  );

  return accelerator::ACC_OK;
}

int cpuFilter(std::vector<int> &source, accelerator::FilterContext ctx,
              std::vector<int> &results) {
  results.resize(source.size());
  for (int i = 0; i < source.size(); ++i) {
    results[i] = ctx(source[i]);
  }

  return accelerator::ACC_OK;
}

int main() {
  if (__builtin_cpu_supports("avx2")) {
    std::cout << "Enable avx" << std::endl;
  } else {
    std::cout << "Disabled avx" << std::endl;
  }

  accelerator::FilterContext ctx = { accelerator::LESS_EQ, 50 };

  const uint64_t kCount = 1000000000;
  const int kMin = 0;
  const int kMax = 100;

  std::vector<int> values;
  std::vector<int> results;

  createRandomNumbers(values, kCount, kMin, kMax);
  std::cout << "[FILTER_TEST] Test Numbers" << std::endl;
  std::cout << values.size() << std::endl;
  // for (int value : values) {
  //   std::cout << value << " ";
  // }
  // std::cout << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  avx::simpleIntFilter(values, ctx, results);
  // cpuFilter(values, ctx, results);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = end - start;
  long long mic_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  // std::cout << "[FILTER_TEST] Results" << std::endl;
  // std::cout << results.size() << std::endl;
  // for (int result : results) {
  //   std::cout << result << " ";
  // }
  // std::cout << std::endl;
  std::cout << "[FILTER_TEST] Elapsed Time: " << mic_elapsed << "us"
      << std::endl;

  return 0;
}