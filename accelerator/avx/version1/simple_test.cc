
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "simple_filter.h"

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

  return avx_filter::AVX_OK;
}

void runAvxFilter(avx_filter::FilterContext &ctx, std::vector<int> &source,
                  std::vector<int> &results) {
  results.clear();

  std::cout << "[FILTER_TEST] Run AVX Filter" << std::endl;
  avx_filter::avxSimpleIntFilter(source, ctx, results);

  std::cout << "[FILTER_TEST] Results" << std::endl;
  for (int result : results) {
    std::cout << result << " ";
  }
  std::cout << std::endl;
}

void runNativeFilter(avx_filter::FilterContext &ctx, std::vector<int> &source,
                     std::vector<int> &results) {
  results.clear();

  std::cout << "[FILTER_TEST] Run Native Filter" << std::endl;
  results.resize(source.size());
  for (int i = 0; i < source.size(); ++i) {
    results[i] = ctx(source[i]);
  }

  std::cout << "[FILTER_TEST] Results" << std::endl;
  for (int result : results) {
    std::cout << result << " ";
  }
  std::cout << std::endl;
}

int main() {
  std::cout << "[FILTER_TEST] Starts" << std::endl;
  avx_filter::FilterContext ctx = {
    avx_filter::LESS_EQ, 50,
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
  // runAvxFilter(ctx, values, results);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = end - start;
  long long mic_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  std::cout << "[FILTER_TEST] Elapsed Time: " << mic_elapsed << "us"
      << std::endl;

  return 0;
}
