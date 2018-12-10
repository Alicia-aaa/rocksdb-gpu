
#include <iostream>
#include <vector>

#include "filter.h"

int main() {
  std::cout << "[FILTER_TEST] Starts" << std::endl;
  ruda::Comparator<int> comp = ruda::Comparator<int>(ruda::EQ, 5);

  std::vector<int> values{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  std::vector<bool> results;

  std::cout << "[FILTER_TEST] Run SST Filter" << std::endl;
  // int result = ruda::test(10);
  // std::cout << result << std::endl;
  ruda::sstFilter<int>(values, comp, results);

  std::cout << "[FILTER_TEST] Results" << std::endl;
  for (int i = 0; i < results.size(); ++i) {
    std::cout << i << ":: " << results[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
