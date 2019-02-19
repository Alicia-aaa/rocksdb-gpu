#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

int main() {
  size_t count = 0;
  size_t size = 64 * 1024 * 1024 * sizeof(float);
  while (true) {
    void *ptr;
    while (posix_memalign(&ptr, 4096, size) != 0) {
      cout << "posix_memalign failed at " << count * 256 << " MB" << endl;
    }
    cout << "ptr = " << ptr << endl;
    while (cudaHostRegister(ptr, size, 0) != cudaSuccess) {
      cout << "cudaHostRegister failed at " << count * 256 << " MB" << endl;
    }
    count++;
    cout << "Allocated " << count * 256 << " MB" << endl;
  }
}
