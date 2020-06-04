
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
extern "C" {
#include <pinpool.h>
#include <filemap.h>
}
#include "accelerator/cuda/block_decoder.h"
#include "accelerator/cuda/filter.h"
#include "rocksdb/slice.h"
#include "table/format.h"
#include "stdio.h"

#define KB 1024
#define MB 1024 * KB
#define GB 1024 * MB
#define MAX_DEPTH       24
#define SELECTION_SORT  16

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

namespace ruda {
void quickSortI(donardSlice* h_results, unsigned long long int* h_result_idx_arr, unsigned long long int left, unsigned long long int right);
void selection_sort(donardSlice *h_results, unsigned long long int* h_result_idx_arr, unsigned long long int left, unsigned long long int right); 
void quick_sort(donardSlice* h_results, unsigned long long int* h_result_idx_arr, unsigned long long int left, unsigned long long int right, int depth);
namespace kernel {
__global__
void rudaDonardFilterKernel(char **file_address, uint64_t size, uint64_t *block_index, uint64_t *g_block_index, uint64_t block_unit, uint64_t *handles,
 RudaSchema *schema, unsigned long long int *d_results_idx, donardSlice *d_results);

//__global__
//void rudaCopyKernel(unsigned long long int count, donardSlice *d_results, char* d_total_results, unsigned long long int *total_results_idx);

__global__
void rudaCopyKernel(unsigned long long int count, donardSlice *d_results, unsigned long long int *d_result_idx_arr, unsigned long long int *d_target_idx, char* d_total_results);

__global__
void makeIndex(unsigned long long int count, unsigned long long int *d_result_idx_arr);

__device__
char toHex(unsigned char v);

__device__
int memcmp_slice(const void *x, const void *y, size_t n); 

__device__
void selection_sort(donardSlice *d_results, unsigned long long int *d_result_idx_arr, unsigned long long int left, unsigned long long int right);

__global__
void cdp_simple_quicksort(donardSlice* d_results, unsigned long long int *d_result_idx_arr, unsigned long long int left, unsigned long long int right, int depth);

__global__
void testKernel(unsigned long long int count, donardSlice *d_results, unsigned long long int *total_results_idx);
}  // namespace kernel

struct DonardManager {

  // fileMap list
  struct filemap ** fmlist; 

  // Parameter

  unsigned long long int *num_entries_;

  int num_file_;
  int total_blocks_;
  int block_unit_;
  int num_thread_;
  int max_results_count_;   
  std::vector<uint64_t> gpu_blocks_;

  unsigned long long int results_size;
  unsigned long long int count;
 
  
  // MetaData
  char **file_address;
  uint64_t *block_index; // the number of blocks in each file
  uint64_t *g_block_index;
  uint64_t *d_handles;

  // Schema
  RudaSchema *d_schema; // device schema
  RudaSchema h_schema; // host schema

  // Result
  unsigned long long int* d_results_idx;
  donardSlice* d_results;
  donardSlice* h_results;

  unsigned long long int *total_results_idx;
  unsigned long long int *d_result_idx_arr;
  unsigned long long int *h_result_idx_arr;

  unsigned long long int *h_target_idx;
  unsigned long long int *d_target_idx;
  char * d_total_results;
 //char * h_total_results;

  DonardManager(int num_file, int total_blocks, int block_unit, int num_thread, int max_results_count) {
    //std::cout << "[DONARD MANAGER INITALIZE]" << std::endl;
    num_file_ = num_file;
    total_blocks_ = total_blocks;
    block_unit_ = block_unit;
    num_thread_ = num_thread;
    max_results_count_ = max_results_count;
    results_size = 0;
    count = 0;
  }

  void populate(std::vector<std::string> files, std::vector<uint64_t> num_blocks, std::vector<uint64_t> handles, const rocksdb::SlicewithSchema &schema) {
    //std::cout << "[DONARD POPULATE]" << std::endl;
    fmlist = (filemap **)malloc(sizeof(filemap *) * num_file_);
    for(uint i = 0; i < num_file_; i++) {
      fmlist[i] = filemap_open_cuda(files[i].c_str());
    }

    //std::cout << "[DONARD POPULATE1] " << num_file_ << " " << total_blocks_ << std::endl;
    //cudaCheckError(cudaMalloc((void ***) &file_address, sizeof(char *) * num_file_));
    cudaCheckError(cudaHostAlloc((void**)&file_address, sizeof(char*) * num_file_, cudaHostAllocMapped));

    //std::cout << "[DONARD POPULATE 1-1]" << std::endl;
    for(uint i = 0; i < num_file_; i++) {
      file_address[i] = (char *)fmlist[i]->data;
    }
    
    int tmp_gpu_blocks = 0;
    for(uint i = 0; i < num_blocks.size(); i++) {
      int unit = 0;
      int remain = 0;
      if( i == 0 ) {
        unit = num_blocks[i] / block_unit_;
        remain = num_blocks[i] % block_unit_;
      } else {
        unit = (num_blocks[i] - num_blocks[i-1]) / block_unit_;
        remain = (num_blocks[i] - num_blocks[i-1]) % block_unit_;
      }
      tmp_gpu_blocks += unit;
      if(remain != 0) tmp_gpu_blocks += 1;
      gpu_blocks_.emplace_back(tmp_gpu_blocks);
    }

    //std::cout << "gpu block index size " << gpu_blocks_.size() << std::endl;

    //std::cout << "[DONARD POPULATE2] block_index " << sizeof(uint64_t) * num_blocks.size() << std::endl;
    cudaCheckError(cudaMalloc((void **) &block_index, sizeof(uint64_t) * num_blocks.size()));
    cudaCheckError(cudaMemcpy(block_index, &num_blocks[0], sizeof(uint64_t) * num_blocks.size(), cudaMemcpyHostToDevice));

    //std::cout << "[DONARD POPULATE2] g_block_index " << sizeof(uint64_t) * gpu_blocks_.size() << std::endl;
    cudaCheckError(cudaMalloc((void **) &g_block_index, sizeof(uint64_t) * gpu_blocks_.size()));
    cudaCheckError(cudaMemcpy(g_block_index, &gpu_blocks_[0], sizeof(uint64_t) * gpu_blocks_.size(), cudaMemcpyHostToDevice));

    //std::cout << "[DONARD POPULATE2] d_handles " << sizeof(uint64_t) * handles.size() << std::endl;
    cudaCheckError(cudaMalloc((void **) &d_handles, sizeof(uint64_t) * handles.size()));
    cudaCheckError(cudaMemcpy(d_handles, &handles[0], sizeof(uint64_t) * handles.size(), cudaMemcpyHostToDevice));
  
    //std::cout << "[DONARD POPULATE3]" << std::endl;
    // Deep copy for Schema
    rocksdb::SlicewithSchema* copy_schema = schema.clone();
    cudaCheckError(cudaHostRegister(&h_schema, sizeof(rocksdb::SlicewithSchema), cudaHostAllocMapped));
    cudaCheckError(cudaMalloc((void **) &d_schema, sizeof(RudaSchema)));
    cudaCheckError(h_schema.populateToCuda(*copy_schema));
    cudaCheckError(cudaMemcpy(d_schema, &h_schema, sizeof(RudaSchema), cudaMemcpyHostToDevice));

    //std::cout << "[DONARD POPULATE4]" << std::endl;
    cudaCheckError(cudaMalloc((void **) &d_results_idx, sizeof(unsigned long long int)));
    cudaCheckError(cudaMemset(d_results_idx, 0, sizeof(unsigned long long int)));

   // std::cout << "[DONARD POPULATE5] d_results " << sizeof(donardSlice) * max_results_count_ << std::endl;
    cudaCheckError(cudaMalloc((void **) &d_results, sizeof(donardSlice) * max_results_count_));

    cudaCheckError(cudaMalloc((void **) &total_results_idx, sizeof(unsigned long long int)));
    cudaCheckError(cudaMemset(total_results_idx, 0, sizeof(unsigned long long int)));
  
  }

  void executeKernel(double *pushdown_evaluate) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //std::cout << "Filterig" << std::endl;
    cudaEventRecord(start);
    //std::cout << "[DONARD KERNEL EXECUTE] : " << gpu_blocks_.back() << std::endl;
    kernel::rudaDonardFilterKernel<<< gpu_blocks_.back(), num_thread_ >>> (file_address, num_file_, block_index, g_block_index, block_unit_, d_handles,
                             d_schema, d_results_idx, d_results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    *pushdown_evaluate = (double) milliseconds; 

    cudaDeviceSynchronize();

    num_entries_ = (unsigned long long int *)malloc(sizeof(unsigned long long int));
    cudaCheckError(cudaMemcpy(num_entries_, d_results_idx, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    count = *num_entries_;
    uint32_t blockGrid = count / num_thread_ ;
    uint32_t remain = count % num_thread_ ;   
    if (remain != 0) blockGrid += 1;

   /**************************SORT IMPLEMENTATION START****************************/
    //cudaCheckError(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));
  
    //unsigned long long int left = 0;
    //unsigned long long int right = *num_entries_ - 1;
    //std::cout << "Make INDEX " << std::endl;
    //cudaCheckError(cudaMalloc((void **) &d_result_idx_arr, sizeof(unsigned long long int) * (right + 1)));
    //kernel::makeIndex<<< blockGrid, num_thread_ >>> (count, d_result_idx_arr);

    //std::cout << "Sorting on the GPU : " << right << std::endl;
    //kernel::cdp_simple_quicksort<<<1, 1>>>(d_results, d_result_idx_arr, left, right, 0);
    //cudaCheckError(cudaDeviceSynchronize());

   /**************************SORT IMPLEMENTATION END****************************/

    unsigned long long int left = 0;
    unsigned long long int right = count;
    h_results = (donardSlice *)malloc(sizeof(donardSlice) * count);
    cudaCheckError(cudaMemcpy(h_results, d_results, sizeof(donardSlice) * count, cudaMemcpyDeviceToHost));

    h_result_idx_arr = (unsigned long long int *)malloc(sizeof(unsigned long long int) * count);
    for(int i = 0; i < count; i++) h_result_idx_arr[i] = i;
    // std::cout << "Sorting on the CPU Quick : " << count << std::endl;
    // selection_sort(h_results, h_result_idx_arr, left, right-1);
    // quick_sort(h_results, h_result_idx_arr, left, right-1, 0);
    quickSortI(h_results, h_result_idx_arr, left, right - 1);
    h_target_idx = (unsigned long long int *)malloc(sizeof(unsigned long long int) * count);

    for(uint i = 0; i < count ; i++) {
      if (i == 0)
        h_target_idx[i] = 0;
      else 
        h_target_idx[i] = h_target_idx[i-1] + h_results[h_result_idx_arr[i-1]].key_size + h_results[h_result_idx_arr[i-1]].d_size + 4;
      results_size += h_results[i].key_size + h_results[i].d_size + 4;
    }

    //std::cout << "Copying Device to Host" << std::endl;

    cudaCheckError(cudaMalloc((void **) &d_result_idx_arr, sizeof(unsigned long long int) * count));
    cudaCheckError(cudaMemcpy(d_result_idx_arr, h_result_idx_arr, sizeof(unsigned long long int) * count, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **) &d_target_idx, sizeof(unsigned long long int) * count));
    cudaCheckError(cudaMemcpy(d_target_idx, h_target_idx, sizeof(unsigned long long int) * count, cudaMemcpyHostToDevice));

    //std::cout << " results_size : " << results_size << std::endl;
    cudaCheckError(cudaMalloc((void **) &d_total_results, sizeof(char) * results_size));


    //std::cout << " blockGrid : " << blockGrid << " count : " << count << std::endl;
    //kernel::rudaCopyKernel<<< blockGrid , num_thread_ >>> (count, d_results, d_total_results, total_results_idx); 
    kernel::rudaCopyKernel<<< blockGrid , num_thread_ >>> (count, d_results, d_result_idx_arr, d_target_idx, d_total_results);
    cudaDeviceSynchronize();
  }

  void translatePairsToSlices(std::vector<rocksdb::PinnableSlice> &keys, std::vector<rocksdb::PinnableSlice> &results, char **data_buf, uint64_t *num_entries) {

    //std::cout << "[DONARD TRANSLATE TO SLICES 0]" << std::endl;
    //h_total_results = (char *)malloc(sizeof(char) * results_size);
    //cudaCheckError(cudaMemcpy(h_total_results, d_total_results, sizeof(char) * results_size, cudaMemcpyDeviceToHost));

    *num_entries = count;
    *data_buf = (char *)malloc(sizeof(char) * results_size);
    char *target_ptr = *data_buf;
    cudaCheckError(cudaMemcpy(target_ptr, d_total_results, sizeof(char) * results_size, cudaMemcpyDeviceToHost));
  
    /*
    //std::cout << "[DONARD TRANSLATE TO SLICES 1]" << std::endl;  
    //std::cout << "[DONARD TRANSLATE TO SLICES 2] " << count << std::endl;

    char *initialPtr = h_total_results;
    for (size_t i = 0; i < count; i++) {
      size_t key_size = *((unsigned short *)initialPtr);
      initialPtr += 2;
      size_t value_size = *((unsigned short *)initialPtr);
      initialPtr += 2;

      keys.emplace_back(std::move(rocksdb::PinnableSlice(initialPtr, key_size)));
      initialPtr += key_size;

      results.emplace_back(std::move(rocksdb::PinnableSlice(initialPtr, value_size)));
      initialPtr += value_size;
    }
   */
  }

  void clear() {
    for(uint i = 0; i < num_file_; i++) {
      filemap_free(fmlist[i]);
    } 
    cudaCheckError(cudaFreeHost(file_address)); 
    cudaCheckError(cudaFree(block_index));
    cudaCheckError(cudaFree(g_block_index));
    cudaCheckError(cudaFree(d_handles));

    cudaCheckError(h_schema.clear());
    cudaCheckError(cudaFree(d_schema));

    cudaCheckError(cudaFree(d_results_idx));
    cudaCheckError(cudaFree(d_results));
    cudaCheckError(cudaFree(total_results_idx));
    cudaCheckError(cudaFree(d_result_idx_arr));
    cudaCheckError(cudaFree(d_total_results));
    cudaCheckError(cudaFree(d_target_idx));

    free(h_target_idx);
    free(h_results);
    free(h_result_idx_arr);
   // free(h_total_results);
  }
};

__global__
void kernel::rudaDonardFilterKernel(char **file_address, uint64_t size, uint64_t *block_index, uint64_t *g_block_index, uint64_t g_block_unit, uint64_t * d_handles,
 RudaSchema *schema, unsigned long long int *results_idx, donardSlice *d_results) {  
  
  // blockDim.x * blockIdx.x + threadIdx.x;
  // blockDim = number of Thread in block

  // Find file location 
  unsigned int idx = getFileIdx(blockIdx.x, size, g_block_index);

  int gBlockOffset = (idx == 0) ? blockIdx.x : blockIdx.x - g_block_index[idx-1];
  int accumulatedBlocks = (idx == 0) ? g_block_unit * gBlockOffset : block_index[idx-1] + g_block_unit * gBlockOffset;

  int gBlockRemain = 0;
  if (blockIdx.x == g_block_index[idx] - 1) {
    gBlockRemain = (idx == 0) ? block_index[idx] % g_block_unit : (block_index[idx] - block_index[idx -1]) % g_block_unit;
  }
  if (gBlockRemain != 0) g_block_unit = gBlockRemain; 

  char *filePtr = file_address[idx];  
  const char *startPtr = (gBlockOffset == 0) ? filePtr : filePtr + d_handles[accumulatedBlocks -1];

  uint32_t blockSize = 0;
  int kDataBlockIndexTypeBitShift = 31;
  uint32_t kNumRestartsMask = (1u << kDataBlockIndexTypeBitShift) - 1u;
  uint32_t kBlockTrailerSize = 5;

  uint32_t threadsPerBlock = blockDim.x / g_block_unit;
  uint32_t threadRemain = blockDim.x % g_block_unit;
  uint32_t threadIdInBlock = threadIdx.x / g_block_unit;
  uint32_t blockLocation = threadIdx.x % g_block_unit; 

  if (blockLocation < threadRemain) threadsPerBlock += 1;

  if (gBlockOffset == 0) {
   if(blockLocation == 0) {
    blockSize = d_handles[accumulatedBlocks] - kBlockTrailerSize;
   } else {
    blockSize = d_handles[accumulatedBlocks + blockLocation] - d_handles[accumulatedBlocks + blockLocation - 1] - kBlockTrailerSize;
    startPtr += d_handles[accumulatedBlocks + blockLocation - 1];
   }
  } else { 
    blockSize = d_handles[accumulatedBlocks + blockLocation] - d_handles[accumulatedBlocks + blockLocation - 1] - kBlockTrailerSize; 
    startPtr += d_handles[accumulatedBlocks + blockLocation - 1] - d_handles[accumulatedBlocks - 1];
  }

  uint32_t numRestarts = DecodeFixed32(startPtr + blockSize - sizeof(uint32_t));
  numRestarts = numRestarts & kNumRestartsMask;
  uint32_t restartOffset = static_cast<uint32_t>(blockSize) - (1 + numRestarts) * sizeof(uint32_t);

  if (numRestarts < threadIdInBlock + 1) return;

  uint32_t numTask = numRestarts / threadsPerBlock;
  uint32_t remainNumTask = numRestarts % threadsPerBlock;

  bool lastThread = false;
  if (numTask == 0 && threadIdInBlock == numRestarts - 1) lastThread = true;
  if (numTask != 0 && threadIdInBlock == threadsPerBlock - 1) lastThread = true; 

  if (threadIdInBlock < remainNumTask) numTask += 1;

  if(numTask == 0) return;

  uint32_t startLocation = restartOffset;

  startLocation += (threadIdInBlock >= remainNumTask) ? (remainNumTask + (numTask * threadIdInBlock)) * sizeof(uint32_t) : (numTask * threadIdInBlock * sizeof(uint32_t));

  if (!lastThread) restartOffset = 0;
  DecodeNFilterOnSchemaDonard(startPtr, restartOffset, startLocation, numTask, schema, results_idx, d_results); 

}

/*
__global__
void kernel::rudaCopyKernel(unsigned long long int count, donardSlice *d_results, char* total_results, unsigned long long int *total_results_idx) {

  unsigned long long int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx >= count) {
    return;
  }

  //printf("blockidx : %d, threadidx : %d\n", blockIdx.x, threadIdx.x);
  size_t key_size = d_results[idx].key_size;
  size_t value_size = d_results[idx].d_size;
  unsigned long long int kvPairSize = key_size + value_size;

  unsigned long long int resultOffset = atomicAdd(total_results_idx, kvPairSize + 4);

  char* targetIdx = total_results + resultOffset;

  char *k_size = (char *)&key_size;
  for (uint i = 0; i < sizeof(unsigned short); i++) {
    targetIdx[i] = k_size[i];
  }

  targetIdx += 2;

  char *v_size = (char *)&value_size;
  for (uint i = 0; i < sizeof(unsigned short); i++) {
    targetIdx[i] = v_size[i];
  }

  targetIdx += 2;

  for(uint i = 0; i < key_size; i++) {
    targetIdx[i] = d_results[idx].key[i];
  }

  targetIdx += key_size;

  for(uint i = 0; i < value_size; i++) {
    targetIdx[i] = d_results[idx].d_data[i];
  }  
} */

__global__
void kernel::rudaCopyKernel(unsigned long long int count, donardSlice *d_results, unsigned long long int *d_result_idx_arr,
                            unsigned long long int *d_target_idx, char* total_results) {

  unsigned long long int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx >= count) {
    return;
  }

  //printf("blockidx : %d, threadidx : %d\n", blockIdx.x, threadIdx.x);
  size_t key_size = d_results[d_result_idx_arr[idx]].key_size;
  size_t value_size = d_results[d_result_idx_arr[idx]].d_size;

  char* targetIdx = total_results + d_target_idx[idx];

  char *k_size = (char *)&key_size;
  for (uint i = 0; i < sizeof(unsigned short); i++) {
    targetIdx[i] = k_size[i];
  }

  targetIdx += 2;

  char *v_size = (char *)&value_size;
  for (uint i = 0; i < sizeof(unsigned short); i++) {
    targetIdx[i] = v_size[i];
  }

  targetIdx += 2;

  for(uint i = 0; i < key_size; i++) {
    targetIdx[i] = d_results[d_result_idx_arr[idx]].key[i];
  }

  targetIdx += key_size;

  for(uint i = 0; i < value_size; i++) {
    targetIdx[i] = d_results[d_result_idx_arr[idx]].d_data[i];
  }  
}

__global__
void kernel::makeIndex(unsigned long long int count, unsigned long long int* d_result_idx_arr) {
  unsigned long long int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx >= count) {
    return;
  }

  d_result_idx_arr[idx] = idx;
}


__device__
char kernel::toHex(unsigned char v) {
  if (v <= 9) {
    return '0' + v;
  }
  return 'A' + v - 10;
}

__device__
int kernel::memcmp_slice(const void* s1, const void* s2, size_t n) {
  const unsigned char *p1 = (const unsigned char *) s1;
  const unsigned char *p2 = (const unsigned char *) s2;
  p1 += 4;
  p2 += 4;
  n -= 4;
  while(n--)
    if( *p1 != *p2 )
      return *p1 - *p2;
    else
      p1++,p2++;
    return 0;
}
/*
__device__ 
void kernel::selection_sort(donardSlice *d_results, unsigned long long int* d_result_idx_arr,
                            unsigned long long int left, unsigned long long int right) {
  for( int i = left ; i <= right ; ++i ) {
    int min_val = d_results[d_result_idx_arr[i]].k_int;
    int min_idx = i;

    for( int j = i+1 ; j <= right ; ++j ) {
      int val_j = d_results[d_result_idx_arr[j]].k_int;
      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if( i != min_idx ) {
      unsigned long long int temp = d_result_idx_arr[i]; 
      d_result_idx_arr[i] = d_result_idx_arr[min_idx];
      d_result_idx_arr[min_idx] = temp;
    }
  }
}

__global__
void kernel::cdp_simple_quicksort(donardSlice* d_results, unsigned long long int* d_result_idx_arr,
             unsigned long long int left, unsigned long long int right, int depth) {
  if( depth >= MAX_DEPTH || right - left <= SELECTION_SORT ) {
    selection_sort(d_results, d_result_idx_arr, left, right);
    return;
  }

  unsigned long long int *lptr = d_result_idx_arr + left;
  unsigned long long int *rptr = d_result_idx_arr + right;
  unsigned long long int pivot = d_result_idx_arr[(left + right)/2];

  while (lptr <= rptr) {
    int lval = d_results[(*lptr)].k_int;
    int rval = d_results[(*rptr)].k_int;
    int pval = d_results[pivot].k_int;

    // Move the left pointer as long as the pointed element is smaller than the pivot.
    while (lval < pval) {
      lptr++;
      lval = d_results[(*lptr)].k_int;
    }

    // Move the right pointer as long as the pointed element is larger than the pivot.
    while (rval > pval) {
      rptr--;
      rval = d_results[(*rptr)].k_int;
    }

    // If the swap points are valid, do the swap!
    if (lptr <= rptr) {
      unsigned long long int temp = *lptr;
      *lptr = *rptr;
      *rptr = temp;
      lptr++;
      rptr--;
    }
  }

  unsigned long long int nright = rptr - d_result_idx_arr;
  unsigned long long int nleft  = lptr - d_result_idx_arr;

  if (left < (rptr - d_result_idx_arr)) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cdp_simple_quicksort<<< 1, 1, 0, s >>>(d_results, d_result_idx_arr, left, nright, depth+1);
    cudaStreamDestroy(s);
  }

  if ((lptr - d_result_idx_arr) < right) {
    cudaStream_t s1;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(d_results, d_result_idx_arr, nleft, right, depth+1);
    cudaStreamDestroy(s1);
  }
}
*/
__global__
void kernel::testKernel(unsigned long long int count, donardSlice *d_results, unsigned long long int *total_results_idx) {
  unsigned long long int idx = blockDim.x * blockIdx.x + threadIdx.x;
  printf("idx : %d\n", idx);
}

int compare(unsigned int a[], unsigned int b[], int pk_num) {
  for(int i = 0; i < pk_num; i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  } 
  return 0;
}

void selection_sort(donardSlice *h_results, unsigned long long int* h_result_idx_arr,
                            unsigned long long int left, unsigned long long int right) {
  for( int i = left ; i <= right ; ++i ) {
    char* min_val = h_results[h_result_idx_arr[i]].key;
    int min_idx = i;

    for( int j = i+1 ; j <= right ; ++j ) {
      char* val_j = h_results[h_result_idx_arr[j]].key;
      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if( i != min_idx ) {
      unsigned long long int temp = h_result_idx_arr[i]; 
      h_result_idx_arr[i] = h_result_idx_arr[min_idx];
      h_result_idx_arr[min_idx] = temp;
    }
  }
}

void quick_sort(donardSlice* h_results, unsigned long long int* h_result_idx_arr, 
                unsigned long long int left, unsigned long long int right, int depth) {
 
  if(depth % 1000 == 0) std::cout << "depth : " << depth << std::endl;
  int key_size = h_results[0].key_size;

  unsigned long long int* lptr = h_result_idx_arr + left;
  unsigned long long int* rptr = h_result_idx_arr + right;
  unsigned long long int pivot = h_result_idx_arr[(left + right)/ 2];

  char* lval;
  char* rval;
  char* pval;

  while(lptr <= rptr) {
    lval = h_results[(*lptr)].key;
    rval = h_results[(*rptr)].key;
    pval = h_results[pivot].key;

    while(memcmp(lval, pval, key_size) < 0 && ((lptr - h_result_idx_arr) < right)) {
      lptr++;
      lval = h_results[(*lptr)].key;
    }

    while(memcmp(rval, pval, key_size) > 0 && ((rptr - h_result_idx_arr) > left)) {
      rptr--;
      rval = h_results[(*rptr)].key;
    }

    if(lptr <= rptr) {
      unsigned long long int temp = *lptr;
      *lptr = *rptr;
      *rptr = temp;
      lptr++;
      rptr--;
    }
  }

  unsigned long long int nright = rptr - h_result_idx_arr;
  unsigned long long int nleft  = lptr - h_result_idx_arr;

  if(left < nright) {
    quick_sort(h_results, h_result_idx_arr, left, nright, depth+1);
  }

  if(nleft < right) {
    quick_sort(h_results, h_result_idx_arr, nleft, right, depth+1);
  }
}

typedef unsigned long long item;

typedef struct stack
{
  uint32_t size;
  uint32_t top;
  item *entries;
} stack;

# define MIN_STACK 128
# define INVALID   0xDeadD00d
# define SWAP(x, y)     { unsigned long long int t = x; x = y; y = t; }

stack *newStack() {
  stack *s = (stack *) calloc(MIN_STACK, sizeof(stack));
  if (s) {
    s->size = MIN_STACK;
    s->top = 0;
    s->entries = (item *) calloc(MIN_STACK, sizeof(item));
    if (s->entries) {
      return s;
    }
  }
  return (stack *)0;
}

void delStack(stack *s) {
  if (s->entries) {
    free(s->entries);
  }
  if (s) {
    free(s);
  }
  return;
}

item pop(stack *s) {
  if (s && s->top > 0) {
    s->top -= 1;
    return s->entries[s->top];
  }
  return INVALID;
}

void push(stack *s, item i) {
  if (s) {
    if (s->top == s->size) {
      s->size *= 2;
      s->entries = (item *) realloc(s->entries, s->size * sizeof(item));
    }
    s->entries[s->top] = i;
    s->top += 1;
  }
  return;
}

bool empty(stack *s) { return s->top == 0; }

unsigned long long int partition(donardSlice* h_results, unsigned long long int* h_result_idx_arr, 
              unsigned long long int left, unsigned long long int right, int key_size) {
  unsigned long long int pivotValue = h_result_idx_arr[(left + right) / 2];

  unsigned long long int i = left - 1;
  unsigned long long int j = right + 1;
  do {
    do {
      i += 1;
    } while (memcmp(h_results[h_result_idx_arr[i]].key, h_results[pivotValue].key, key_size) < 0);
    do {
      j -= 1;
    } while (memcmp(h_results[h_result_idx_arr[j]].key, h_results[pivotValue].key, key_size) > 0);
    if (i < j) {
      SWAP(h_result_idx_arr[i], h_result_idx_arr[j]);
    }
  } while (i < j);
  return j;
}

void quickSortI(donardSlice* h_results, unsigned long long int* h_result_idx_arr, 
                unsigned long long int left, unsigned long long int right) {
  stack *s = newStack();

  int key_size = h_results[0].key_size;
  push(s, left);
  push(s, right);
  while (!empty(s)) {
    unsigned long long int high = pop(s);
    unsigned long long int low = pop(s);
    unsigned long long int p = partition(h_results, h_result_idx_arr, low, high, key_size);
    if (p + 1 < high) {
      push(s, p + 1);
      push(s, high);
    }
    if (low < p) {
      push(s, low);
      push(s, p);
    }
  }
  delStack(s);
  return;
}

int donardFilter( std::vector<std::string> files, std::vector<uint64_t> num_blocks, std::vector<uint64_t> handles, const rocksdb::SlicewithSchema &schema,
                  uint64_t max_results_count,
                  std::vector<rocksdb::PinnableSlice> &keys,
                  std::vector<rocksdb::PinnableSlice> &results, char **data_buf, uint64_t *num_entries, double* pushdown_evaluate) {

  // std::cout << "[GPU][donardFilter] START" << std::endl;

  void *warming_up;
  cudaCheckError(cudaMalloc(&warming_up, 0));
  cudaCheckError(cudaFree(warming_up));

  DonardManager donard_mgr(
      files.size(),
      num_blocks.back(),
      30,
      128 /* kBlockSize */,
      max_results_count);

  donard_mgr.populate(files, num_blocks, handles, schema);

  donard_mgr.executeKernel(pushdown_evaluate);

  donard_mgr.translatePairsToSlices(keys, results, data_buf, num_entries);
  donard_mgr.clear();

//  std::cout << "This is end " << std::endl;
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return accelerator::ACC_OK;
}

}  // namespace ruda
