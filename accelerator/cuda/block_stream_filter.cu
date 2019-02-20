
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "accelerator/cuda/block_decoder.h"
#include "accelerator/cuda/filter.h"
#include "rocksdb/slice.h"
#include "table/format.h"

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
// Kernels
__global__
void rudaStreamIntBlockFilterKernel(// Parameters (ReadOnly)
                                    size_t offset, size_t kSize,
                                    size_t dataSize, size_t resultsCount,
                                    char *data, uint64_t *seek_indices,
                                    accelerator::FilterContext *ctx,
                                    uint64_t *block_seek_start_indices,
                                    // Variables
                                    unsigned long long int *results_idx,
                                    // Results
                                    RudaKVPair *results);
}  // namespace kernel

struct RudaBlockStreamContext {
  cudaStream_t stream;
  uint64_t *d_gpu_block_seek_starts;

  // Cuda Kernel Parameters
  const size_t kSize = 0;             // Total seek indices count
  const int kBlockSize = 0;
  const int kGridSize = 0;
  const size_t kMaxResultsCount = 0;  // Total count of filtered Key-Value pairs

  // Cuda Kernel Parameters - stream
  const int kStreamCount = 0;
  const int kStreamSize = 0;
  const int kGridSizePerStream = 0;

  // Max cached datablocks size on same gpu block (For using SharedMemory)
  size_t kMaxCacheSize = 0;

  // Allocated offsets on gpu
  size_t seek_start_offset, seek_size, datablocks_start_offset, datablocks_size;

  uint64_t *gpu_block_seek_starts;

  RudaBlockStreamContext(const size_t total_size, const int block_size,
                         const int grid_size, const size_t max_results_count,
                         const int stream_count, const int stream_size,
                         const int grid_size_per_stream)
      : kSize(total_size), kBlockSize(block_size), kGridSize(grid_size),
        kMaxResultsCount(max_results_count), kStreamCount(stream_count),
        kStreamSize(stream_size), kGridSizePerStream(grid_size_per_stream) {
    cudaCheckError(cudaMallocHost(
        (void **) &gpu_block_seek_starts,
        sizeof(uint64_t) * kGridSizePerStream));
  }

  void freeCudaObjects() {
    cudaCheckError( cudaFree(d_gpu_block_seek_starts) );
  }

  void initializeStream() {
    cudaCheckError( cudaStreamCreate(&stream) );
  }

  void destroyStream() {
    cudaCheckError( cudaStreamDestroy(stream) );
  }

  void clear() {
    freeCudaObjects();
    destroyStream();
    cudaCheckError( cudaFreeHost(gpu_block_seek_starts) );
  }

  void cudaMallocGpuBlockSeekStarts() {
    cudaCheckError(cudaMalloc(
        (void **) &d_gpu_block_seek_starts,
        sizeof(uint64_t) * kGridSizePerStream));
  }

  size_t calculateGpuBlockSeekStarts(const std::vector<char> &datablocks,
                                     const std::vector<uint64_t> &seek_indices,
                                     size_t start, size_t size) {
    for (size_t i = 0; i < kGridSizePerStream; ++i) {
      size_t thread_idx = start + i * kBlockSize;
      if (thread_idx >= kSize) {
        gpu_block_seek_starts[i] = 0;
      } else {
        gpu_block_seek_starts[i] = seek_indices[thread_idx];
      }
    }

    size_t max_cache_size = 0;
    for (size_t i = 0; i < kGridSizePerStream; ++i) {
      size_t cache_size;
      if (i == kGridSizePerStream - 1) {
        if ((start + size) >= kSize) {
          cache_size = datablocks.size() - gpu_block_seek_starts[i];
        } else {
          cache_size =
              seek_indices[start + kStreamSize - 1] - gpu_block_seek_starts[i];
        }
        if (cache_size > max_cache_size) {
          max_cache_size = cache_size;
        }
        break;
      }

      cache_size = gpu_block_seek_starts[i+1] - gpu_block_seek_starts[i];
      if (cache_size > max_cache_size) {
        max_cache_size = cache_size;
      }
    }
    return sizeof(char) * max_cache_size;
  }

  void initParams(const std::vector<char> &datablocks,
                  const std::vector<uint64_t> &seek_indices,
                  size_t start, size_t size, size_t start_datablocks,
                  size_t size_datablocks) {
    seek_start_offset = start;
    seek_size = size;
    datablocks_start_offset = start_datablocks;
    datablocks_size = size_datablocks;
    kMaxCacheSize = calculateGpuBlockSeekStarts(
        datablocks, seek_indices, start, size);
  }

  void populateToCuda(const std::vector<char> &datablocks,
                      const std::vector<uint64_t> &seek_indices,
                      char *d_datablocks, uint64_t *d_seek_indices,
                      size_t start, size_t size, size_t start_datablocks,
                      size_t size_datablocks) {
    // cudaCheckError(cudaMemcpyAsync(
    //     &d_datablocks[start_datablocks], &datablocks[start_datablocks],
    //     sizeof(char) * size_datablocks, cudaMemcpyHostToDevice,
    //     stream));
    // cudaCheckError(cudaMemcpyAsync(
    //     &d_seek_indices[start], &seek_indices[start],
    //     sizeof(uint64_t) * size, cudaMemcpyHostToDevice,
    //     stream));
    // cudaCheckError(cudaMemcpyAsync(
    //     d_gpu_block_seek_starts, gpu_block_seek_starts,
    //     sizeof(uint64_t) * kGridSizePerStream, cudaMemcpyHostToDevice,
    //     stream));
  }
};

struct RudaBlockStreamManager {
  // Cuda Kernel Parameters
  // IMPORTANT: Kernel Parameters never be changed except in constructor.
  size_t kSize = 0;             // Total seek indices count
  int kBlockSize = 0;
  int kGridSize = 0;
  size_t kMaxResultsCount = 0;  // Total count of filtered Key-Value pairs

  // Cuda Kernel Parameters - stream
  int kStreamCount = 0;
  int kStreamSize = 0;
  int kGridSizePerStream = 0;

  // Streams
  std::vector<RudaBlockStreamContext> stream_ctxs;

  // Parameters
  char *d_datablocks;
  uint64_t *d_seek_indices;
  accelerator::FilterContext *d_cond_ctx;

  // Results - Device
  unsigned long long int *d_results_idx;    // Atomic increment counter index
  RudaKVPair *d_results;    // Filtered KV pairs

  // Results - Host
  // Total results count copied from 'd_results_idx' after kernel call...
  unsigned long long int h_results_count;
  RudaKVPair *h_results;    // Filtered KV Pairs

  RudaBlockStreamManager(const size_t total_size, const int block_size,
                         const size_t stream_count,
                         const size_t max_results_count) {
    kSize = total_size;
    kBlockSize = block_size;
    kStreamCount = stream_count;
    kStreamSize = ceil((float) total_size / (float) stream_count);
    kGridSize = ceil((float) total_size / (float) block_size);
    kGridSizePerStream = ceil((float) kStreamSize / (float) block_size);
    while (kStreamSize <= kBlockSize && kBlockSize != 1) {
      kBlockSize = kBlockSize >> 1;
      kGridSize = ceil((float) total_size / (float) kBlockSize);
      kGridSizePerStream = ceil((float) kStreamSize / (float) kBlockSize);
    }
    for (size_t i = 0; i < kStreamCount; ++i) {
      stream_ctxs.emplace_back(
          kSize, kBlockSize, kGridSize, kMaxResultsCount, kStreamCount,
          kStreamSize, kGridSizePerStream);
    }
  }

  void registerPinnedMemory(std::vector<char> &datablocks,
                            std::vector<uint64_t> &seek_indices,
                            accelerator::FilterContext &cond_ctx) {
    cudaCheckError(cudaHostRegister(
        &datablocks[0], sizeof(char) * datablocks.size(), cudaHostAllocMapped));
    cudaCheckError(cudaHostRegister(
        &seek_indices[0], sizeof(uint64_t) * seek_indices.size(),
        cudaHostAllocMapped));
    cudaCheckError(cudaHostRegister(
        &cond_ctx, sizeof(accelerator::FilterContext), cudaHostAllocMapped));
  }

  void unregisterPinnedMemory(std::vector<char> &datablocks,
                              std::vector<uint64_t> &seek_indices,
                              accelerator::FilterContext &cond_ctx) {
    cudaCheckError( cudaHostUnregister(&datablocks[0]) );
    cudaCheckError( cudaHostUnregister(&seek_indices[0]) );
    cudaCheckError( cudaHostUnregister(&cond_ctx) );
  }

  void initParams(const std::vector<char> &datablocks,
                  const std::vector<uint64_t> &seek_indices,
                  const accelerator::FilterContext &cond_ctx) {
    for (size_t i = 0; i < kStreamCount; ++i) {
      RudaBlockStreamContext &ctx = stream_ctxs[i];
      uint64_t start = i * kStreamSize;
      uint64_t start_datablocks = seek_indices[start];

      uint64_t size, size_datablocks;
      if (i < kStreamCount - 1) {
        size = kStreamSize;
        size_datablocks = seek_indices[start + size] - start_datablocks;
      } else {
        size = kSize - start;
        size_datablocks = datablocks.size() - start_datablocks;
      }

      // Copies sources to GPU (datablocks, seek_indices)
      // Accelerated by stream-pipelining...
      ctx.initParams(
          datablocks, seek_indices, start, size, start_datablocks,
          size_datablocks);
    }
  }

  void populateToCuda(const std::vector<char> &datablocks,
                      const std::vector<uint64_t> &seek_indices,
                      const accelerator::FilterContext &cond_ctx) {
    // Allocation Part
    // Cuda Parameters
    cudaCheckError(cudaMalloc(
        (void **) &d_datablocks, sizeof(char) * datablocks.size()));
    cudaCheckError(cudaMalloc(
        (void **) &d_seek_indices, sizeof(uint64_t) * kSize));
    cudaCheckError(cudaMalloc(
        (void **) &d_cond_ctx, sizeof(accelerator::FilterContext)));
    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.initializeStream();
      ctx.cudaMallocGpuBlockSeekStarts();
    }

    // Cuda Results
    cudaCheckError(cudaMalloc(
        (void **) &d_results_idx, sizeof(unsigned long long int)));
    cudaCheckError(cudaMalloc(
        (void **) &d_results, sizeof(RudaKVPair) * kMaxResultsCount));

    cudaCheckError(cudaMemcpy(
        d_cond_ctx, &cond_ctx, sizeof(accelerator::FilterContext),
        cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(
        d_results_idx, 0, sizeof(unsigned long long int)));

    // Asynchronous memory copying
    for (size_t i = 0; i < kStreamCount; ++i) {
      RudaBlockStreamContext &ctx = stream_ctxs[i];
      uint64_t start = i * kStreamSize;
      uint64_t start_datablocks = seek_indices[start];

      uint64_t size, size_datablocks;
      if (i < kStreamCount - 1) {
        size = kStreamSize;
        size_datablocks = seek_indices[start + size] - start_datablocks;
      } else {
        size = kSize - start;
        size_datablocks = seek_indices[kSize - 1] - start_datablocks;
      }

      // Copies sources to GPU (datablocks, seek_indices)
      // Accelerated by stream-pipelining...
      ctx.populateToCuda(
          datablocks, seek_indices, d_datablocks, d_seek_indices,
          start, size, start_datablocks, size_datablocks);
    }
  }

  void freeCudaObjects() {
    cudaCheckError( cudaFree(d_datablocks) );
    cudaCheckError( cudaFree(d_seek_indices) );
    cudaCheckError( cudaFree(d_cond_ctx) );
    cudaCheckError( cudaFree(d_results_idx) );
    cudaCheckError( cudaFree(d_results) );
  }

  void clear() {
    freeCudaObjects();
    for (auto &ctx : stream_ctxs) {
      ctx.clear();
    }
    stream_ctxs.clear();
  }
};

__global__
void kernel::rudaStreamIntBlockFilterKernel(// Parameters (ReadOnly)
                                            size_t offset, size_t kSize,
                                            size_t dataSize,
                                            size_t resultsCount,
                                            char *data, uint64_t *seek_indices,
                                            accelerator::FilterContext *ctx,
                                            uint64_t *block_seek_start_indices,
                                            // Variables
                                            unsigned long long int *results_idx,
                                            // Results
                                            RudaKVPair *results) {
  uint64_t i = offset + blockDim.x * blockIdx.x + threadIdx.x;

  // Overflow kernel ptr case.
  if (i >= kSize) {
    return;
  }

  // Shared variables.
  // Caches data used from threads in single block.
  extern __shared__ char cached_data[];

  uint64_t block_seek_start_index = block_seek_start_indices[blockIdx.x];
  uint64_t start = seek_indices[i] - block_seek_start_index;
  uint64_t end = 0;
  if (i == (kSize - 1)) {
    // Last seek index case. 'end' must be end of data.
    end = dataSize - block_seek_start_index;
  } else {
    // 'end' must be next seek index.
    end = seek_indices[i + 1] - block_seek_start_index;
  }

  for (size_t j = start; j < end; ++j) {
    size_t data_idx = block_seek_start_index + j;
    cached_data[j] = data[data_idx];
  }

  __syncthreads();

  size_t size = end - start;
  DecodeSubDataBlocks(
      // Parameters
      cached_data, size, start, end, ctx,
      // Results
      results_idx, results);
}

int sstStreamIntBlockFilter(std::vector<char> &datablocks,
                            std::vector<uint64_t> &seek_indices,
                            accelerator::FilterContext context,
                            const size_t max_results_count,
                            std::vector<rocksdb::Slice> &keys,
                            std::vector<rocksdb::Slice> &values) {
  RudaBlockStreamManager block_stream_mgr(
      seek_indices.size() /* kSize */,
      64 /* kBlockSize */,
      4 /* kStreamCount */,
      max_results_count);

  // Copy & Initializes variables from host to device.
  block_stream_mgr.initParams(datablocks, seek_indices, context);

  std::cout << "[CUDA][BlockStreamManager]" << std::endl
      << "kSize: " << block_stream_mgr.kSize << std::endl
      << "kGridSize: " << block_stream_mgr.kGridSize << std::endl
      << "kBlockSize: " << block_stream_mgr.kBlockSize << std::endl
      << "kGridSizePerStream: " << block_stream_mgr.kGridSizePerStream << std::endl
      << "kStreamCount: " << block_stream_mgr.kStreamCount << std::endl
      << "kStreamSize: " << block_stream_mgr.kStreamSize << std::endl
      << "DataSize: " << datablocks.size() << std::endl
      << "Max Results Count: " << block_stream_mgr.kMaxResultsCount << std::endl
      << "======================" << std::endl;

  std::cout << "BlockStreamContexts" << std::endl;
  for (size_t i = 0; i < block_stream_mgr.kStreamCount; ++i) {
    RudaBlockStreamContext &ctx = block_stream_mgr.stream_ctxs[i];
    std::cout << "Stream: " << i << std::endl
        << "Start Offset: " << ctx.seek_start_offset << std::endl
        << "Size: " << ctx.seek_size << std::endl
        << "Start DataBlocks: " << ctx.datablocks_start_offset << std::endl
        << "Size DataBlocks: " << ctx.datablocks_size << std::endl
        << "_____" << std::endl;
    for (size_t j = 0; j < ctx.kGridSizePerStream; ++j) {
      std::cout << "GPU Block Seek Start[" << j << "]: "
          << ctx.gpu_block_seek_starts[j] << std::endl;
    }
    std::cout << "-----------" << std::endl;
  }
  // block_stream_mgr.populateToCuda(datablocks, seek_indices, context);

  cudaDeviceSynchronize();
  block_stream_mgr.clear();

  // cudaCheckError(cudaDeviceSetLimit(
  //     cudaLimitMallocHeapSize, 100 * sizeof(char) * datablocks.size()));

  // // Call kernel per streams.
  // for (size_t i = 0; i < block_context.kStreamCount; ++i) {
  //   cudaStream_t stream = block_context.streams[i];
  //   kernel::rudaIntBlockFilterKernel<<<block_context.kGridSizePerStream,
  //                                      block_context.kBlockSize,
  //                                      block_context.kMaxCacheSize,
  //                                      stream>>>(
  //       // Kernel Parameters
  //       i * block_context.kStreamSize, // Stream Offset
  //       block_context.kSize, datablocks.size(),
  //       block_context.kMaxResultsCount, block_context.d_datablocks,
  //       block_context.d_seek_indices, block_context.d_cond_ctx,
  //       block_context.d_block_seek_start_indices_per_stream[i],
  //       // Kernel Variables
  //       block_context.d_results_idx,
  //       // Kernel Results
  //       block_context.d_results);
  // }

  // block_context.populateResultsFromCuda();
  // block_context.copyToFinalResults(keys, values);

  // std::cout << "Total Results Count: " << block_context.h_results_count
  //     << std::endl;

  // // Free device variables.
  // block_context.freeAllFromCuda();

  return accelerator::ACC_OK;
}

}  // namespace ruda
