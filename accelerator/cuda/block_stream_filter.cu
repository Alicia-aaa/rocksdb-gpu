
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "accelerator/cuda/block_decoder.h"
#include "accelerator/cuda/filter.h"
#include "rocksdb/slice.h"
#include "table/format.h"

#define KB 1024
#define MB 1024 * KB
#define GB 1024 * MB

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
                                    size_t dataSize, size_t maxCacheSize,
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

  // Cuda Kernel Parameters
  const size_t kSize = 0;             // Total seek indices count
  const int kBlockSize = 0;
  const int kGridSize = 0;
  const size_t kMaxResultsCount = 0;  // Total count of filtered Key-Value pairs
  size_t kApproxResultsCount = 0;

  // Cuda Kernel Parameters - stream
  const int kStreamCount = 0;
  const int kStreamSize = 0;
  const int kGridSizePerStream = 0;

  // Cuda Results - Device
  RudaKVPair *d_results;                  // Filtered KV pairs
  unsigned long long int *d_results_idx;  // Atomic increment counter index

  // Cuda Results - Host
  // Total results count copied from 'd_results_idx' after kernel call...
  RudaKVPair *h_results;
  unsigned long long int h_results_count;

  // Max cached datablocks size on same gpu block (For using SharedMemory)
  size_t kMaxCacheSize = 0;

  // Allocated offsets on gpu
  size_t seek_start_offset, seek_size, datablocks_start_offset, datablocks_size;

  uint64_t *d_gpu_block_seek_starts;
  uint64_t *gpu_block_seek_starts;

  // Log
  size_t total_gpu_used_memory = 0;

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
    cudaCheckError(cudaMalloc(
        (void **) &d_results_idx, sizeof(unsigned long long int)));
    total_gpu_used_memory += sizeof(unsigned long long int);
    kApproxResultsCount = kMaxResultsCount / (kStreamCount - 1);
    cudaCheckError(cudaMalloc(
        (void **) &d_results, sizeof(RudaKVPair) * kApproxResultsCount));
    total_gpu_used_memory += sizeof(RudaKVPair) * kApproxResultsCount;
    cudaCheckError(cudaMallocHost(
        (void **) &h_results, sizeof(RudaKVPair) * kApproxResultsCount));
  }

  void cudaMallocGpuBlockSeekStarts() {
    cudaCheckError(cudaMalloc(
        (void **) &d_gpu_block_seek_starts,
        sizeof(uint64_t) * kGridSizePerStream));
    total_gpu_used_memory += sizeof(uint64_t) * kGridSizePerStream;
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
      size_t thread_idx = start + i * kBlockSize;
      if (thread_idx >= kSize) {
        break;
      }

      size_t cache_size;
      if (start + size == kSize) {
        // Last Stream case
        size_t next_block_thread_idx = start + (i + 1) * kBlockSize;
        if (next_block_thread_idx >= kSize) {
          cache_size = datablocks.size() - gpu_block_seek_starts[i];
        } else {
          cache_size = gpu_block_seek_starts[i+1] - gpu_block_seek_starts[i];
        }
      } else {
        // Non-last Stream case
        if (i == kGridSizePerStream - 1) {
          cache_size = seek_indices[start + size] - gpu_block_seek_starts[i];
        } else {
          cache_size = gpu_block_seek_starts[i+1] - gpu_block_seek_starts[i];
        }
      }

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
                      char *d_datablocks, uint64_t *d_seek_indices) {
    populateToCuda_d_results_idx();
    populateToCuda_d_datablocks(datablocks, d_datablocks);
    populateToCuda_d_seek_indices(seek_indices, d_seek_indices);
    populateToCuda_d_gpu_block_seek_starts();
  }

  void populateToCuda_d_results_idx() {
    cudaCheckError(cudaMemsetAsync(
        d_results_idx, 0, sizeof(unsigned long long int), stream));
  }

  void populateToCuda_d_datablocks(const std::vector<char> &datablocks,
                                   char *d_datablocks) {
    cudaCheckError(cudaMemcpyAsync(
        &d_datablocks[datablocks_start_offset],
        &datablocks[datablocks_start_offset],
        sizeof(char) * datablocks_size, cudaMemcpyHostToDevice,
        stream));
  }

  void populateToCuda_d_seek_indices(const std::vector<uint64_t> &seek_indices,
                                     uint64_t *d_seek_indices) {
    cudaCheckError(cudaMemcpyAsync(
        &d_seek_indices[seek_start_offset], &seek_indices[seek_start_offset],
        sizeof(uint64_t) * seek_size, cudaMemcpyHostToDevice,
        stream));
  }

  void populateToCuda_d_gpu_block_seek_starts() {
    cudaCheckError(cudaMemcpyAsync(
        d_gpu_block_seek_starts, gpu_block_seek_starts,
        sizeof(uint64_t) * kGridSizePerStream, cudaMemcpyHostToDevice,
        stream));
  }

  void executeKernel(// Kernel Parameter
                     size_t kTotalDataSize,
                     // Sources
                     char *d_datablocks, uint64_t *d_seek_indices,
                     accelerator::FilterContext *d_cond_ctx) {
    kernel::rudaStreamIntBlockFilterKernel<<<kGridSizePerStream,
                                             kBlockSize,
                                             kMaxCacheSize,
                                             stream>>>(
      seek_start_offset, kSize, kTotalDataSize, kMaxCacheSize,
      d_datablocks, d_seek_indices, d_cond_ctx, d_gpu_block_seek_starts,
      d_results_idx, d_results
    );
  }

  void copyFromCuda() {
    cudaCheckError(cudaMemcpyAsync(
        &h_results_count, d_results_idx, sizeof(unsigned long long int),
        cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(
        h_results, d_results, sizeof(RudaKVPair) * h_results_count,
        cudaMemcpyDeviceToHost, stream));
  }

  void freeCudaObjects() {
    cudaCheckError( cudaFree(d_gpu_block_seek_starts) );
    cudaCheckError( cudaFree(d_results_idx) );
    cudaCheckError( cudaFree(d_results) );
  }

  void initializeStream() {
    cudaCheckError( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
  }

  void destroyStream() {
    cudaCheckError( cudaStreamDestroy(stream) );
  }

  void clear() {
    freeCudaObjects();
    destroyStream();
    cudaCheckError( cudaFreeHost(gpu_block_seek_starts) );
    cudaCheckError( cudaFreeHost(h_results) );
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
  int kApproxStreamSize = 0;
  int kApproxGridSizePerStream = 0;

  // Streams
  std::vector<RudaBlockStreamContext> stream_ctxs;

  // Parameters
  char *d_datablocks;
  uint64_t *d_seek_indices;
  accelerator::FilterContext *d_cond_ctx;

  // Log
  size_t total_gpu_used_memory = 0;

  RudaBlockStreamManager(const size_t total_size, const int block_size,
                         const size_t stream_count,
                         const size_t max_results_count) {
    kSize = total_size;
    kBlockSize = block_size;
    kStreamCount = stream_count;
    kMaxResultsCount = max_results_count;
    size_t threads_per_stream = ceil((float) total_size / (float) stream_count);
    while (threads_per_stream <= kBlockSize && kBlockSize != 4) {
      kBlockSize = kBlockSize >> 1;
    }
    kGridSize = ceil((float) total_size / (float) kBlockSize);

    // Stream grid pre-process
    // ex) kSize = 672, kGridSize = 11, kBlockSize = 64, kStreamCount = 4
    // --> kApproxGridSizePerStream = 2, kApproxStreamSize = 128
    //
    // Stream1   { kGridSizePerStream = 2, kStreamSize = 128 }
    // Stream2   { kGridSizePerStream = 3, kStreamSize = 192 }
    // Stream3   { kGridSizePerStream = 3, kStreamSize = 192 }
    // Stream4   { kGridSizePerStream = 3, kStreamSize = 192 (actually 160) }
    // -----Results-----
    // <Stream1> <Stream2>    <Stream3>     <Stream4>
    // [64][64]  [64][64][64] [64][64][64]  [64][64][32]
    kApproxGridSizePerStream = kGridSize / kStreamCount;
    size_t additional_grid_count = kGridSize % kStreamCount;
    kApproxStreamSize = kApproxGridSizePerStream * kBlockSize;
    for (size_t i = 0; i < kStreamCount; ++i) {
      size_t grid_size_per_stream, stream_size;
      if (i >= kStreamCount - additional_grid_count) {
        grid_size_per_stream = kApproxGridSizePerStream + 1;
        stream_size = grid_size_per_stream * kBlockSize;
      } else {
        grid_size_per_stream = kApproxGridSizePerStream;
        stream_size = kApproxStreamSize;
      }
      stream_ctxs.emplace_back(
          kSize, kBlockSize, kGridSize, kMaxResultsCount, kStreamCount,
          stream_size, grid_size_per_stream);
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
    uint64_t start = 0;
    uint64_t start_datablocks = seek_indices[start];
    for (size_t i = 0; i < kStreamCount; ++i) {
      RudaBlockStreamContext &ctx = stream_ctxs[i];

      uint64_t size, size_datablocks;
      if (i == kStreamCount - 1) {
        size = kSize - start;
        size_datablocks = datablocks.size() - start_datablocks;
      } else {
        size = ctx.kStreamSize;
        size_datablocks = seek_indices[start + size] - start_datablocks;
      }

      // Copies sources to GPU (datablocks, seek_indices)
      // Accelerated by stream-pipelining...
      ctx.initParams(
          datablocks, seek_indices, start, size, start_datablocks,
          size_datablocks);

      start += ctx.kStreamSize;
      if (start >= kSize) break;
      start_datablocks = seek_indices[start];
    }
  }

  void populateToCuda(const std::vector<char> &datablocks,
                      const std::vector<uint64_t> &seek_indices,
                      const accelerator::FilterContext &cond_ctx) {
    // Allocation Part
    // Cuda Parameters
    cudaCheckError(cudaMalloc(
        (void **) &d_datablocks, sizeof(char) * datablocks.size()));
    total_gpu_used_memory += sizeof(char) * datablocks.size();
    cudaCheckError(cudaMalloc(
        (void **) &d_seek_indices, sizeof(uint64_t) * kSize));
    total_gpu_used_memory += sizeof(uint64_t) * kSize;
    cudaCheckError(cudaMalloc(
        (void **) &d_cond_ctx, sizeof(accelerator::FilterContext)));
    total_gpu_used_memory += sizeof(accelerator::FilterContext);
    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.cudaMallocGpuBlockSeekStarts();
      total_gpu_used_memory += ctx.total_gpu_used_memory;
    }

    cudaCheckError(cudaMemcpy(
        d_cond_ctx, &cond_ctx, sizeof(accelerator::FilterContext),
        cudaMemcpyHostToDevice));

    // Asynchronous memory copying
    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.initializeStream();
    }

    // Copies sources to GPU (datablocks, seek_indices)
    // Accelerated by stream-pipelining...
    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.populateToCuda_d_results_idx();
    }

    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.populateToCuda_d_datablocks(datablocks, d_datablocks);
    }

    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.populateToCuda_d_seek_indices(seek_indices, d_seek_indices);
    }

    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.populateToCuda_d_gpu_block_seek_starts();
    }
  }

  void executeKernels(size_t kTotalDataSize) {
    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.executeKernel(
          // Parameters
          kTotalDataSize,
          // Sources
          d_datablocks, d_seek_indices, d_cond_ctx);
    }
  }

  void copyFromCuda() {
    for (RudaBlockStreamContext &ctx : stream_ctxs) {
      ctx.copyFromCuda();
    }
  }

  void translatePairsToSlices(std::vector<rocksdb::Slice> &keys,
                              std::vector<rocksdb::Slice> &values) {
    for (auto &ctx : stream_ctxs) {
      cudaCheckError( cudaStreamSynchronize(ctx.stream) );
      for (size_t i = 0; i < ctx.h_results_count; ++i) {
        RudaKVPair &result = ctx.h_results[i];
        size_t key_size = result.key()->size_;
        size_t value_size = result.value()->size_;
        char *key = new char[key_size];
        char *value = new char[value_size];
        memcpy(
            key, result.key()->stack_data_, sizeof(char) * key_size);
        memcpy(
            value, result.value()->stack_data_, sizeof(char) * value_size);
        keys.emplace_back(key, key_size);
        values.emplace_back(value, value_size);
      }
    }
  }

  void log() {
    std::cout << "[CUDA][BlockStreamManager]" << std::endl
        << "kSize: " << kSize << std::endl
        << "kGridSize: " << kGridSize << std::endl
        << "kBlockSize: " << kBlockSize << std::endl
        << "kStreamCount: " << kStreamCount << std::endl
        << "kApproxGridSizePerStream: "
            << kApproxGridSizePerStream << std::endl
        << "kApproxStreamSize: "
            << kApproxStreamSize << std::endl
        << "Max Results Count: " << kMaxResultsCount << std::endl
        << "======================" << std::endl;

    std::cout << "BlockStreamContexts" << std::endl;
    for (size_t i = 0; i < kStreamCount; ++i) {
      RudaBlockStreamContext &ctx = stream_ctxs[i];
      std::cout << "Stream: " << i << std::endl
          << "kStreamSize: " << ctx.kStreamSize << std::endl
          << "kGridSizePerStream: " << ctx.kGridSizePerStream << std::endl
          << "Start Offset: " << ctx.seek_start_offset << std::endl
          << "Size: " << ctx.seek_size << std::endl
          << "Start DataBlocks: " << ctx.datablocks_start_offset << std::endl
          << "Size DataBlocks: " << ctx.datablocks_size << std::endl
          << "_____" << std::endl
          << "Max cache size: " << ctx.kMaxCacheSize << std::endl;
      for (size_t j = 0; j < ctx.kGridSizePerStream; ++j) {
        std::cout << "GPU Block Seek Start[" << j << "]: "
            << ctx.gpu_block_seek_starts[j] << std::endl;
      }
      std::cout << "-----------" << std::endl;
    }
  }

  void freeCudaObjects() {
    cudaCheckError( cudaFree(d_datablocks) );
    cudaCheckError( cudaFree(d_seek_indices) );
    cudaCheckError( cudaFree(d_cond_ctx) );
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
                                            size_t maxCacheSize,
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
    if (data_idx >= dataSize || j >= maxCacheSize) {
      break;
    }
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
  // block_stream_mgr.log();

  block_stream_mgr.registerPinnedMemory(datablocks, seek_indices, context);
  // ----------------------------------------------
  // Cuda Stream Pipelined (Accelerate)
  block_stream_mgr.populateToCuda(datablocks, seek_indices, context);
  std::cout << "MB: " << MB << std::endl;
  std::cout << "Total GPU used memory: "
      << (block_stream_mgr.total_gpu_used_memory / (MB)) << "MB" << std::endl;
  block_stream_mgr.executeKernels(datablocks.size());
  block_stream_mgr.copyFromCuda();
  // ----------------------------------------------
  block_stream_mgr.translatePairsToSlices(keys, values);
  block_stream_mgr.unregisterPinnedMemory(datablocks, seek_indices, context);
  block_stream_mgr.clear();

  return accelerator::ACC_OK;
}

}  // namespace ruda
