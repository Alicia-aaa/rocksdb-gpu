
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
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
__global__
void rudaRecordFilterKernel(// Parameters (ReadOnly)
                            size_t offset, size_t kSize,
                            size_t dataSize, size_t maxCacheSize,
                            char *data, uint64_t *seek_indices,
                            RudaSchema *schema,
                            uint64_t *block_seek_start_indices,
                            // Variables
                            unsigned long long int *results_idx,
                            // Results
                            RudaKVIndexPair *results);
}  // namespace kernel

struct RudaRecordBlockContext {
  cudaStream_t stream;
  cudaEvent_t kernel_finish_event;

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
  RudaKVIndexPair *d_results;                  // Filtered KV pairs
  unsigned long long int *d_results_idx;  // Atomic increment counter index

  // Cuda Results - Host
  // Total results count copied from 'd_results_idx' after kernel call...
  RudaKVIndexPair *h_results;
  unsigned long long int *h_results_count;

  // Max cached datablocks size on same gpu block (For using SharedMemory)
  size_t kMaxCacheSize = 0;

  // Allocated offsets on gpu
  size_t seek_start_offset, seek_size, datablocks_start_offset, datablocks_size;

  uint64_t *d_gpu_block_seek_starts;
  uint64_t *gpu_block_seek_starts;

  // Log
  size_t total_gpu_used_memory = 0;

  RudaRecordBlockContext(const size_t total_size, const int block_size,
                         const int grid_size, const size_t max_results_count,
                         const int stream_count, const int stream_size,
                         const int grid_size_per_stream)
      : kSize(total_size), kBlockSize(block_size), kGridSize(grid_size),
        kMaxResultsCount(max_results_count), kStreamCount(stream_count),
        kStreamSize(stream_size), kGridSizePerStream(grid_size_per_stream) {
    cudaCheckError(cudaHostAlloc(
      (void **) &gpu_block_seek_starts,
      sizeof(uint64_t) * kGridSizePerStream, cudaHostAllocMapped));
    cudaCheckError(cudaMalloc(
        (void **) &d_results_idx, sizeof(unsigned long long int)));
    total_gpu_used_memory += sizeof(unsigned long long int);
    kApproxResultsCount = kMaxResultsCount / (kStreamCount - 1);
    cudaCheckError(cudaMalloc(
        (void **) &d_results, sizeof(RudaKVIndexPair) * kApproxResultsCount));
    total_gpu_used_memory += sizeof(RudaKVIndexPair) * kApproxResultsCount;
    cudaCheckError(cudaHostAlloc(
        (void **) &h_results, sizeof(RudaKVIndexPair) * kApproxResultsCount,
        cudaHostAllocMapped));
    cudaCheckError(cudaHostAlloc(
        (void **) &h_results_count, sizeof(unsigned long long int),
        cudaHostAllocMapped));
    cudaCheckError( cudaEventCreate(&kernel_finish_event) );
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
                     RudaSchema *d_schema) {
    kernel::rudaRecordFilterKernel<<<kGridSizePerStream,
                                     kBlockSize,
                                     kMaxCacheSize,
                                     stream>>>(
      seek_start_offset, kSize, kTotalDataSize, kMaxCacheSize,
      d_datablocks, d_seek_indices, d_schema, d_gpu_block_seek_starts,
      d_results_idx, d_results
    );
  }

  void copyFromCuda() {
    cudaCheckError(cudaMemcpyAsync(
        h_results_count, d_results_idx, sizeof(unsigned long long int),
        cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(
        h_results, d_results, sizeof(RudaKVIndexPair) * kApproxResultsCount,
        cudaMemcpyDeviceToHost, stream));
    cudaCheckError( cudaEventRecord(kernel_finish_event, stream) );
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
    cudaCheckError( cudaEventDestroy(kernel_finish_event) );
    cudaCheckError( cudaFreeHost(gpu_block_seek_starts) );
    cudaCheckError( cudaFreeHost(h_results) );
    cudaCheckError( cudaFreeHost(h_results_count) );
  }
};

struct RudaRecordBlockManager {
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
  std::vector<RudaRecordBlockContext> stream_ctxs;

  // Parameters
  char *d_datablocks;
  uint64_t *d_seek_indices;
  RudaSchema h_schema;
  RudaSchema *d_schema;

  // Log
  size_t total_gpu_used_memory = 0;

  RudaRecordBlockManager(const size_t total_size, const int block_size,
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
                            rocksdb::SlicewithSchema &schema) {
    cudaCheckError(cudaHostRegister(
        &datablocks[0], sizeof(char) * datablocks.size(), cudaHostAllocMapped));
    cudaCheckError(cudaHostRegister(
        &seek_indices[0], sizeof(uint64_t) * seek_indices.size(),
        cudaHostAllocMapped));
    cudaCheckError(cudaHostRegister(
        &schema, sizeof(rocksdb::SlicewithSchema), cudaHostAllocMapped));
  }

  void unregisterPinnedMemory(std::vector<char> &datablocks,
                              std::vector<uint64_t> &seek_indices,
                              rocksdb::SlicewithSchema &schema) {
    cudaCheckError( cudaHostUnregister(&datablocks[0]) );
    cudaCheckError( cudaHostUnregister(&seek_indices[0]) );
    cudaCheckError( cudaHostUnregister(&schema) );
  }

  void initParams(const std::vector<char> &datablocks,
                  const std::vector<uint64_t> &seek_indices) {
    uint64_t start = 0;
    uint64_t start_datablocks = seek_indices[start];
    for (size_t i = 0; i < kStreamCount; ++i) {
      auto &ctx = stream_ctxs[i];

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
                      const rocksdb::SlicewithSchema &schema) {
    // Allocation Part
    // Cuda Parameters
    cudaCheckError(cudaMalloc(
        (void **) &d_datablocks, sizeof(char) * datablocks.size()));
    total_gpu_used_memory += sizeof(char) * datablocks.size();
    cudaCheckError(cudaMalloc(
        (void **) &d_seek_indices, sizeof(uint64_t) * kSize));
    total_gpu_used_memory += sizeof(uint64_t) * kSize;
    cudaCheckError( cudaMalloc((void **) &d_schema, sizeof(RudaSchema)) );
    total_gpu_used_memory += sizeof(RudaSchema);
    for (auto &ctx : stream_ctxs) {
      ctx.cudaMallocGpuBlockSeekStarts();
      total_gpu_used_memory += ctx.total_gpu_used_memory;
    }

    cudaCheckError( h_schema.populateToCuda(schema) );
    cudaCheckError(cudaMemcpy(
        d_schema, &h_schema, sizeof(RudaSchema), cudaMemcpyHostToDevice));

    // Asynchronous memory copying
    for (auto &ctx : stream_ctxs) {
      ctx.initializeStream();
    }

    // Copies sources to GPU (datablocks, seek_indices)
    // Accelerated by stream-pipelining...
    for (auto &ctx : stream_ctxs) {
      ctx.populateToCuda(
          datablocks, seek_indices, d_datablocks, d_seek_indices);
    }
  }

  void executeKernels(size_t kTotalDataSize) {
    for (auto &ctx : stream_ctxs) {
      ctx.executeKernel(
          // Parameters
          kTotalDataSize,
          // Sources
          d_datablocks, d_seek_indices, d_schema);
    }
  }

  void copyFromCuda() {
    for (auto &ctx : stream_ctxs) {
      ctx.copyFromCuda();
    }
  }

  void _translatePairsToSlices(RudaRecordBlockContext &ctx,
                               std::vector<char> &datablocks,
                               std::vector<rocksdb::Slice> &sub_values) {
    unsigned long long int count = *ctx.h_results_count;
    for (size_t i = 0; i < count; ++i) {
      RudaKVIndexPair &result = ctx.h_results[i];
      size_t value_size =
          result.value_index_.end_ - result.value_index_.start_;
      sub_values.emplace_back(
          &datablocks[0] + result.value_index_.start_, value_size);
    }
  }

  void translatePairsToSlices(std::vector<char> &datablocks,
                              std::vector<rocksdb::PinnableSlice> &values) {
    std::chrono::high_resolution_clock::time_point begin, end;

    begin = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> workers;
    std::vector< std::vector<rocksdb::Slice> > sub_values_arr(kStreamCount);

    auto worker_func = [&, this](
        RudaRecordBlockContext &ctx,
        std::vector<char> &datablocks,
        std::vector<rocksdb::Slice> &sub_values) {
      cudaCheckError( cudaEventSynchronize(ctx.kernel_finish_event) );
      this->_translatePairsToSlices(ctx, datablocks, sub_values);
    };

    for (size_t i = 0; i < kStreamCount; ++i) {
      std::vector<rocksdb::Slice> &sub_values = sub_values_arr[i];
      workers.emplace_back(
          worker_func, std::ref(stream_ctxs[i]), std::ref(datablocks),
          std::ref(sub_values));
    }

    for (size_t i = 0; i < kStreamCount; ++i) {
      workers[i].join();
    }

    for (size_t i = 0; i < kStreamCount; ++i) {
      std::vector<rocksdb::Slice> &sub_values = sub_values_arr[i];
      for (auto &sub_value : sub_values) {
        values.emplace_back(std::move(rocksdb::PinnableSlice(
            sub_value.data_, sub_value.size_)));
      }
    }

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - begin;
    std::cout << "[GPU][translatePairsToSlices] Execution Time: "
        << elapsed.count() << std::endl;
  }

  void log() {
    std::cout << "[CUDA][RudaRecordBlockManager]" << std::endl
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

    std::cout << "RecordBlockContexts" << std::endl;
    for (size_t i = 0; i < kStreamCount; ++i) {
      auto &ctx = stream_ctxs[i];
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

  void clear() {
    cudaCheckError( cudaFree(d_datablocks) );
    cudaCheckError( cudaFree(d_seek_indices) );
    cudaCheckError( h_schema.clear() );
    cudaCheckError( cudaFree(d_schema) );
    for (auto &ctx : stream_ctxs) {
      ctx.clear();
    }
    stream_ctxs.clear();
  }
};

__global__
void kernel::rudaRecordFilterKernel(// Parameters (ReadOnly)
                                    size_t offset, size_t kSize,
                                    size_t dataSize, size_t maxCacheSize,
                                    char *data, uint64_t *seek_indices,
                                    RudaSchema *schema,
                                    uint64_t *block_seek_start_indices,
                                    // Variables
                                    unsigned long long int *results_idx,
                                    // Results
                                    RudaKVIndexPair *results) {
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
  DecodeNFilterOnSchema(
      // Parameters
      cached_data, size, block_seek_start_index, start, end, schema,
      // Results
      results_idx, results);
}

int recordBlockFilter(/* const */ std::vector<char> &datablocks,
                      /* const */ std::vector<uint64_t> &seek_indices,
                      const rocksdb::SlicewithSchema &schema,
                      const size_t max_results_count,
                      std::vector<rocksdb::PinnableSlice> &values) {
  std::cout << "[GPU][recordBlockFilter] START" << std::endl;
  if (seek_indices.size() < 256) {
    // Not allowed small size seek_indices... (Meaningless on GPU)
    return accelerator::ACC_ERR;
  }

  // Warming up
  // Note(totoro): Because, there is a warming up latency on gpu when
  // gpu-related function called(ex. set up gpu driver). So, we ignore this
  // latency by just firing meaningless malloc function.
  void *warming_up;
  cudaCheckError(cudaMalloc(&warming_up, 0));
  cudaCheckError(cudaFree(warming_up));

  // Cuda can't use const variable. So we copy SlicewithSchema. (Shallow copy)
  rocksdb::SlicewithSchema* copied_schema = schema.clone();

  RudaRecordBlockManager block_mgr(
      seek_indices.size() /* kSize */,
      64 /* kBlockSize */,
      4,
      max_results_count);

  // Copy & Initializes variables from host to device.
  block_mgr.initParams(datablocks, seek_indices);
  block_mgr.log();
  block_mgr.registerPinnedMemory(datablocks, seek_indices, *copied_schema);
  // ----------------------------------------------
  // Cuda Stream Pipelined (Accelerate)
  block_mgr.populateToCuda(datablocks, seek_indices, *copied_schema);
  std::cout << "[GPU][recordBlockFilter] Total GPU used memory: "
      << (block_mgr.total_gpu_used_memory / (MB)) << "MB" << std::endl;
  block_mgr.executeKernels(datablocks.size());
  block_mgr.copyFromCuda();
  // ----------------------------------------------
  block_mgr.translatePairsToSlices(datablocks, values);
  block_mgr.unregisterPinnedMemory(datablocks, seek_indices, *copied_schema);
  block_mgr.clear();
  delete copied_schema;

  return accelerator::ACC_OK;
}

}  // namespace ruda
