#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "accelerator/cuda/async_manager.h"
#include "accelerator/cuda/filter.h"
#include "accelerator/cuda/block_decoder.h"
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
void rudaAsyncFilterKernel(// Parameters (ReadOnly)
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

void incr(RudaAsyncBlockContext * ruda_block) {
    ruda_block->mt_->lock();
    *(ruda_block->kComplete_)++;
    ruda_block->mt_->unlock();    
}

void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void * ruda_block) {
    std::thread thd(incr, (RudaAsyncBlockContext *)ruda_block);
    thd.join();    
}

RudaAsyncBlockContext::RudaAsyncBlockContext(const size_t total_size, const int block_size,
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

void RudaAsyncBlockContext::cudaMallocGpuBlockSeekStarts() {
    cudaCheckError(cudaMalloc(
        (void **) &d_gpu_block_seek_starts,
        sizeof(uint64_t) * kGridSizePerStream));
    total_gpu_used_memory += sizeof(uint64_t) * kGridSizePerStream;
}

size_t RudaAsyncBlockContext::calculateGpuBlockSeekStarts(std::vector<char> &datablocks,
                                     std::vector<uint64_t> &seek_indices,
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

void RudaAsyncBlockContext::initParams(std::vector<char> &datablocks,
                  std::vector<uint64_t> &seek_indices,
                  size_t start, size_t size, size_t start_datablocks,
                  size_t size_datablocks, std::mutex * mt) {
    seek_start_offset = start;
    seek_size = size;
    datablocks_start_offset = start_datablocks;
    datablocks_size = size_datablocks;
    kMaxCacheSize = calculateGpuBlockSeekStarts(
        datablocks, seek_indices, start, size);
    mt_ = mt;
}

void RudaAsyncBlockContext::populateToCuda(std::vector<char> &datablocks,
                      std::vector<uint64_t> &seek_indices,
                      char *d_datablocks, uint64_t *d_seek_indices) {
    populateToCuda_d_results_idx();
    populateToCuda_d_datablocks(datablocks, d_datablocks);
    populateToCuda_d_seek_indices(seek_indices, d_seek_indices);
    populateToCuda_d_gpu_block_seek_starts();
}

void RudaAsyncBlockContext::populateToCuda_d_results_idx() {
    cudaCheckError(cudaMemsetAsync(
        d_results_idx, 0, sizeof(unsigned long long int), *stream));
}

void RudaAsyncBlockContext::populateToCuda_d_datablocks(std::vector<char> &datablocks,
                                   char *d_datablocks) {
    cudaCheckError(cudaMemcpyAsync(
        &d_datablocks[datablocks_start_offset],
        &datablocks[datablocks_start_offset],
        sizeof(char) * datablocks_size, cudaMemcpyHostToDevice,
        *stream));
}

void RudaAsyncBlockContext::populateToCuda_d_seek_indices(std::vector<uint64_t> &seek_indices,
                                     uint64_t *d_seek_indices) {
    cudaCheckError(cudaMemcpyAsync(
        &d_seek_indices[seek_start_offset], &seek_indices[seek_start_offset],
        sizeof(uint64_t) * seek_size, cudaMemcpyHostToDevice,
        *stream));
}

void RudaAsyncBlockContext::populateToCuda_d_gpu_block_seek_starts() {
    cudaCheckError(cudaMemcpyAsync(
        d_gpu_block_seek_starts, gpu_block_seek_starts,
        sizeof(uint64_t) * kGridSizePerStream, cudaMemcpyHostToDevice,
        *stream));
}

void RudaAsyncBlockContext::executeKernel(// Kernel Parameter
                     size_t kTotalDataSize,
                     // Sources
                     char *d_datablocks, uint64_t *d_seek_indices,
                     RudaSchema *d_schema) {
    kernel::rudaAsyncFilterKernel<<<kGridSizePerStream,
                                     kBlockSize,
                                     kMaxCacheSize,
                                     *stream>>>(
      seek_start_offset, kSize, kTotalDataSize, kMaxCacheSize,
      d_datablocks, d_seek_indices, d_schema, d_gpu_block_seek_starts,
      d_results_idx, d_results
    );
}

void RudaAsyncBlockContext::copyFromCuda(uint64_t * kComplete) {
    this->kComplete_ = kComplete;
    cudaCheckError(cudaMemcpyAsync(
        h_results_count, d_results_idx, sizeof(unsigned long long int),
        cudaMemcpyDeviceToHost, *stream));
    cudaCheckError(cudaMemcpyAsync(
        h_results, d_results, sizeof(RudaKVIndexPair) * kApproxResultsCount,
        cudaMemcpyDeviceToHost, *stream));
    cudaCheckError( cudaStreamAddCallback(*stream, callback, this, 0) );
}

void RudaAsyncBlockContext::freeCudaObjects() {
    cudaCheckError( cudaFree(d_gpu_block_seek_starts) );
    cudaCheckError( cudaFree(d_results_idx) );
    cudaCheckError( cudaFree(d_results) );
}

void RudaAsyncBlockContext::populateStream(cudaStream_t * allocated_stream) {
    stream = allocated_stream;     
}

void RudaAsyncBlockContext::destroyStream() {
    cudaCheckError(cudaStreamDestroy(*stream) );
}

void RudaAsyncBlockContext::clear() {
    freeCudaObjects();
    //destroyStream();
    cudaCheckError( cudaEventDestroy(kernel_finish_event) );
    cudaCheckError( cudaFreeHost(gpu_block_seek_starts) );
    cudaCheckError( cudaFreeHost(h_results) );
    cudaCheckError( cudaFreeHost(h_results_count) );
}

RudaAsyncManager::RudaAsyncManager(RudaSchema * d_schema, std::mutex * ml, cudaStream_t * cuda_stream) {
    ml_ = ml;
    d_schema_ = d_schema;
    cuda_stream_ = cuda_stream;
    
}

void RudaAsyncManager::calStreamContext(const size_t total_size, const int block_size,
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

void RudaAsyncManager::registerPinnedMemory(std::vector<char> &datablocks,
                            std::vector<uint64_t> &seek_indices) {
    cudaCheckError(cudaHostRegister(
        &datablocks[0], sizeof(char) * datablocks.size(), cudaHostAllocMapped));
    cudaCheckError(cudaHostRegister(
        &seek_indices[0], sizeof(uint64_t) * seek_indices.size(),
        cudaHostAllocMapped));
}

void RudaAsyncManager::unregisterPinnedMemory(std::vector<char> &datablocks,
                              std::vector<uint64_t> &seek_indices) {
    cudaCheckError( cudaHostUnregister(&datablocks[0]) );
    cudaCheckError( cudaHostUnregister(&seek_indices[0]) );
}

void RudaAsyncManager::initParams(std::vector<char> &datablocks,
                  std::vector<uint64_t> &seek_indices, int join_idx) {
    h_datablocks = &datablocks;
    h_seek_indices = &seek_indices;
    join_idx_ = join_idx;
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
          size_datablocks, ml_);

      start += ctx.kStreamSize;
      if (start >= kSize) break;
      start_datablocks = seek_indices[start];
    }
}

void RudaAsyncManager::populateToCuda(std::vector<char> &datablocks,
                      std::vector<uint64_t> &seek_indices) {
    // Allocation Part
    // Cuda Parameters
    cudaCheckError(cudaMalloc(
        (void **) &d_datablocks, sizeof(char) * datablocks.size()));
    total_gpu_used_memory += sizeof(char) * datablocks.size();
    cudaCheckError(cudaMalloc(
        (void **) &d_seek_indices, sizeof(uint64_t) * kSize));
    total_gpu_used_memory += sizeof(uint64_t) * kSize;
 
    for (auto &ctx : stream_ctxs) {
      ctx.cudaMallocGpuBlockSeekStarts();
      total_gpu_used_memory += ctx.total_gpu_used_memory;
    }

    // Asynchronous memory copying
    for (auto &ctx : stream_ctxs) {
      uint64_t idx = 0;
      ctx.populateStream(&(cuda_stream_[idx++]));
    }

    // Copies sources to GPU (datablocks, seek_indices)
    // Accelerated by stream-pipelining...
    for (auto &ctx : stream_ctxs) {
      ctx.populateToCuda(
          datablocks, seek_indices, d_datablocks, d_seek_indices);
    }
}

void RudaAsyncManager::executeKernels(size_t kTotalDataSize, int join_idx) {
    for (auto &ctx : stream_ctxs) {
      ctx.executeKernel(
          // Parameters
          kTotalDataSize,
          // Sources
          d_datablocks, d_seek_indices, d_schema_ + join_idx);
    }
}

void RudaAsyncManager::copyFromCuda() {
    for (auto &ctx : stream_ctxs) {
      ctx.copyFromCuda(&kComplete);
    }
}

void RudaAsyncManager::_translatePairsToSlices(RudaAsyncBlockContext &ctx,
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

void RudaAsyncManager::translatePairsToSlices(std::vector<char> &datablocks,
                              std::vector<rocksdb::PinnableSlice> &values) {
    std::chrono::high_resolution_clock::time_point begin, end;

    begin = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> workers;
    std::vector< std::vector<rocksdb::Slice> > sub_values_arr(kStreamCount);

    auto worker_func = [&, this](
        RudaAsyncBlockContext &ctx,
        std::vector<char> &datablocks,
        std::vector<rocksdb::Slice> &sub_values) {
      cudaCheckError( cudaEventSynchronize(ctx.kernel_finish_event));
      //cudaCheckError(cudaEventSynchronize(kernel_finish_event));
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

void RudaAsyncManager::log() {
    std::cout << "[CUDA][RudaAsyncBlockManager]" << std::endl
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

void RudaAsyncManager::clear() {
    cudaCheckError( cudaFree(d_datablocks) );
    cudaCheckError( cudaFree(d_seek_indices) );
    //cudaCheckError( h_schema->clear() );
    //cudaCheckError( cudaFree(d_schema) );
    for (auto &ctx : stream_ctxs) {
      ctx.clear();
    }
    stream_ctxs.clear();
}

void RudaAsyncManager::release() {
//    cudaCheckError( h_schema->clear() );
    cudaCheckError( cudaFree(d_schema_) );
//    for(uint64_t i = 0; i < 3; ++i) {
//        cudaCheckError(cudaStreamDestroy(streams[i]));
//    }
}

__global__
void kernel::rudaAsyncFilterKernel(// Parameters (ReadOnly)
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

int recordAsyncFilter(/* const */ std::vector<char> &datablocks,
                      /* const */ std::vector<uint64_t> &seek_indices,
                      int join_idx,              
                      const size_t max_results_count,
                      std::vector<rocksdb::PinnableSlice> &results,
                      ruda::RudaAsyncManager * async_mgr) {
  std::cout << "[GPU][recordAsyncFilter] START" << std::endl;

  async_mgr->calStreamContext(seek_indices.size(), 64, 4, max_results_count);
  async_mgr->initParams(datablocks, seek_indices, join_idx);
  async_mgr->log();
  async_mgr->registerPinnedMemory(datablocks, seek_indices);
  // ----------------------------------------------
  // Cuda Stream Pipelined (Accelerate)
  async_mgr->populateToCuda(datablocks, seek_indices);
  async_mgr->executeKernels(datablocks.size(), join_idx);
  async_mgr->copyFromCuda();
  // ----------------------------------------------
  // callback
//  async_mgr->translatePairsToSlices(datablocks, results);
//  async_mgr->unregisterPinnedMemory(datablocks, seek_indices);
//  async_mgr->clear();
  return accelerator::ACC_OK;
}

int releaseAsyncManager(ruda::RudaAsyncManager * async_manager) {
    async_manager->release();
    return accelerator::ACC_OK;
}

int initializeGlobal(std::vector<rocksdb::SlicewithSchema> &schema, cudaStream_t * cuda_stream, uint64_t stream_num, ruda::RudaSchema * d_schema) {
    
    uint64_t table_num = schema.size();
    void *warming_up;
    cudaCheckError(cudaMalloc(&warming_up, 0));
    cudaCheckError(cudaFree(warming_up));
    
    cudaCheckError(cudaMalloc((void **) &d_schema, table_num * sizeof(RudaSchema)));
    for(uint64_t i = 0; i < table_num; ++i) {
        RudaSchema h_schema;
        cudaCheckError(h_schema.populateToCuda(schema[i]));
        cudaCheckError(cudaMemcpy(d_schema + i, &h_schema, sizeof(RudaSchema), cudaMemcpyHostToDevice));
    }
    for(uint64_t i = 0; i < stream_num; ++i) {
        cudaCheckError(cudaStreamCreateWithFlags(&(cuda_stream[i]), cudaStreamNonBlocking));
    }
    return accelerator::ACC_OK;
}

bool capacityCheck() {
    size_t free_byte;
    size_t total_byte;
    
    cudaMemGetInfo(&free_byte, &total_byte);
    
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    
    return free_db > (total_db / 10);
}

}  // namespace ruda
