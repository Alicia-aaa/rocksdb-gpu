#include <cooperative_groups.h>

#include "accelerator/cuda/block_decoder.h"
#include "accelerator/cuda/cuda_util.h"

#define NUM_TABLE_BYTES 4
#define DEFAULT_KEY_BUF_SIZE 16

namespace ruda {

// Note(totoro): This implementation copied from 'block.h', 'coding.h',
//               'coding.cc' to use on cuda codes.
// Helper routine: decode the next block entry starting at "p",
// storing the number of shared key bytes, non_shared key bytes,
// and the length of the value in "*shared", "*non_shared", and
// "*value_length", respectively.  Will not derefence past "limit".
//
// If any errors are detected, returns nullptr.  Otherwise, returns a
// pointer to the key delta (just past the three decoded values).
struct DecodeEntry {

  __device__
  const char* GetVarint32PtrFallback(const char* p, const char* limit,
                                     uint32_t* value) {
    uint32_t result = 0;
    for (uint32_t shift = 0; shift <= 28 && p < limit; shift += 7) {
      uint32_t byte = *(reinterpret_cast<const unsigned char*>(p));
      p++;
      if (byte & 128) {
        // More bytes are present
        result |= ((byte & 127) << shift);
      } else {
        result |= (byte << shift);
        *value = result;
        return reinterpret_cast<const char*>(p);
      }
    }
    return nullptr;
  }

  __device__
  const char* GetVarint32Ptr(const char* p,
                             const char* limit,
                             uint32_t* value) {
    if (p < limit) {
      uint32_t result = *(reinterpret_cast<const unsigned char*>(p));
      if ((result & 128) == 0) {
        *value = result;
        return p + 1;
      }
    }
    return GetVarint32PtrFallback(p, limit, value);
  }

  __device__
  const char* operator()(const char* p, const char* limit,
                         uint32_t* shared, uint32_t* non_shared,
                         uint32_t* value_length) {
    // We need 2 bytes for shared and non_shared size. We also need one more
    // byte either for value size or the actual value in case of value delta
    // encoding.
    // assert(limit - p >= 3);
    *shared = reinterpret_cast<const unsigned char*>(p)[0];
    *non_shared = reinterpret_cast<const unsigned char*>(p)[1];
    *value_length = reinterpret_cast<const unsigned char*>(p)[2];
    if ((*shared | *non_shared | *value_length) < 128) {
      // Fast path: all three values are encoded in one byte each
      p += 3;
    } else {
      if ((p = GetVarint32Ptr(p, limit, shared)) == nullptr) return nullptr;
      if ((p = GetVarint32Ptr(p, limit, non_shared)) == nullptr) return nullptr;
      if ((p = GetVarint32Ptr(p, limit, value_length)) == nullptr) {
        return nullptr;
      }
    }

    // Using an assert in place of "return null" since we should not pay the
    // cost of checking for corruption on every single key decoding
    // assert(!(static_cast<uint32_t>(limit - p) < (*non_shared + *value_length)));
    return p;
  }
};

__host__ __device__
uint32_t DecodeFixed32(const char* ptr) {
  // if (port::kLittleEndian) {
  //   // Load the raw bytes
  //   uint32_t result;
  //   memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
  //   return result;
  // }
  return ((static_cast<uint32_t>(static_cast<unsigned char>(ptr[0])))
      | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[1])) << 8)
      | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[2])) << 16)
      | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[3])) << 24));
}

__host__ __device__
uint64_t DecodeFixed64(const char* ptr) {
  // if (port::kLittleEndian) {
  //   // Load the raw bytes
  //   uint64_t result;
  //   memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
  //   return result;
  // }
  uint64_t lo = DecodeFixed32(ptr);
  uint64_t hi = DecodeFixed32(ptr + 4);
  return (hi << 32) | lo;
}

__host__ __device__
unsigned int getFileIdx(uint32_t blockId, uint64_t size, uint64_t *g_block_index) {
  int start = 0;
  int end = size - 1;
  while ( start < end ) {
    int mid = (start + end) / 2;
    if(blockId <= g_block_index[mid] - 1) {
      end = mid;
    } else {
      start = mid + 1;
    }
  }
  return end;
}

__device__
bool memcmp(char * left, char * right, size_t length) {
  for(int i = 0; i < length - 1; i++) {
      if(left[i] != right[i]) return false;
  }
  return true;
}

__device__
unsigned long long int atomicAggInc(unsigned long long int *counter) {
  auto g = cooperative_groups::coalesced_threads();
  unsigned long long int warp_res;
  if (g.thread_rank() == 0) {
    warp_res = atomicAdd(counter, g.size());
  }
  return g.shfl(warp_res, 0) + g.thread_rank();
}

__device__
void DecodeNFilterSubDataBlocks(// Parameters
                                const char *cached_data,
                                const uint64_t cached_data_size,
                                const uint64_t block_offset,
                                const uint64_t start_idx,
                                const uint64_t end_idx,
                                accelerator::FilterContext *ctx,
                                // Results
                                unsigned long long int *results_idx,
                                ruda::RudaKVIndexPair *results) {
  const char *subblock = &cached_data[start_idx];
  const char *limit = &cached_data[end_idx];
  size_t key_buf_size = DEFAULT_KEY_BUF_SIZE;
  //size_t key_buf_length = 0;
  char *key_buf = new char[key_buf_size];
  while (subblock < limit) {
    uint32_t shared, non_shared, value_size;
    subblock = DecodeEntry()(subblock, limit, &shared, &non_shared,
                             &value_size);
    const char *key;
    size_t key_size;
    if (shared == 0) {
      key = subblock;
      key_size = non_shared;
      if (key_size > key_buf_size) {
        delete[] key_buf;
        key_buf_size = key_size;
        key_buf = new char[key_buf_size];
      }
      memset(key_buf, 0, sizeof(char) * key_buf_size);
      memcpy(key_buf, key, sizeof(char) * key_size);
      //key_buf_length = key_size;
    } else {
      key = subblock;
      key_size = shared + non_shared;
      if (key_size > key_buf_size) {
        char *new_key_buf = new char[key_size];
        memcpy(new_key_buf, key_buf, sizeof(char) * shared);
        delete[] key_buf;
        key_buf_size = key_size;
        key_buf = new_key_buf;
      }
      memcpy(key_buf + shared, key, sizeof(char) * non_shared);
      //key_buf_length = key_size;
    }

    const char *value = subblock + non_shared;
    uint64_t decoded_value = DecodeFixed64(value);
    bool filter_result = false;
    switch (ctx->_op) {
      case accelerator::EQ:
        filter_result = decoded_value == ctx->_pivot;
        break;
      case accelerator::LESS:
        filter_result = decoded_value < ctx->_pivot;
        break;
      case accelerator::GREATER:
        filter_result = decoded_value > ctx->_pivot;
        break;
      case accelerator::LESS_EQ:
        filter_result = decoded_value <= ctx->_pivot;
        break;
      case accelerator::GREATER_EQ:
        filter_result = decoded_value >= ctx->_pivot;
        break;
      default:
        break;
    }
    if (filter_result) {
      unsigned long long int idx = atomicAdd(results_idx, 1);
      size_t key_start = key - cached_data;
      size_t value_start = value - cached_data;
      results[idx] = RudaKVIndexPair(
          block_offset + key_start,
          block_offset + key_start + key_size,
          block_offset + value_start,
          block_offset + value_start + value_size);
      // if (shared != 0) {
      //   results[idx].pinKeyBuf(key_buf, key_buf_length);
      // }
    }

    // Next DataKey...
    subblock = value + value_size;
  }

  delete[] key_buf;
}

__device__
void CachedDecodeNFilterOnSchema(// Parameters
                                 const char *cached_data,
                                 const uint64_t cached_data_size,
                                 const uint64_t block_offset,
                                 const uint64_t start_idx,
                                 const uint64_t end_idx,
                                 RudaSchema *schema,
                                 // Results
                                 unsigned long long int *results_idx,
                                 ruda::RudaKVIndexPair *results) {
  const char *subblock = cached_data + start_idx;
  const char *limit = cached_data + end_idx;
  size_t key_buf_size = DEFAULT_KEY_BUF_SIZE;
  //size_t key_buf_length = 0;
  char *key_buf = new char[key_buf_size];
  while (subblock < limit) {
    uint32_t shared, non_shared, value_size;
    subblock = DecodeEntry()(subblock, limit, &shared, &non_shared,
                             &value_size);
    const char *key;
    size_t key_size;
    if (shared == 0) {
      key = subblock;
      key_size = non_shared;
      if (key_size > key_buf_size) {
        delete[] key_buf;
        key_buf_size = key_size;
        key_buf = new char[key_buf_size];
      }
      memset(key_buf, 0, sizeof(char) * key_buf_size);
      memcpy(key_buf, key, sizeof(char) * key_size);
      //key_buf_length = key_size;
    } else {
      key = subblock;
      key_size = shared + non_shared;
      if (key_size > key_buf_size) {
        char *new_key_buf = new char[key_size];
        memcpy(new_key_buf, key_buf, sizeof(char) * shared);
        delete[] key_buf;
        key_buf_size = key_size;
        key_buf = new_key_buf;
      }
      memcpy(key_buf + shared, key, sizeof(char) * non_shared);
      //key_buf_length = key_size;
    }

    const char *value = subblock + non_shared;

    bool is_equal_to_schema = true;
    for (size_t i = 0; i < NUM_TABLE_BYTES; ++i) {
      if (key_buf[i] != schema->data[i]) {
        is_equal_to_schema = false;
        break;
      }
    }
    if (!is_equal_to_schema) {
      subblock = value + value_size;
      continue;
    }
    
    char pivot[32] = {0,};
    long decoded_value = rudaConvertRecord(schema, value, pivot);
    bool filter_result = false;
    switch (schema->ctx._op) {
      case accelerator::EQ:
        filter_result = decoded_value == schema->ctx._pivot;
        break;
      case accelerator::LESS:
        filter_result = decoded_value < schema->ctx._pivot;
        break;
      case accelerator::GREATER:
        filter_result = decoded_value > schema->ctx._pivot;
        break;
      case accelerator::LESS_EQ:
        filter_result = decoded_value <= schema->ctx._pivot;
        break;
      case accelerator::GREATER_EQ:
        filter_result = decoded_value >= schema->ctx._pivot;
        break;
      case accelerator::NOT_EQ:
        filter_result = decoded_value != schema->ctx._pivot;
        break;
      case accelerator::MATCH:  
        for(int i = 0; i < schema->ctx.str_num; ++i) {
          if (memcmp(schema->ctx.cpivot[i], pivot, decoded_value)) {
            filter_result = true;
            break;
          }  
        } 
        break;       
      case accelerator::INVALID:
        // INVALID case, return all data to result.
        filter_result = true;
        break;
      default:
        break;
    }
    if (filter_result) {
      unsigned long long int idx = atomicAdd(results_idx, 1);
      size_t value_start = value - cached_data;
      results[idx] = RudaKVIndexPair(
          block_offset + value_start,
          block_offset + value_start + value_size);
    }

    // Next DataKey...
    subblock = value + value_size;
  }

  delete[] key_buf;
}

__device__
void DecodeNFilterOnSchema(// Parameters
                           const char *data,
                           const uint64_t lookup_size,
                           const uint64_t block_offset,
                           const uint64_t start_idx,
                           const uint64_t end_idx,
                           RudaSchema *schema,
                           // Results
                           unsigned long long int *results_idx,
                           ruda::RudaKVIndexPair *results) {
  const char *subblock = data + block_offset + start_idx;
  const char *limit = data + block_offset + end_idx;
  size_t key_buf_size = DEFAULT_KEY_BUF_SIZE;
 // size_t key_buf_length = 0;

  char *key_buf = new char[key_buf_size];
 
  while (subblock < limit) {

    uint32_t shared, non_shared, value_size;
    subblock = DecodeEntry()(subblock, limit, &shared, &non_shared,
                             &value_size);
    const char *key;
    size_t key_size;
    if (shared == 0) {
      key = subblock;
      key_size = non_shared;
      if (key_size > key_buf_size) {
        delete[] key_buf;
        key_buf_size = key_size;
        key_buf = new char[key_buf_size];
      }
      memset(key_buf, 0, sizeof(char) * key_buf_size);
      memcpy(key_buf, key, sizeof(char) * key_size);
      //key_buf_length = key_size;
    } else {
      key = subblock;
      key_size = shared + non_shared;
      if (key_size > key_buf_size) {
        char *new_key_buf = new char[key_size];
        memcpy(new_key_buf, key_buf, sizeof(char) * shared);
        delete[] key_buf;
        key_buf_size = key_size;
        key_buf = new_key_buf;
      }
      memcpy(key_buf + shared, key, sizeof(char) * non_shared);
      //key_buf_length = key_size;
    }

    const char *value = subblock + non_shared;

    bool is_equal_to_schema = true;
    for (size_t i = 0; i < NUM_TABLE_BYTES; ++i) {
      if (key_buf[i] != schema->data[i]) {
        is_equal_to_schema = false;
        break;
      }
    }
    if (!is_equal_to_schema) {
      subblock = value + value_size;
      continue;
    }

    char pivot[32] = {0,};
    long decoded_value = rudaConvertRecord(schema, value, pivot);
//    for(int i = 0; i < schema->ctx.str_num; ++i) {
//      printf("pivot : %s\n" , schema->ctx.cpivot[i]);
//    }
    bool filter_result = false;
    switch (schema->ctx._op) {
      case accelerator::EQ:
        filter_result = decoded_value == schema->ctx._pivot;
        break;
      case accelerator::LESS:
        filter_result = decoded_value < schema->ctx._pivot;
        break;
      case accelerator::GREATER:
        filter_result = decoded_value > schema->ctx._pivot;
        break;
      case accelerator::LESS_EQ:
        filter_result = decoded_value <= schema->ctx._pivot;
        break;
      case accelerator::GREATER_EQ:
        filter_result = decoded_value >= schema->ctx._pivot;
        break;
      case accelerator::NOT_EQ:
        filter_result = decoded_value != schema->ctx._pivot;
        break;
      case accelerator::MATCH:
        for(int i = 0; i < schema->ctx.str_num; ++i) {
//            printf("cpivot = %s and pivot = %s and length = %d\n", schema->ctx.cpivot[i], pivot, decoded_value);
//          printf("cpivot = %d\n", schema->ctx.cpivot[i][decoded_value]);
//          printf("pivot = %d\n", pivot[decoded_value]);
          if (memcmp(schema->ctx.cpivot[i], pivot, decoded_value)) {
            filter_result = true;
            break;
          }  
        }  
        break; 
      case accelerator::INVALID:
        // INVALID case, return all data to result.
        filter_result = true;
        break;
      default:
        break;
    }
    
    if (filter_result) {
      unsigned long long int idx = atomicAggInc(results_idx);
      size_t value_start = value - data;
      results[idx] = RudaKVIndexPair(value_start, value_start + value_size);
      results[idx].copyKey(key_buf, key_size - 8);     
    }

    // Next DataKey...
    subblock = value + value_size;
  }
  
//  if (((blockDim.x * blockIdx.x + threadIdx.x) % 1000) == 0) {
//      printf("counter = %d\n", counter);
//  }

  delete[] key_buf;
}

__device__
void DecodeNFilterOnSchemaDonard(// Parameters
                           const char *start_ptr,
                           uint32_t restart_offset,
                           uint32_t start_offset,
                           uint32_t num_task,
                           RudaSchema *schema,
                           // Results
                           unsigned long long int *results_idx,
                           ruda::donardSlice *d_results){


  int start = DecodeFixed32(start_ptr + start_offset);
  int end;
  const char *subblock = start_ptr + DecodeFixed32(start_ptr + start_offset);
  const char *limit;
  if( restart_offset != 0) { 
    limit = start_ptr + restart_offset;
    end = restart_offset;
  } else { 
    limit = start_ptr + DecodeFixed32(start_ptr + start_offset + num_task * sizeof(uint32_t));
    end = DecodeFixed32(start_ptr + start_offset + num_task * sizeof(uint32_t));
  }
 
  size_t key_buf_size = DEFAULT_KEY_BUF_SIZE;
  char *key_buf = new char[key_buf_size];
 
  while (subblock < limit) {
    
    uint32_t shared, non_shared, value_size;
    subblock = DecodeEntry()(subblock, limit, &shared, &non_shared,
                             &value_size);

    const char *key;
    size_t key_size;
    if (shared == 0) {
      key = subblock;
      key_size = non_shared;
      if (key_size > key_buf_size) {
        delete[] key_buf;
        key_buf_size = key_size;
        key_buf = new char[key_buf_size];
      }
      memset(key_buf, 0, sizeof(char) * key_buf_size);
      memcpy(key_buf, key, sizeof(char) * key_size);
      //key_buf_length = key_size;
    } else {
      key = subblock;
      key_size = shared + non_shared;
      if (key_size > key_buf_size) {
        char *new_key_buf = new char[key_size];
        memcpy(new_key_buf, key_buf, sizeof(char) * shared);
        delete[] key_buf;
        key_buf_size = key_size;
        key_buf = new_key_buf;
      }
      memcpy(key_buf + shared, key, sizeof(char) * non_shared);
      //key_buf_length = key_size;
    }

    const char *value = subblock + non_shared;

    bool is_equal_to_schema = true;
    for (size_t i = 0; i < NUM_TABLE_BYTES; ++i) {
      if (key_buf[i] != schema->data[i]) {
        is_equal_to_schema = false;
        break;
      }
    }
    if (!is_equal_to_schema) {
      subblock = value + value_size;
      continue;
    }

    char pivot[32] = {0,};
    long decoded_value = rudaConvertRecord(schema, value, pivot);

    bool filter_result = false;
    switch (schema->ctx._op) {
      case accelerator::EQ:
        filter_result = decoded_value == schema->ctx._pivot;
        break;
      case accelerator::LESS:
        filter_result = decoded_value < schema->ctx._pivot;
        break;
      case accelerator::GREATER:
        filter_result = decoded_value > schema->ctx._pivot;
        break;
      case accelerator::LESS_EQ:
        filter_result = decoded_value <= schema->ctx._pivot;
        break;
      case accelerator::GREATER_EQ:
        filter_result = decoded_value >= schema->ctx._pivot;
        break;
      case accelerator::NOT_EQ:
        filter_result = decoded_value != schema->ctx._pivot;
        break;
      case accelerator::MATCH:
        for(int i = 0; i < schema->ctx.str_num; ++i) {
          if (memcmp(schema->ctx.cpivot[i], pivot, decoded_value)) {
            filter_result = true;
            break;
          }  
        }  
        break; 
      case accelerator::INVALID:
        // INVALID case, return all data to result.
        filter_result = true;
        break;
      default:
        break;
    }
    
    if (filter_result) {
      unsigned long long int idx = atomicAdd(results_idx, 1);
      //printf("idx : %d\n", idx);
      d_results[idx] = donardSlice(value, value_size);    
      d_results[idx].copyKey(key_buf, key_size - 8);  
    }

    // Next DataKey...
    subblock = value + value_size;
  }
  delete[] key_buf; 
}

}  // namespace ruda
