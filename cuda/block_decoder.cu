#include <cstdio>

#include "cuda/block_decoder.h"

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

__device__
void DecodeSubDataBlocks(// Parameters
                         const char *cached_data,
                         const uint64_t cached_data_size,
                         const uint64_t start_idx, const uint64_t end_idx,
                         ConditionContext *ctx,
                         // Results
                         unsigned long long int *results_idx,
                         ruda::RudaSlice *results_keys,
                         ruda::RudaSlice *results_values) {
  const char *subblock = &cached_data[start_idx];
  const char *limit = &cached_data[end_idx];
  while (subblock < limit) {
    uint32_t shared, non_shared, value_size;
    subblock = DecodeEntry()(subblock, limit, &shared, &non_shared,
                             &value_size);
    const char *key;
    size_t key_size;
    if (shared == 0) {
      key = subblock;
      key_size = non_shared;
    } else {
      // TODO(totoro): We need to consider 'shared' data within subblock.
      key = subblock;
      key_size = shared + non_shared;
    }

    const char *value = subblock + non_shared;

    unsigned long long int idx = atomicAdd(results_idx, 1);
    char *results_key = new char[key_size];
    char *results_value = new char[value_size];
    memcpy(results_key, key, key_size);
    memcpy(results_value, value, value_size);
    results_keys[idx] = RudaSlice(results_key, key_size);
    results_values[idx] = RudaSlice(results_value, value_size);

    // Next DataKey...
    subblock = value + value_size;
  }
}

}  // namespace ruda
