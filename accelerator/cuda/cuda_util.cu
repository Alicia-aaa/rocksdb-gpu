
#include "accelerator/cuda/cuda_util.h"
#include "accelerator/cuda/block_decoder.h"

namespace ruda {

// Note(totoro): These converting functions are related to 'misaligned address'
// issue on cuda.
// If you want to know why do convert like this, check a link.
// https://stackoverflow.com/questions/37323053/misaligned-address-in-cuda
__device__
int sint4korr(const char *record_ptr) {
  int result;
  char *result_ptr = (char *) &result;
  for (unsigned long i = 0; i < sizeof(int); ++i) {
    result_ptr[i] = record_ptr[i];
  }
  return result;
}

__device__
uint uint2korr(const char *record_ptr) {
  unsigned short result;
  char *result_ptr = (char *) &result;
  for (unsigned long i = 0; i < sizeof(unsigned short); ++i) {
    result_ptr[i] = record_ptr[i];
  }
  return result;
}

__device__
int uint3korr(const char *record_ptr) {
  unsigned int result;
  char *result_ptr = (char *) &result;
  for (unsigned long i = 0; i < sizeof(unsigned int); ++i) {
    result_ptr[i] = record_ptr[i];
  }
  return result & 0xFFFFFF;
}

// Note(totoro): This function is copy of accelerator::convertRecord()...
__device__
long rudaConvertRecord(RudaSchema *schema, const char *record_ptr) {
  if (schema->target_idx < 0) {
    return -1;
  }

  // Skip other columns...
  for (int i = 0; i < schema->target_idx; ++i) {
    // TODO(totoro): Needs to handles 'm_null_bytes_in_rec' byte on record_ptr...
    // If column has 'Nullable' constraint, record has a 1 byte for notifying
    // 'this column value is null'.
    // So, when decode a nullable column, below code must handles null notifier
    // byte.
    if (schema->field_type[i] == 15) {
      uint data_len = schema->field_length[i] == 1
          ? (unsigned char) record_ptr[0]
          : uint2korr(record_ptr);
      record_ptr += data_len + schema->field_length[i] + schema->field_skip[i];
    } else {
      record_ptr += (schema->field_length[i] + schema->field_skip[i]);
    }
  }
  record_ptr += schema->field_skip[schema->target_idx];

  long result;
  if (schema->field_type[schema->target_idx] == 14) { // Date type
    result = uint3korr(record_ptr);
  } else if (schema->field_type[schema->target_idx] == 3) { // Long type
    result = sint4korr(record_ptr);
  } else {
    result = -1;
  }

  return result;
}

}  // namespace ruda
