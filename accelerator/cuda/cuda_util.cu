
#include "accelerator/cuda/cuda_util.h"
#include "accelerator/cuda/block_decoder.h"

namespace ruda {

// Note(totoro): These converting functions are related to 'misaligned address'
// issue on cuda.
// If you want to know why do convert like this, check a link.
// https://stackoverflow.com/questions/37323053/misaligned-address-in-cuda

__device__    
const unsigned char *skip_trailing_space(const unsigned char *ptr, size_t len) {
  const unsigned char *end= ptr + len;

  if (len > 20)
  {
    const unsigned char *end_words= (const unsigned char *)(long)
      (((unsigned long long int)(long)end) / 4 * 4);
    const unsigned char *start_words= (const unsigned char *)(long)
       ((((unsigned long long int)(long)ptr) + 4 - 1) / 4 * 4);
    
    if (end_words > ptr)
    {
      while (end > end_words && end[-1] == 0x20)
        end--;
      if (end[-1] == 0x20 && start_words < end_words)
        while (end > start_words && ((unsigned *)end)[-1] == 0x20202020)
          end -= 4;
    }
  }
  while (end > ptr && end[-1] == 0x20)
    end--;
  return (end);
}
    
    
__device__
long long int float4korr(const char *record_ptr) {
  float result;
  char *result_ptr = (char *) &result;
  for (unsigned long i = 0; i < sizeof(int); ++i) {
    result_ptr[i] = record_ptr[i];
  }
  return rint(result);
}
    
    
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
long rudaConvertRecord(RudaSchema *schema, const char *record_ptr, char * pivot) {
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
      record_ptr += schema->field_skip[i];
      uint data_len = schema->field_length[i] == 1
          ? (unsigned char) record_ptr[0]
          : uint2korr(record_ptr);
      record_ptr += data_len + schema->field_length[i];
    } else {
      record_ptr += (schema->field_length[i] + schema->field_skip[i]);
    }
  }
  record_ptr += schema->field_skip[schema->target_idx];

  long result = -1;
  int target_type = schema->field_type[schema->target_idx];
  if (target_type == 14 ) { // Date type
    result = uint3korr(record_ptr);
  } else if (target_type == 3 ) { // Long type
    result = sint4korr(record_ptr);
  } else if (target_type == 4 ) { // float type : require to check validity
    result = float4korr(record_ptr);
  } else if (target_type == 254) {
    const char *end = (const char *)skip_trailing_space((const unsigned char*) record_ptr, schema->field_length[schema->target_idx]);
    size_t len = (size_t) (end - record_ptr);
//    printf("record ptr  = %p\n", record_ptr);
//    unsigned char *temp = (unsigned char *)malloc(sizeof(unsigned char) * len);
    memcpy(pivot, record_ptr + 1, len);
//    for (unsigned int i = 0; i < 25; ++i) {
//      printf("%c", temp[i]);
//    }
//    printf("size = %d\n" , len);
//    printf("str = %s\n", temp);    
//    printf("\n");
    result = len;
  }

  return result;
}

}  // namespace ruda
