
#include "accelerator/cuda/cuda_util.h"
#include "accelerator/cuda/block_decoder.h"

namespace ruda {

// Note(totoro): This function is copy of accelerator::convertRecord()...
__device__
long rudaConvertRecord(RudaSchema *schema, const char *record_ptr) {
  if (schema->target_idx < 0) {
    return -1;
  }

  // printf("[util.h][convertRecord] START\n");
  // Skip other columns...
  for (int i = 0; i < schema->target_idx; ++i) {
    if (schema->field_type[i] == 15) {
      uint data_len = schema->field_length[i] == 1
          ? (unsigned char) record_ptr[0]
          : (unsigned short)(*((unsigned short *)(record_ptr)));
      record_ptr += data_len;
    } else {
      record_ptr += schema->field_length[i];
    }
  }
  // printf("[util.h][convertRecord] After skip other columns\n");

  long result = (int)(*((int *)(record_ptr)));
  // printf("[util.h][convertRecord] converted value: %ld\n", result);
  return result;
}

}  // namespace ruda
