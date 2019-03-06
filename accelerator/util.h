
/*
 * util.h
 *
 *  Created on: Mar 05, 2019
 *      Author: totoro
 */

#pragma once

#include "rocksdb/slice.h"

namespace accelerator {

long convertRecord(const rocksdb::SlicewithSchema &schema_key,
                   const char *record_ptr) {
  // printf("[util.h][convertRecord] START\n");
  // Skip other columns...
  for (int i = 0; i < schema_key.getTarget(); ++i) {
    if (schema_key.getType(i) == 15) {
      uint data_len = schema_key.getLength(i) == 1
          ? (unsigned char) record_ptr[0]
          : (unsigned short)(*((unsigned short *)(record_ptr)));
      record_ptr += data_len;
    } else {
      record_ptr += schema_key.getLength(i);
    }
  }
  // printf("[util.h][convertRecord] After skip other columns\n");

  long result = (int)(*((int *)(record_ptr)));
  // printf("[util.h][convertRecord] converted value: %ld\n", result);
  return result;
}

}  // namespace accelerator
