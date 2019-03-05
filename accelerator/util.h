
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

  uint len = 0;
  if (schema_key.getType(schema_key.getTarget()) == 15) {
    len = schema_key.getLength(schema_key.getTarget()) == 1
        ? (unsigned char) record_ptr[0]
        : (unsigned short)(*((unsigned short *)(record_ptr)));
  } else {
    len = schema_key.getLength(schema_key.getTarget());
  }

  unsigned char* str = new unsigned char[len];
  memcpy(str, record_ptr, len * sizeof(unsigned char));
  long result = (long)(*((int *)(str)));
  delete str;
  return result;
}

}  // namespace accelerator
