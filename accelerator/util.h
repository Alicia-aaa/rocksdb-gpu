
/*
 * util.h
 *
 *  Created on: Mar 05, 2019
 *      Author: totoro
 */

#pragma once

#include "rocksdb/slice.h"
#include <iostream>


#define sint4korr(A)    (int)  (*((int *) (A)))
#define uint3korr(A)    (unsigned int) (*((unsigned int *) (A)) & 0xFFFFFF)


namespace accelerator {

inline long convertRecord(const rocksdb::SlicewithSchema &schema_key,
                          const char *record_ptr) {
  int target_idx = schema_key.getTarget();
  long result = 0;
  if (target_idx < 0) {
    return -1;
  }

  // printf("[util.h][convertRecord] START\n");
  // Skip other columns...
  for (int i = 0; i < target_idx; ++i) {
    if (schema_key.getType(i) == 15) {
      uint data_len = schema_key.getLength(i) == 1
          ? (unsigned char) record_ptr[0]
          : (unsigned short)(*((unsigned short *)(record_ptr)));
      record_ptr += data_len + schema_key.getSkip(i);
    } else {
      record_ptr += (schema_key.getLength(i) + schema_key.getSkip(i));
    }
  }
  record_ptr += schema_key.getSkip(target_idx);
  // printf("[util.h][convertRecord] After skip other columns\n");
  if(schema_key.getType(target_idx) == 14) { // DATE type
      result = uint3korr(record_ptr);

    //  result = (j % 32L) + (j / 32L % 16L) * 100L + (j/(16L*32L)) * 10000L;
  } else if (schema_key.getType(target_idx) == 3) { // Long Type
      result = sint4korr(record_ptr);
  }
  // printf("[util.h][convertRecord] converted value: %ld\n", result);
  return result;
}

}  // namespace accelerator