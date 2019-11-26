
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
#define uint4korr(A)	(unsigned int) (*((unsigned int *) (A)))
#define uint3korr(A)    (unsigned int) (*((unsigned int *) (A)) & 0xFFFFFF)



namespace accelerator {
    
inline const unsigned char *skip_trailing_space(const unsigned char *ptr, size_t len) {
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

inline long long compute_hash(std::string const &s) {
    const int p = 95;
    const int m = 1e9 + 9;
    
    long long hash_value = 0;
    long long p_pow = 1;
    for (char c : s) {
      hash_value = (hash_value + (c - 0x20 + 1) * p_pow) % m;
      p_pow = (p_pow * p) % m;
    }
    return hash_value;
}    
    
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
    // TODO(totoro): Needs to handles 'm_null_bytes_in_rec' byte on record_ptr...
    // If column has 'Nullable' constraint, record has a 1 byte for notifying
    // 'this column value is null'.
    // So, when decode a nullable column, below code must handles null notifier
    // byte.
    if (schema_key.getType(i) == 15) {
      record_ptr += schema_key.getSkip(i);
      uint data_len = schema_key.getLength(i) == 1
          ? (unsigned char) record_ptr[0]
          : (unsigned short)(*((unsigned short *)(record_ptr)));
      record_ptr += data_len + schema_key.getLength(i);
    } else {
      record_ptr += (schema_key.getLength(i) + schema_key.getSkip(i));
    }
  }

  record_ptr += schema_key.getSkip(target_idx);
  // printf("[util.h][convertRecord] After skip other columns\n");
  if(schema_key.getType(target_idx) == 14) { // DATE type
    result = uint3korr(record_ptr);
    // long year = result / (16 * 32);
    // long month = (result - year) / 32;
    // long day = (result - year - month);
    // printf("[util.h][convertRecord] year: %ld, month: %ld, day: %ld\n",
    //   year, month, day);
    // long test = (result % 32L) + (result / 32L % 16L) * 100L + (result/(16L*32L)) * 10000L;
    // printf("[util.h][convertRecord] test: %ld\n", test);
  } else if (schema_key.getType(target_idx) == 3 ) { // Long Type
    /* For YCSB */
    //result = sint4korr((unsigned char*)record_ptr+2);
    //    std::cout << "result1 : " << result << std::endl;
    //result = sint4korr(record_ptr);
    //    std::cout << "result2 : " << result << std::endl;
    /* General Case */
    result = sint4korr((unsigned char*)record_ptr);
     //   std::cout << "result3 : " << result << std::endl;

  } else if (schema_key.getType(target_idx) == 4 ) {
    result = sint4korr(record_ptr);
  } else  if (schema_key.getType(target_idx) == 254 ) {
    const char *end = (const char *)skip_trailing_space((const unsigned char*) record_ptr, schema_key.field_length[schema_key.target_idx]);
    size_t len = (size_t) (end - record_ptr);

    if(record_ptr[0] == 0x00) {
      record_ptr += 1;
      len -= 1;
    }

    std::string str;
    str.assign(record_ptr, len);

    result = compute_hash(str);
    //std::cout << "str = " << str << " and " << result << " and  " << len << std::endl;    
  }
  // printf("[util.h][convertRecord] converted value: %ld\n", result);
  return result;
}

}  // namespace accelerator
