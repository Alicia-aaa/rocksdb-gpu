
/*
 * filter.h
 *
 *  Created on: Dec 10, 2018
 *      Author: totoro
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "accelerator/common.h"
#include "rocksdb/slice.h"

namespace ruda {

int sstThrustFilter(const std::vector<int> &values,
                    const accelerator::FilterContext context,
                    std::vector<int> &results);

// Note: Not Implemented.
int sstChunkFilter(const char *values,
                   const accelerator::FilterContext context,
                   char *results);

int sstIntNativeFilter(const std::vector<int> &values,
                       const accelerator::FilterContext context,
                       std::vector<int> &results);

int sstIntBlockFilter(const std::vector<char> &datablocks,
                      const std::vector<uint64_t> &seek_indices,
                      const accelerator::FilterContext context,
                      const size_t results_count,
                      std::vector<rocksdb::Slice> &keys,
                      std::vector<rocksdb::Slice> &values);

}  // namespace ruda
