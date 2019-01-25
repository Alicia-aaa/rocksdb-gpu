
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

#include "rocksdb/slice.h"

namespace ruda {

const int RUDA_ERR = -1;
const int RUDA_OK = 0;

enum Operator {
  EQ = 0, LESS, GREATER, LESS_EQ, GREATER_EQ,
};

inline std::string toStringOperator(Operator op) {
  switch (op) {
    case EQ: return "EQ";
    case LESS: return "LESS";
    case GREATER: return "GREATER";
    case LESS_EQ: return "LESS_EQ";
    case GREATER_EQ: return "GREATER_EQ";
    default: return "INVALID";
  }
}

struct ConditionContext {
  Operator _op;
  int _pivot;

  std::string toString() const {
    std::stringstream ss;
    ss << "_op: " << toStringOperator(this->_op) << ", "
      << "_pivot: " << _pivot;
    return ss.str();
  }
};

int sstIntFilter(const std::vector<int> &values,
                 const ConditionContext context,
                 std::vector<int> &results);

// Note: Not Implemented.
int sstChunkFilter(const char *values,
                   const ConditionContext context,
                   char *results);

int sstIntNativeFilter(const std::vector<int> &values,
                       const ConditionContext context,
                       std::vector<int> &results);

int sstIntBlockFilter(const std::vector<char> &datablocks,
                      const std::vector<uint64_t> &seek_indices,
                      const ConditionContext context,
                      const size_t results_count,
                      std::vector<rocksdb::Slice> &keys,
                      std::vector<rocksdb::Slice> &values);

}  // namespace ruda
