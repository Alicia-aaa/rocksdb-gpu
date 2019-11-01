
/*
 * common.h
 *
 *  Created on: Feb 18, 2019
 *      Author: totoro
 */

#pragma once

#include <sstream>
#include <string>

namespace accelerator {

const int ACC_ERR = 1;
const int ACC_OK = 0;

enum class ValueFilterMode {
  NORMAL = 0, AVX, AVX_BLOCK, GPU, ASYNC, DONARD,
};

enum Operator {
  EQ = 0, LESS, GREATER, LESS_EQ, GREATER_EQ, NOT_EQ, STRMATCH, INVALID
};

inline std::string toStringOperator(Operator op) {
  switch (op) {
    case EQ: return "EQ";
    case LESS: return "LESS";
    case GREATER: return "GREATER";
    case LESS_EQ: return "LESS_EQ";
    case GREATER_EQ: return "GREATER_EQ";
    case NOT_EQ: return "NOT_EQ";
    case STRMATCH: return "STRMATCH";
    default: return "INVALID";
  }
}

struct FilterContext {
  Operator _op;
  long _pivot;
  int comp_field;
  int str_num;
  char cpivot[10][32];
  long long pivots[10];

  std::string toString() const {
    std::stringstream ss;
    ss << "_op: " << toStringOperator(this->_op) << ", "
      << "_pivot: " << _pivot;
    return ss.str();
  }

  bool isValidOp() const {
    return _op == EQ || _op == LESS || _op == GREATER || _op == LESS_EQ
        || _op == GREATER_EQ || _op == NOT_EQ || _op == STRMATCH;
  }

  int operator()(const long target) const {
    switch (_op) {
      case EQ:
        return target == _pivot ? 1 : 0;
      case LESS:
        return target < _pivot ? 1 : 0;
      case GREATER:
        return target > _pivot ? 1 : 0;
      case LESS_EQ:
        return target <= _pivot ? 1 : 0;
      case GREATER_EQ:
        return target >= _pivot ? 1 : 0;
      case NOT_EQ:
        return target != _pivot ? 1 : 0;
      case STRMATCH:
      {
        for(int i = 0 ; i < str_num; i++) {
          if(target == pivots[i]) return 1;
        }
        return 0;
      }
      default:
        return 0;
    }
  }
};

}  // namespace accelerator
