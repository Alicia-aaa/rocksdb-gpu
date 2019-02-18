
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

struct FilterContext {
  Operator _op;
  int _pivot;

  std::string toString() const {
    std::stringstream ss;
    ss << "_op: " << toStringOperator(this->_op) << ", "
      << "_pivot: " << _pivot;
    return ss.str();
  }

  int operator()(const int target) const {
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
      default:
        return 0;
    }
  }
};

}  // namespace accelerator
