
/*
 * filter.h
 *
 *  Created on: Feb 18, 2019
 *      Author: totoro
 */

#pragma once

#include <vector>

#include "accelerator/common.h"

namespace avx {

int simpleIntFilter(std::vector<int> &source, accelerator::FilterContext ctx,
                    std::vector<int> &results);

}  // namespace avx
