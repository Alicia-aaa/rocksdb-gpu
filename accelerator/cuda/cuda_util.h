
/*
 * cuda_util.h
 *
 *  Created on: Mar 05, 2019
 *      Author: totoro
 */

#pragma once

#include "accelerator/cuda/block_decoder.h"

namespace ruda {

__device__
long rudaConvertRecord(RudaSchema *schema, const char *record_ptr);

}  // namespace ruda
