#include "reducetype.h"
#include <torch/torch.h>
#pragma once

static inline ReductionType get_reduction_enum(const c10::string_view &reduce) {
  if (reduce == "max" || reduce == "amax") {
    return ReductionType::MAX;
  } else if (reduce == "mean") {
    return ReductionType::MEAN;
  } else if (reduce == "min" || reduce == "amin") {
    return ReductionType::MIN;
  } else if (reduce == "sum") {
    return ReductionType::SUM;
  } else if (reduce == "prod") {
    return ReductionType::PROD;
  } else {
    TORCH_CHECK(
        false,
        "reduce argument must be either sum, prod, mean, amax or amin, got ",
        reduce);
  }
}
