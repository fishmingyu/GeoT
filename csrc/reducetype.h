#pragma once

enum class ReductionType { MAX, MEAN, MIN, SUM, PROD };

#define DISPATCH_REDUCTION_TYPES(op, ...)                                      \
  [&] {                                                                        \
    switch (op) {                                                              \
    case ReductionType::SUM: {                                                 \
      static constexpr auto reduce = ReductionType::SUM;                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case ReductionType::MEAN: {                                                \
      static constexpr auto reduce = ReductionType::MEAN;                      \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case ReductionType::MIN: {                                                 \
      static constexpr auto reduce = ReductionType::MIN;                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case ReductionType::MAX: {                                                 \
      static constexpr auto reduce = ReductionType::MAX;                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case ReductionType::PROD: {                                                \
      static constexpr auto reduce = ReductionType::PROD;                      \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    }                                                                          \
  }()
