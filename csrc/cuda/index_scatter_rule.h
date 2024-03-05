
#pragma once
#include "index_scatter_base.h"
template <typename scalar_t, ReductionType reduce>
void index_scatter_sorted_wrapper(const at::Tensor &index,
                                  const at::Tensor &src,
                                  const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto feature_size = src.numel() / nnz;
  const auto keys = dst.numel() / feature_size;
  int avg = nnz / keys;
  if (feature_size < 8) {
    if (feature_size <= 3.0) {
      if (feature_size <= 1.5) {
        if (avg <= 502.41) {
          if (avg <= 288.4) {
            if (avg <= 1.15) {
              segreduce_pr_sorted<scalar_t, 1, 1, 2, 4, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 1, 1, 8, 32>(index, src, dst);
            }
          } else {
            if (avg <= 492.02) {
              segreduce_pr_sorted<scalar_t, 1, 1, 4, 4, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 1, 1, 8, 32>(index, src, dst);
            }
          }
        } else {
          if (avg <= 607.25) {
            if (avg <= 544.1) {
              segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 1, 1, 4, 32>(index, src, dst);
            }
          } else {
            segreduce_pr_sorted<scalar_t, 1, 4, 1, 4, 32>(index, src, dst);
          }
        }
      } else {
        if (avg <= 492.42) {
          if (avg <= 288.4) {
            if (avg <= 84.56) {
              segreduce_pr_sorted<scalar_t, 2, 1, 1, 4, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
            }
          } else {
            if (avg <= 491.67) {
              segreduce_pr_sorted<scalar_t, 2, 1, 4, 4, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 2, 1, 2, 8, 32>(index, src, dst);
            }
          }
        } else {
          if (avg <= 505.15) {
            if (avg <= 494.17) {
              segreduce_pr_sorted<scalar_t, 1, 2, 2, 4, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 2, 1, 8, 32>(index, src, dst);
            }
          } else {
            if (avg <= 544.1) {
              segreduce_pr_sorted<scalar_t, 1, 2, 1, 2, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 1, 1, 4, 32>(index, src, dst);
            }
          }
        }
      }
    } else {
      if (avg <= 3.24) {
        if (avg <= 1.15) {
          if (avg <= 1.15) {
            segreduce_pr_sorted<scalar_t, 2, 2, 2, 1, 16>(index, src, dst);
          } else {
            segreduce_pr_sorted<scalar_t, 1, 4, 4, 1, 16>(index, src, dst);
          }
        } else {
          if (avg <= 1.64) {
            if (avg <= 1.15) {
              segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 4, 1, 2, 32>(index, src, dst);
            }
          } else {
            if (avg <= 2.33) {
              segreduce_pr_sorted<scalar_t, 2, 2, 2, 2, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 2, 2, 2, 2, 32>(index, src, dst);
            }
          }
        }
      } else {
        if (avg <= 492.42) {
          if (avg <= 288.4) {
            if (avg <= 84.56) {
              segreduce_pr_sorted<scalar_t, 2, 2, 1, 2, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 4, 1, 2, 32>(index, src, dst);
            }
          } else {
            if (avg <= 491.12) {
              segreduce_pr_sorted<scalar_t, 4, 1, 4, 2, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 4, 1, 4, 4, 32>(index, src, dst);
            }
          }
        } else {
          if (avg <= 505.15) {
            if (avg <= 494.17) {
              segreduce_pr_sorted<scalar_t, 1, 2, 2, 4, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 4, 1, 4, 32>(index, src, dst);
            }
          } else {
            if (avg <= 607.25) {
              segreduce_pr_sorted<scalar_t, 1, 4, 1, 2, 32>(index, src, dst);
            } else {
              segreduce_pr_sorted<scalar_t, 1, 4, 1, 2, 32>(index, src, dst);
            }
          }
        }
      }
    }
  } else {
    if (feature_size <= 48.0) {
      if (feature_size <= 24.0) {
        if (feature_size <= 12.0) {
          if (avg <= 49.8) {
            if (avg <= 8.67) {
              segreduce_sr_sorted<scalar_t, 1, 16, 4, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 8, 4, 8>(index, src, dst);
            }
          } else {
            if (avg <= 57.28) {
              segreduce_sr_sorted<scalar_t, 1, 8, 16, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 8, 8, 8>(index, src, dst);
            }
          }
        } else {
          if (avg <= 5.74) {
            if (avg <= 1.64) {
              segreduce_sr_sorted<scalar_t, 1, 16, 4, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 16, 8, 8>(index, src, dst);
            }
          } else {
            if (avg <= 38.23) {
              segreduce_sr_sorted<scalar_t, 1, 16, 4, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 16, 8, 8>(index, src, dst);
            }
          }
        }
      } else {
        if (avg <= 497.47) {
          if (avg <= 288.4) {
            if (avg <= 84.45) {
              segreduce_sr_sorted<scalar_t, 2, 16, 8, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 32, 4, 8>(index, src, dst);
            }
          } else {
            if (avg <= 492.42) {
              segreduce_sr_sorted<scalar_t, 2, 16, 8, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 32, 8, 8>(index, src, dst);
            }
          }
        } else {
          if (avg <= 505.15) {
            if (avg <= 502.41) {
              segreduce_sr_sorted<scalar_t, 1, 32, 4, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 64, 8, 4>(index, src, dst);
            }
          } else {
            segreduce_sr_sorted<scalar_t, 1, 32, 4, 8>(index, src, dst);
          }
        }
      }
    } else {
      if (feature_size <= 96.0) {
        if (avg <= 288.4) {
          if (avg <= 14.9) {
            if (avg <= 13.83) {
              segreduce_sr_sorted<scalar_t, 2, 32, 8, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 2, 32, 16, 8>(index, src, dst);
            }
          } else {
            if (avg <= 36.81) {
              segreduce_sr_sorted<scalar_t, 2, 32, 4, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 2, 32, 8, 8>(index, src, dst);
            }
          }
        } else {
          if (avg <= 497.47) {
            if (avg <= 491.32) {
              segreduce_sr_sorted<scalar_t, 2, 32, 16, 8>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 2, 32, 16, 4>(index, src, dst);
            }
          } else {
            if (avg <= 499.7) {
              segreduce_sr_sorted<scalar_t, 2, 32, 8, 4>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 64, 4, 8>(index, src, dst);
            }
          }
        }
      } else {
        if (feature_size <= 192.0) {
          if (avg <= 288.4) {
            if (avg <= 14.9) {
              segreduce_sr_sorted<scalar_t, 2, 64, 8, 4>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 2, 64, 8, 4>(index, src, dst);
            }
          } else {
            if (avg <= 499.7) {
              segreduce_sr_sorted<scalar_t, 2, 64, 16, 2>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 1, 64, 8, 8>(index, src, dst);
            }
          }
        } else {
          if (avg <= 15.21) {
            if (avg <= 3.95) {
              segreduce_sr_sorted<scalar_t, 2, 64, 8, 4>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 2, 64, 8, 4>(index, src, dst);
            }
          } else {
            if (avg <= 41.46) {
              segreduce_sr_sorted<scalar_t, 2, 64, 4, 4>(index, src, dst);
            } else {
              segreduce_sr_sorted<scalar_t, 2, 64, 8, 4>(index, src, dst);
            }
          }
        }
      }
    }
  }
}