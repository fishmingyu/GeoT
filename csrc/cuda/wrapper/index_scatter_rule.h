#pragma once
#include "index_scatter_base.h"
template <typename scalar_t>
void index_scatter_sorted_wrapper(const at::Tensor &index,
                                  const at::Tensor &src,
                                  const at::Tensor &dst) {
    const auto size = index.numel();
    const auto feature_size = src.numel() / size;
    const auto keys = dst.numel() / feature_size;
    int avg = size / keys;
    if (feature_size < 8) {
        if (size <= 6677.0) {
            if (size <= 2404.0) {
                if (feature_size <= 3.0) {
                    if (size <= 1435.0) {
                        if (size <= 1370.5) {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 2, 32>(index, src, dst);
                        }
                    } else {
                        if (feature_size <= 1.5) {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        }
                    }
                } else {
                    if (size <= 1937.0) {
                        if (size <= 1062.5) {
                            segreduce_pr_sorted<scalar_t, 1, 4, 1, 2, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 4, 1, 2, 32>(index, src, dst);
                        }
                    } else {
                        if (size <= 2202.0) {
                            segreduce_pr_sorted<scalar_t, 1, 4, 1, 2, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        }
                    }
                }
            } else {
                if (feature_size <= 1.5) {
                    if (size <= 4729.5) {
                        if (avg <= 30.3) {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 8, 32>(index, src, dst);
                        }
                    } else {
                        if (size <= 6285.5) {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        }
                    }
                } else {
                    if (size <= 3948.0) {
                        if (size <= 2961.0) {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        }
                    } else {
                        if (avg <= 38.33) {
                            segreduce_pr_sorted<scalar_t, 1, 4, 1, 4, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 4, 1, 4, 32>(index, src, dst);
                        }
                    }
                }
            }
        } else {
            if (avg <= 1.15) {
                if (feature_size <= 3.0) {
                    if (size <= 703065.0) {
                        if (size <= 351532.5) {
                            segreduce_pr_sorted<scalar_t, 1, 1, 2, 4, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 2, 1, 2, 8, 32>(index, src, dst);
                        }
                    } else {
                        if (feature_size <= 1.5) {
                            segreduce_pr_sorted<scalar_t, 1, 1, 2, 8, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 2, 2, 2, 32>(index, src, dst);
                        }
                    }
                } else {
                    if (size <= 1406130.0) {
                        if (avg <= 1.15) {
                            segreduce_pr_sorted<scalar_t, 2, 2, 2, 1, 16>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 4, 4, 1, 16>(index, src, dst);
                        }
                    } else {
                        segreduce_pr_sorted<scalar_t, 2, 2, 4, 1, 16>(index, src, dst);
                    }
                }
            } else {
                if (feature_size <= 1.5) {
                    if (size <= 359204.0) {
                        if (avg <= 7.19) {
                            segreduce_pr_sorted<scalar_t, 1, 1, 1, 8, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 1, 1, 8, 32>(index, src, dst);
                        }
                    } else {
                        if (size <= 2188236.0) {
                            segreduce_pr_sorted<scalar_t, 1, 1, 2, 8, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 1, 1, 2, 8, 32>(index, src, dst);
                        }
                    }
                } else {
                    if (avg <= 35.71) {
                        if (size <= 70469.0) {
                            segreduce_pr_sorted<scalar_t, 1, 2, 1, 4, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 4, 1, 2, 4, 32>(index, src, dst);
                        }
                    } else {
                        if (size <= 331916.5) {
                            segreduce_pr_sorted<scalar_t, 2, 2, 1, 8, 32>(index, src, dst);
                        } else {
                            segreduce_pr_sorted<scalar_t, 4, 1, 2, 4, 32>(index, src, dst);
                        }
                    }
                }
            }
        }
    } else {
        if (feature_size <= 48.0) {
            if (feature_size <= 24.0) {
                if (size <= 20974.5) {
                    if (avg <= 9.39) {
                        if (size <= 7155.0) {
                            segreduce_sr_sorted<scalar_t, 1, 32, 4, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 1, 32, 4, 8>(index, src, dst);
                        }
                    } else {
                        if (avg <= 35.43) {
                            segreduce_sr_sorted<scalar_t, 1, 16, 4, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 1, 16, 4, 8>(index, src, dst);
                        }
                    }
                } else {
                    if (feature_size <= 12.0) {
                        if (size <= 93022.5) {
                            segreduce_sr_sorted<scalar_t, 1, 8, 4, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 1, 8, 8, 8>(index, src, dst);
                        }
                    } else {
                        if (size <= 247962.5) {
                            segreduce_sr_sorted<scalar_t, 1, 16, 8, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 8, 8, 8>(index, src, dst);
                        }
                    }
                }
            } else {
                if (size <= 55061.5) {
                    if (size <= 6521.0) {
                        if (size <= 6413.5) {
                            segreduce_sr_sorted<scalar_t, 1, 32, 4, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 1, 64, 8, 4>(index, src, dst);
                        }
                    } else {
                        if (avg <= 502.41) {
                            segreduce_sr_sorted<scalar_t, 1, 32, 4, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 1, 64, 8, 4>(index, src, dst);
                        }
                    }
                } else {
                    if (avg <= 4.22) {
                        if (size <= 2351251.0) {
                            segreduce_sr_sorted<scalar_t, 2, 16, 8, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 16, 32, 8>(index, src, dst);
                        }
                    } else {
                        if (size <= 220246.0) {
                            segreduce_sr_sorted<scalar_t, 2, 16, 8, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 16, 8, 8>(index, src, dst);
                        }
                    }
                }
            }
        } else {
            if (feature_size <= 96.0) {
                if (size <= 10487.0) {
                    if (avg <= 1.15) {
                        segreduce_sr_sorted<scalar_t, 1, 32, 4, 8>(index, src, dst);
                    } else {
                        if (avg <= 38.97) {
                            segreduce_sr_sorted<scalar_t, 1, 64, 4, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 1, 64, 4, 8>(index, src, dst);
                        }
                    }
                } else {
                    if (size <= 1094118.0) {
                        if (avg <= 492.02) {
                            segreduce_sr_sorted<scalar_t, 2, 32, 8, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 32, 16, 8>(index, src, dst);
                        }
                    } else {
                        if (size <= 1127504.0) {
                            segreduce_sr_sorted<scalar_t, 2, 32, 32, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 32, 16, 8>(index, src, dst);
                        }
                    }
                }
            } else {
                if (size <= 18918.5) {
                    if (size <= 5341.5) {
                        if (size <= 3142.5) {
                            segreduce_sr_sorted<scalar_t, 1, 64, 4, 8>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 64, 4, 4>(index, src, dst);
                        }
                    } else {
                        if (avg <= 36.3) {
                            segreduce_sr_sorted<scalar_t, 2, 32, 8, 4>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 64, 8, 4>(index, src, dst);
                        }
                    }
                } else {
                    if (size <= 747442.0) {
                        if (avg <= 492.8) {
                            segreduce_sr_sorted<scalar_t, 2, 64, 8, 4>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 64, 16, 4>(index, src, dst);
                        }
                    } else {
                        if (avg <= 15.04) {
                            segreduce_sr_sorted<scalar_t, 2, 64, 16, 4>(index, src, dst);
                        } else {
                            segreduce_sr_sorted<scalar_t, 2, 64, 8, 4>(index, src, dst);
                        }
                    }
                }
            }
        }
    }
}