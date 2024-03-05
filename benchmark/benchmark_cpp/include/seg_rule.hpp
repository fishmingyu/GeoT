#include "util.cuh"

template <typename scalar_t>
void segreduce_naive_rule(std::ofstream &out_file, char *data_name, int nnz,
                          int N, int keys, util::RamArray<Index> &index,
                          util::RamArray<DType> &src,
                          util::RamArray<DType> &dst) {
  float time = 0;
  float gflops = 0;
  int avg_key_len = nnz / keys;
  if (N >= 1 && N <= 4) {
    time = segreduce_pr_test<scalar_t, 1, 1, 2, 4, 32>(nnz, N, keys, index, src,
                                                  dst);
  } else if (N > 4 && N <= 16) {
    time = segreduce_pr_test<scalar_t, 2, 2, 2, 4, 32>(nnz, N, keys, index, src,
                                                  dst);
  } else if (N > 16 && N < 64) {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 16, 16, 2>(nnz, N, keys, index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 16, 32, 2>(nnz, N, keys, index, src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 16, 32, 4>(nnz, N, keys, index, src, dst);
    }
  } else if (N >= 64 && N < 128) {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 32, 16, 2>(nnz, N, keys, index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 32, 32, 2>(nnz, N, keys, index, src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 32, 32, 4>(nnz, N, keys, index, src, dst);
    }
  } else {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 64, 16, 2>(nnz, N, keys, index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 64, 32, 2>(nnz, N, keys, index, src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 64, 32, 4>(nnz, N, keys, index, src, dst);
    }
  }
  gflops = nnz * N / time / 1e6;
  out_file << data_name << "," << "naive" << "," << N << "," << keys << ","
           << time << "," << gflops << std::endl;
}

template <typename scalar_t>
void segreduce_dtree_rule(std::ofstream &out_file, char *data_name, int nnz,
                          int N, int keys, util::RamArray<Index> &index,
                          util::RamArray<DType> &src,
                          util::RamArray<DType> &dst) {
  float time = 0;
  float gflops = 0;
  int avg = nnz / keys;
  if (N < 8) {
    if (N <= 3.0) {
      if (N <= 1.5) {
        if (avg <= 502.41) {
          if (avg <= 288.4) {
            if (avg <= 1.15) {
              time = segreduce_pr_test<scalar_t, 1, 1, 2, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 1, 1, 8, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          } else {
            if (avg <= 492.02) {
              time = segreduce_pr_test<scalar_t, 1, 1, 4, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 1, 1, 8, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          }
        } else {
          if (avg <= 607.25) {
            if (avg <= 544.1) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 1, 1, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          } else {
            time = segreduce_pr_test<scalar_t, 1, 4, 1, 4, 32>(nnz, N, keys, index,
                                                          src, dst);
          }
        }
      } else {
        if (avg <= 492.42) {
          if (avg <= 288.4) {
            if (avg <= 84.56) {
              time = segreduce_pr_test<scalar_t, 2, 1, 1, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          } else {
            if (avg <= 491.67) {
              time = segreduce_pr_test<scalar_t, 2, 1, 4, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 2, 1, 2, 8, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          }
        } else {
          if (avg <= 505.15) {
            if (avg <= 494.17) {
              time = segreduce_pr_test<scalar_t, 1, 2, 2, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 8, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          } else {
            if (avg <= 544.1) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 1, 1, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          }
        }
      }
    } else {
      if (avg <= 3.24) {
        if (avg <= 1.15) {
          if (avg <= 1.15) {
            time = segreduce_pr_test<scalar_t, 2, 2, 2, 1, 16>(nnz, N, keys, index,
                                                          src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 1, 4, 4, 1, 16>(nnz, N, keys, index,
                                                          src, dst);
          }
        } else {
          if (avg <= 1.64) {
            if (avg <= 1.15) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          } else {
            if (avg <= 2.33) {
              time = segreduce_pr_test<scalar_t, 2, 2, 2, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 2, 2, 2, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          }
        }
      } else {
        if (avg <= 492.42) {
          if (avg <= 288.4) {
            if (avg <= 84.56) {
              time = segreduce_pr_test<scalar_t, 2, 2, 1, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          } else {
            if (avg <= 491.12) {
              time = segreduce_pr_test<scalar_t, 4, 1, 4, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 4, 1, 4, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          }
        } else {
          if (avg <= 505.15) {
            if (avg <= 494.17) {
              time = segreduce_pr_test<scalar_t, 1, 2, 2, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 4, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          } else {
            if (avg <= 607.25) {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 2, 32>(nnz, N, keys, index,
                                                            src, dst);
            }
          }
        }
      }
    }
  } else {
    if (N <= 48.0) {
      if (N <= 24.0) {
        if (N <= 12.0) {
          if (avg <= 49.8) {
            if (avg <= 8.67) {
              time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 8, 4, 8>(nnz, N, keys, index, src,
                                                      dst);
            }
          } else {
            if (avg <= 57.28) {
              time = segreduce_sr_test<scalar_t, 1, 8, 16, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 8, 8, 8>(nnz, N, keys, index, src,
                                                      dst);
            }
          }
        } else {
          if (avg <= 5.74) {
            if (avg <= 1.64) {
              time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 16, 8, 8>(nnz, N, keys, index, src,
                                                       dst);
            }
          } else {
            if (avg <= 38.23) {
              time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 16, 8, 8>(nnz, N, keys, index, src,
                                                       dst);
            }
          }
        }
      } else {
        if (avg <= 497.47) {
          if (avg <= 288.4) {
            if (avg <= 84.45) {
              time = segreduce_sr_test<scalar_t, 2, 16, 8, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 32, 4, 8>(nnz, N, keys, index, src,
                                                       dst);
            }
          } else {
            if (avg <= 492.42) {
              time = segreduce_sr_test<scalar_t, 2, 16, 8, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 32, 8, 8>(nnz, N, keys, index, src,
                                                       dst);
            }
          }
        } else {
          if (avg <= 505.15) {
            if (avg <= 502.41) {
              time = segreduce_sr_test<scalar_t, 1, 32, 4, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 64, 8, 4>(nnz, N, keys, index, src,
                                                       dst);
            }
          } else {
            time = segreduce_sr_test<scalar_t, 1, 32, 4, 8>(nnz, N, keys, index, src,
                                                     dst);
          }
        }
      }
    } else {
      if (N <= 96.0) {
        if (avg <= 288.4) {
          if (avg <= 14.9) {
            if (avg <= 13.83) {
              time = segreduce_sr_test<scalar_t, 2, 32, 8, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 32, 16, 8>(nnz, N, keys, index,
                                                        src, dst);
            }
          } else {
            if (avg <= 36.81) {
              time = segreduce_sr_test<scalar_t, 2, 32, 4, 8>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 32, 8, 8>(nnz, N, keys, index, src,
                                                       dst);
            }
          }
        } else {
          if (avg <= 497.47) {
            if (avg <= 491.32) {
              time = segreduce_sr_test<scalar_t, 2, 32, 16, 8>(nnz, N, keys, index,
                                                        src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 32, 16, 4>(nnz, N, keys, index,
                                                        src, dst);
            }
          } else {
            if (avg <= 499.7) {
              time = segreduce_sr_test<scalar_t, 2, 32, 8, 4>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 64, 4, 8>(nnz, N, keys, index, src,
                                                       dst);
            }
          }
        }
      } else {
        if (N <= 192.0) {
          if (avg <= 288.4) {
            if (avg <= 14.9) {
              time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(nnz, N, keys, index, src,
                                                       dst);
            }
          } else {
            if (avg <= 499.7) {
              time = segreduce_sr_test<scalar_t, 2, 64, 16, 2>(nnz, N, keys, index,
                                                        src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 64, 8, 8>(nnz, N, keys, index, src,
                                                       dst);
            }
          }
        } else {
          if (avg <= 15.21) {
            if (avg <= 3.95) {
              time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(nnz, N, keys, index, src,
                                                       dst);
            }
          } else {
            if (avg <= 41.46) {
              time = segreduce_sr_test<scalar_t, 2, 64, 4, 4>(nnz, N, keys, index, src,
                                                       dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(nnz, N, keys, index, src,
                                                       dst);
            }
          }
        }
      }
    }
  }
  gflops = nnz * N / time / 1e6;
  out_file << data_name << "," << "dtree" << "," << N << "," << keys << ","
           << time << "," << gflops << std::endl;
}