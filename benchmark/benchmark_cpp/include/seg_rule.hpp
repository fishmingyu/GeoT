#include "util.cuh"

template <typename scalar_t>
void segreduce_naive_rule(std::ofstream &out_file, char *data_name, int size,
                          int N, int keys, util::RamArray<Index> &index,
                          util::RamArray<DType> &src,
                          util::RamArray<DType> &dst) {
  float time = 0;
  float gflops = 0;
  int avg_key_len = size / keys;
  if (N >= 1 && N <= 4) {
    time = segreduce_pr_test<scalar_t, 1, 1, 2, 4, 32>(size, N, keys, index,
                                                       src, dst);
  } else if (N > 4 && N <= 16) {
    time = segreduce_pr_test<scalar_t, 2, 2, 2, 4, 32>(size, N, keys, index,
                                                       src, dst);
  } else if (N > 16 && N < 64) {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 16, 16, 2>(size, N, keys, index,
                                                       src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 16, 32, 2>(size, N, keys, index,
                                                       src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 16, 32, 4>(size, N, keys, index,
                                                       src, dst);
    }
  } else if (N >= 64 && N < 128) {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 32, 16, 2>(size, N, keys, index,
                                                       src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 32, 32, 2>(size, N, keys, index,
                                                       src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 32, 32, 4>(size, N, keys, index,
                                                       src, dst);
    }
  } else {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 64, 16, 2>(size, N, keys, index,
                                                       src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 64, 32, 2>(size, N, keys, index,
                                                       src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 64, 32, 4>(size, N, keys, index,
                                                       src, dst);
    }
  }
  gflops = size * N / time / 1e6;
  out_file << data_name << ","
           << "naive"
           << "," << N << "," << size << "," << time << "," << gflops
           << std::endl;
}

template <typename scalar_t>
void segreduce_dtree_rule(std::ofstream &out_file, char *data_name, int size,
                          int N, int keys, util::RamArray<Index> &index,
                          util::RamArray<DType> &src,
                          util::RamArray<DType> &dst) {
  float time = 0;
  float gflops = 0;
  int avg = size / keys;
  if (N < 8) {
    if (size <= 6677.0) {
      if (size <= 2404.0) {
        if (N <= 3.0) {
          if (size <= 1435.0) {
            if (size <= 1370.5) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 2, 32>(
                  size, N, keys, index, src, dst);
            }
          } else {
            if (N <= 1.5) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            }
          }
        } else {
          if (size <= 1937.0) {
            if (size <= 1062.5) {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 2, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 2, 32>(
                  size, N, keys, index, src, dst);
            }
          } else {
            if (size <= 2202.0) {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 2, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            }
          }
        }
      } else {
        if (N <= 1.5) {
          if (size <= 4729.5) {
            if (avg <= 30.3) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 8, 32>(
                  size, N, keys, index, src, dst);
            }
          } else {
            if (size <= 6285.5) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            }
          }
        } else {
          if (size <= 3948.0) {
            if (size <= 2961.0) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            }
          } else {
            if (avg <= 38.33) {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 4, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            }
          }
        }
      }
    } else {
      if (avg <= 1.15) {
        if (N <= 3.0) {
          if (size <= 703065.0) {
            if (size <= 351532.5) {
              time = segreduce_pr_test<scalar_t, 1, 1, 2, 4, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 2, 1, 2, 8, 32>(
                  size, N, keys, index, src, dst);
            }
          } else {
            if (N <= 1.5) {
              time = segreduce_pr_test<scalar_t, 1, 1, 2, 8, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 2, 2, 2, 32>(
                  size, N, keys, index, src, dst);
            }
          }
        } else {
          if (size <= 1406130.0) {
            if (avg <= 1.15) {
              time = segreduce_pr_test<scalar_t, 2, 2, 2, 1, 16>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 4, 4, 1, 16>(
                  size, N, keys, index, src, dst);
            }
          } else {
            time = segreduce_pr_test<scalar_t, 2, 2, 4, 1, 16>(size, N, keys,
                                                               index, src, dst);
          }
        }
      } else {
        if (N <= 1.5) {
          if (size <= 359204.0) {
            if (avg <= 7.19) {
              time = segreduce_pr_test<scalar_t, 1, 1, 1, 8, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 1, 1, 8, 32>(
                  size, N, keys, index, src, dst);
            }
          } else {
            if (size <= 2188236.0) {
              time = segreduce_pr_test<scalar_t, 1, 1, 2, 8, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 1, 1, 2, 8, 32>(
                  size, N, keys, index, src, dst);
            }
          }
        } else {
          if (avg <= 35.71) {
            if (size <= 70469.0) {
              time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 4, 1, 2, 4, 32>(
                  size, N, keys, index, src, dst);
            }
          } else {
            if (size <= 331916.5) {
              time = segreduce_pr_test<scalar_t, 2, 2, 1, 8, 32>(
                  size, N, keys, index, src, dst);
            } else {
              time = segreduce_pr_test<scalar_t, 4, 1, 2, 4, 32>(
                  size, N, keys, index, src, dst);
            }
          }
        }
      }
    }
  } else {
    if (N <= 48.0) {
      if (N <= 24.0) {
        if (size <= 20974.5) {
          if (avg <= 9.39) {
            if (size <= 7155.0) {
              time = segreduce_sr_test<scalar_t, 1, 32, 4, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 32, 4, 8>(size, N, keys,
                                                              index, src, dst);
            }
          } else {
            if (avg <= 35.43) {
              time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(size, N, keys,
                                                              index, src, dst);
            }
          }
        } else {
          if (N <= 12.0) {
            if (size <= 93022.5) {
              time = segreduce_sr_test<scalar_t, 1, 8, 4, 8>(size, N, keys,
                                                             index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 8, 8, 8>(size, N, keys,
                                                             index, src, dst);
            }
          } else {
            if (size <= 247962.5) {
              time = segreduce_sr_test<scalar_t, 1, 16, 8, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 8, 8, 8>(size, N, keys,
                                                             index, src, dst);
            }
          }
        }
      } else {
        if (size <= 55061.5) {
          if (size <= 6521.0) {
            if (size <= 6413.5) {
              time = segreduce_sr_test<scalar_t, 1, 32, 4, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 64, 8, 4>(size, N, keys,
                                                              index, src, dst);
            }
          } else {
            if (avg <= 502.41) {
              time = segreduce_sr_test<scalar_t, 1, 32, 4, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 64, 8, 4>(size, N, keys,
                                                              index, src, dst);
            }
          }
        } else {
          if (avg <= 4.22) {
            if (size <= 2351251.0) {
              time = segreduce_sr_test<scalar_t, 2, 16, 8, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 16, 32, 8>(size, N, keys,
                                                               index, src, dst);
            }
          } else {
            if (size <= 220246.0) {
              time = segreduce_sr_test<scalar_t, 2, 16, 8, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 16, 8, 8>(size, N, keys,
                                                              index, src, dst);
            }
          }
        }
      }
    } else {
      if (N <= 96.0) {
        if (size <= 10487.0) {
          if (avg <= 1.15) {
            time = segreduce_sr_test<scalar_t, 1, 32, 4, 8>(size, N, keys,
                                                            index, src, dst);
          } else {
            if (avg <= 38.97) {
              time = segreduce_sr_test<scalar_t, 1, 64, 4, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 1, 64, 4, 8>(size, N, keys,
                                                              index, src, dst);
            }
          }
        } else {
          if (size <= 1094118.0) {
            if (avg <= 492.02) {
              time = segreduce_sr_test<scalar_t, 2, 32, 8, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 32, 16, 8>(size, N, keys,
                                                               index, src, dst);
            }
          } else {
            if (size <= 1127504.0) {
              time = segreduce_sr_test<scalar_t, 2, 32, 32, 8>(size, N, keys,
                                                               index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 32, 16, 8>(size, N, keys,
                                                               index, src, dst);
            }
          }
        }
      } else {
        if (size <= 18918.5) {
          if (size <= 5341.5) {
            if (size <= 3142.5) {
              time = segreduce_sr_test<scalar_t, 1, 64, 4, 8>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 64, 4, 4>(size, N, keys,
                                                              index, src, dst);
            }
          } else {
            if (avg <= 36.3) {
              time = segreduce_sr_test<scalar_t, 2, 32, 8, 4>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(size, N, keys,
                                                              index, src, dst);
            }
          }
        } else {
          if (size <= 747442.0) {
            if (avg <= 492.8) {
              time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(size, N, keys,
                                                              index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 64, 16, 4>(size, N, keys,
                                                               index, src, dst);
            }
          } else {
            if (avg <= 15.04) {
              time = segreduce_sr_test<scalar_t, 2, 64, 16, 4>(size, N, keys,
                                                               index, src, dst);
            } else {
              time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(size, N, keys,
                                                              index, src, dst);
            }
          }
        }
      }
    }
  }
  gflops = size * N / time / 1e6;
  out_file << data_name << ","
           << "dtree"
           << "," << N << "," << size << "," << time << "," << gflops
           << std::endl;
}
