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
    time = segreduce_pr_test<scalar_t, 1, 1, 2, 4, 32>(nnz, N, keys,
                                                              index, src, dst);
  } else if (N > 4 && N <= 16) {
    time = segreduce_pr_test<scalar_t, 2, 2, 2, 4, 32>(nnz, N, keys,
                                                              index, src, dst);
  } else if (N > 16 && N < 64) {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 16, 16, 2>(nnz, N, keys,
                                                              index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 16, 32, 2>(nnz, N, keys,
                                                              index, src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 16, 32, 4>(nnz, N, keys,
                                                              index, src, dst);
    }
  } else if (N >= 64 && N < 128) {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 32, 16, 2>(nnz, N, keys,
                                                              index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 32, 32, 2>(nnz, N, keys,
                                                              index, src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 32, 32, 4>(nnz, N, keys,
                                                              index, src, dst);
    }
  } else {
    if (avg_key_len < 16) {
      time = segreduce_sr_test<scalar_t, 2, 64, 16, 2>(nnz, N, keys,
                                                              index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      time = segreduce_sr_test<scalar_t, 2, 64, 32, 2>(nnz, N, keys,
                                                              index, src, dst);
    } else {
      time = segreduce_sr_test<scalar_t, 2, 64, 32, 4>(nnz, N, keys,
                                                              index, src, dst);
    }
  }
  gflops = nnz * N / time / 1e6;
  out_file << data_name << ","
           << "naive"
           << "," << N << "," << keys << "," << time << "," << gflops
           << std::endl;
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
    if (N <= 1.5) {
      if (avg <= 41.46) {
        if (avg <= 18.37) {
          if (avg <= 17.96) {
            time = segreduce_pr_test<scalar_t, 1, 1, 1, 8, 32>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 1, 1, 1, 4, 32>(nnz, N, keys, index, src, dst);
          }
        } else {
          if (avg <= 18.75) {
            time = segreduce_pr_test<scalar_t, 1, 1, 2, 8, 32>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 1, 1, 1, 8, 32>(nnz, N, keys, index, src, dst);
          }
        }
      } else {
        if (avg <= 57.24) {
          if (avg <= 49.8) {
            time = segreduce_pr_test<scalar_t, 1, 1, 1, 8, 32>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 1, 1, 4, 8, 32>(nnz, N, keys, index, src, dst);
          }
        } else {
          if (avg <= 491.02) {
            time = segreduce_pr_test<scalar_t, 1, 1, 1, 8, 32>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 1, 1, 2, 8, 32>(nnz, N, keys, index, src, dst);
          }
        }
      }
    } else {
      if (avg <= 40.33) {
        if (avg <= 1.64) {
          if (N <= 3.0) {
            time = segreduce_pr_test<scalar_t, 1, 2, 1, 4, 32>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 1, 2, 2, 2, 16>(nnz, N, keys, index, src, dst);
          }
        } else {
          if (avg <= 38.74) {
            time = segreduce_pr_test<scalar_t, 2, 2, 1, 4, 32>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 1, 2, 1, 2, 32>(nnz, N, keys, index, src, dst);
          }
        }
      } else {
        if (avg <= 505.15) {
          if (avg <= 288.4) {
            time = segreduce_pr_test<scalar_t, 2, 2, 1, 4, 32>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 2, 2, 2, 4, 32>(nnz, N, keys, index, src, dst);
          }
        } else {
          if (N <= 3.0) {
            time = segreduce_pr_test<scalar_t, 1, 2, 1, 2, 32>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_pr_test<scalar_t, 1, 4, 1, 2, 32>(nnz, N, keys, index, src, dst);
          }
        }
      }
    }
  } else {
    if (N <= 24.0) {
      if (N <= 12.0) {
        if (avg <= 8.67) {
          if (avg <= 1.15) {
            time = segreduce_sr_test<scalar_t, 1, 8, 4, 8>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(nnz, N, keys, index, src, dst);
          }
        } else {
          if (avg <= 38.74) {
            time = segreduce_sr_test<scalar_t, 1, 8, 4, 8>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_sr_test<scalar_t, 1, 8, 8, 8>(nnz, N, keys, index, src, dst);
          }
        }
      } else {
        if (avg <= 497.47) {
          if (avg <= 49.8) {
            time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_sr_test<scalar_t, 1, 16, 8, 8>(nnz, N, keys, index, src, dst);
          }
        } else {
          if (avg <= 505.15) {
            time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_sr_test<scalar_t, 1, 16, 4, 8>(nnz, N, keys, index, src, dst);
          }
        }
      }
    } else {
      if (N <= 48.0) {
        if (avg <= 14.9) {
          if (avg <= 11.16) {
            time = segreduce_sr_test<scalar_t, 2, 32, 8, 8>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_sr_test<scalar_t, 2, 32, 8, 8>(nnz, N, keys, index, src, dst);
          }
        } else {
          if (avg <= 35.68) {
            time = segreduce_sr_test<scalar_t, 2, 16, 4, 8>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_sr_test<scalar_t, 2, 32, 8, 8>(nnz, N, keys, index, src, dst);
          }
        }
      } else {
        if (avg <= 14.9) {
          if (avg <= 13.83) {
            time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_sr_test<scalar_t, 2, 64, 16, 4>(nnz, N, keys, index, src, dst);
          }
        } else {
          if (avg <= 41.46) {
            time = segreduce_sr_test<scalar_t, 2, 64, 4, 4>(nnz, N, keys, index, src, dst);
          } else {
            time = segreduce_sr_test<scalar_t, 2, 64, 8, 4>(nnz, N, keys, index, src, dst);
          }
        }
      }
    }
  }
  gflops = nnz * N / time / 1e6;
  out_file << data_name << ","
           << "dtree"
           << "," << N << "," << keys << "," << time << "," << gflops
           << std::endl;
}
