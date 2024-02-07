#ifndef CHECK_
#define CHECK_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define checkCuSparseError(a)                                                  \
  do {                                                                         \
    if (CUSPARSE_STATUS_SUCCESS != (a)) {                                      \
      fprintf(stderr, "CuSparse runTime error in line %d of file %s : %s \n",  \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define checkCudaError(a)                                                      \
  do {                                                                         \
    if (cudaSuccess != (a)) {                                                  \
      fprintf(stderr, "Cuda runTime error in line %d of file %s : %s \n",      \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
}

namespace util {

template <typename DType>
bool check_result(int M, int N, DType *C, DType *C_ref, int errors = 10) {
  bool passed = true;
  int cnt = 0;
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      DType c = C[i * N + j];
      DType c_ref = C_ref[i * N + j];
      if (fabs(c - c_ref) > 1e-2 * fabs(c_ref)) {
        cnt++;
        if (cnt < errors) {
          printf("Wrong result: i = %ld, j = %ld, result = %lf, reference = "
                 "%lf.\n",
                 i, j, c, c_ref);
        }
        passed = false;
        break;
      }
    }
  }
  return passed;
}

template <typename ValueType, typename IndexType>
void segment_coo_sequencial(const ValueType *src, const IndexType *index,
                            const int nnz, const int N, const int dst_len,
                            ValueType *dst) {
  for (int i = 0; i < nnz; i++) {
#pragma unroll
    for (int j = 0; j < N; j++) {
      dst[index[i] * N + j] += src[i * N + j];
    }
  }
}

template <typename ValueType, typename IndexType>
void gather_sequencial(const ValueType *src, const IndexType *index, 
                       const int nnz, const int N, const int dst_len,
                       ValueType *dst) {
  for (int i = 0; i < nnz; i++) {
#pragma unroll
    for (int j = 0; j < N; j++) {
      dst[i * N + j] = src[index[i] * N + j];
    }
  }
}

template <typename ValueType, typename IndexType>
bool checkSegScan(ValueType *dst, ValueType *src, IndexType *index, int nnz,
                  int N, int dst_len) {
  std::vector<ValueType> dst_cpu(dst_len * N, 0);
  segment_coo_sequencial<ValueType, IndexType>(src, index, nnz, N, dst_len,
                                               dst_cpu.data());
  std::cout << "Finish golden" << std::endl;
  for (int i = 0; i < N * dst_len; i++) {
    if (fabs(dst[i] - dst_cpu[i]) > 1e-2 * fabs(dst_cpu[i])) {
      printf("Error[%d][%d]: dst = %f, dst_cpu = %f\n", i / N, i % N, dst[i],
             dst_cpu[i]);
      return false;
    }
  }
  printf("Check passed!\n");
  return true;
}

template <typename ValueType, typename IndexType>
bool checkGather(ValueType *dst, ValueType *src, IndexType *index, int nnz,
                 int N, int dst_len) {
  std::vector<ValueType> dst_cpu(nnz * N, 0);
  gather_sequencial<ValueType, IndexType>(src, index, nnz, N, dst_len,
                                          dst_cpu.data());
  // std::cout << "Finish golden" << std::endl;
  for (int i = 0; i < N * nnz; i++) {
    if (fabs(dst[i] - dst_cpu[i]) > 1e-2 * fabs(dst_cpu[i])) {
      printf("Error[%d][%d]: dst = %f, dst_cpu = %f\n", i / N, i % N, dst[i],
             dst_cpu[i]);
      return false;
    }
  }
  // printf("Check passed!\n");
  return true;
}

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
}

} // namespace util

#endif // CHECK_