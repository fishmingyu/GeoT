#ifndef CHECK_
#define CHECK_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef int Index;
typedef float DType;
const int RefThreadPerBlock = 256;
#define FULLMASK 0xffffffff
#define GROUP_SIZE 16
#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
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

#define checkSpMMError(a) \
  do {                    \
    out_feature.reset();  \
    a<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());                    \
    out_feature.download();\
    bool pass = util::check_result(H.nrow, feature_size, out_feature.h_array.get(), out_ref.h_array.get());\
    if (pass) {             \
      std::cout<<"Passed!"<<std::endl;\
    } else {                \
      std::cout<<"Not Passed!"<<std::endl;\
    }   \
  } while(0)                \

#define checkSpMMsuffix \
  do {                    \
    out_feature.download();\
    bool pass = util::check_result(H.nrow, feature_size, out_feature.h_array.get(), out_ref.h_array.get());\
    if (pass) {             \
      std::cout<<"Passed!"<<std::endl;\
    } else {                \
      std::cout<<"Not Passed!"<<std::endl;\
    }   \
    out_feature.reset();    \
  } while(0)                \
  
#define SHFL_DOWN_REDUCE(v) \
v += __shfl_down_sync(FULLMASK, v, 16);\
v += __shfl_down_sync(FULLMASK, v, 8);\
v += __shfl_down_sync(FULLMASK, v, 4);\
v += __shfl_down_sync(FULLMASK, v, 2);\
v += __shfl_down_sync(FULLMASK, v, 1);

#define SEG_SHFL_SCAN(v, tmpv, segid, tmps) \
tmpv = __shfl_down_sync(FULLMASK, v, 1); tmps = __shfl_down_sync(FULLMASK, segid, 1); if (tmps == segid && lane_id < 31) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 2); tmps = __shfl_down_sync(FULLMASK, segid, 2); if (tmps == segid && lane_id < 30) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 4); tmps = __shfl_down_sync(FULLMASK, segid, 4); if (tmps == segid && lane_id < 28) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 8); tmps = __shfl_down_sync(FULLMASK, segid, 8); if (tmps == segid && lane_id < 24) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 16); tmps = __shfl_down_sync(FULLMASK, segid, 16); if (tmps == segid && lane_id < 16) v += tmpv;

template <typename index_t>
__device__ __forceinline__ index_t
binary_search_segment_number(const index_t *seg_offsets, const index_t n_seg,
                             const index_t n_elem, const index_t elem_id) {
  index_t lo = 1, hi = n_seg, mid;
  while (lo < hi) {
    mid = (lo + hi) >> 1;
    if (seg_offsets[mid] <= elem_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (hi - 1);
}

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
          printf(
              "Wrong result: i = %ld, j = %ld, result = %lf, reference = %lf.\n",
              i, j, c, c_ref);
        }
        passed = false;
        break;
      }
    }
  }
  return passed;
}

// Compute spmm correct numbers. All arrays are host memory locations.
template <typename Index, typename DType>
void spmm_reference_host(int M,       // number of A-rows
                         int feature, // number of B_columns
                         Index *csr_indptr, Index *csr_indices,
                         DType *csr_values, // three arrays of A's CSR format
                         DType *B,          // assume row-major
                         DType *C_ref)      // assume row-major
{
  for (int64_t i = 0; i < M; i++) {
    Index begin = csr_indptr[i];
    Index end = csr_indptr[i + 1];
    for (Index p = begin; p < end; p++) {
      int k = csr_indices[p];
      DType val = csr_values[p];
      for (int64_t j = 0; j < feature; j++) {
        C_ref[i * feature + j] += val * B[k * feature + j];
      }
    }
  }
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