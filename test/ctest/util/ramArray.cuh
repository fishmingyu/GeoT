#ifndef RAM_ARRAY
#define RAM_ARRAY

#include "check.cuh"
#include <limits.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace util {

template <typename DType> struct deleter {
  void operator()(DType const *ptr) { delete[] ptr; }
};

template <typename DType> struct cudaDeleter {
  void operator()(DType const *ptr) { checkCudaError(cudaFree((void *)ptr)); }
};

template <class DType> class RamArray {
public:
  RamArray(int len);
  RamArray();
  void upload(); // upload
  void download();
  void create(int len); // create a new array and fill random
  void create(int len, std::vector<DType> vec);
  void fill_random_h();
  void fill_zero_h();
  void fill_default_one();
  void reset();
  ~RamArray();
  std::shared_ptr<DType> h_array;
  std::shared_ptr<DType> d_array;
  int len;

private:
  size_t size;
};

template <typename DType> RamArray<DType>::RamArray() {}

template <typename DType> RamArray<DType>::RamArray(int _len) {
  len = _len;
  size = len * sizeof(DType);
  h_array = std::shared_ptr<DType>(new DType[len], deleter<DType>());
  d_array = std::shared_ptr<DType>(nullptr, cudaDeleter<DType>());
  checkCudaError(cudaMalloc((void **)&d_array, size));
}

template <typename DType> void RamArray<DType>::create(int _len) {
  len = _len;
  size = len * sizeof(DType);
  h_array = std::shared_ptr<DType>(new DType[len], deleter<DType>());
  d_array = std::shared_ptr<DType>(nullptr, cudaDeleter<DType>());
  checkCudaError(cudaMalloc((void **)&d_array, size));
}

template <typename DType>
void RamArray<DType>::create(int _len, std::vector<DType> vec) {
  len = _len;
  size = len * sizeof(DType);
  h_array = std::shared_ptr<DType>(new DType[len], deleter<DType>());
  std::copy(vec.begin(), vec.end(), h_array.get());
  d_array = std::shared_ptr<DType>(nullptr, cudaDeleter<DType>());
  checkCudaError(cudaMalloc((void **)&d_array, size));
}

template <typename DType> RamArray<DType>::~RamArray() {}

template <typename DType> void RamArray<DType>::fill_random_h() {
  for (int i = 0; i < len; i++) {
    h_array.get()[i] = (DType)(std::rand() % 10) / 10;
  }
}

template <typename DType> void RamArray<DType>::fill_zero_h() {
  std::fill(h_array.get(), h_array.get() + len, 0x0);
}

template <typename DType> void RamArray<DType>::fill_default_one() {
  std::fill(h_array.get(), h_array.get() + len, 1);
}

template <typename DType> void RamArray<DType>::reset() {
  fill_zero_h();
  checkCudaError(cudaMemset(d_array.get(), 0, size));
}

template <typename DType> void RamArray<DType>::upload() {
  checkCudaError(cudaMemcpy((void *)d_array.get(), (void *)h_array.get(), size,
                            cudaMemcpyHostToDevice));
}

template <typename DType> void RamArray<DType>::download() {
  checkCudaError(cudaMemcpy((void *)h_array.get(), (void *)d_array.get(), size,
                            cudaMemcpyDeviceToHost));
}
} // namespace util

#endif // RAM_ARRAY
