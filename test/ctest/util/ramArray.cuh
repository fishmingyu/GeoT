#ifndef RAM_ARRAY
#define RAM_ARRAY

#include "check.cuh"
#include <limits.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace util {

template <typename ValueType> struct deleter {
  void operator()(ValueType const *ptr) { delete[] ptr; }
};

template <typename ValueType> struct cudaDeleter {
  void operator()(ValueType const *ptr) {
    checkCudaError(cudaFree((void *)ptr));
  }
};

template <class ValueType> class RamArray {
public:
  RamArray(int len);
  RamArray();
  void tocuda(); // tocuda
  void tocpu();
  void create(int len); // create a new array and fill random
  void create(int len, std::vector<ValueType> vec);
  void fill_random_h();
  void fill_zero_h();
  void fill_default_one();
  void reset();
  ~RamArray();
  std::shared_ptr<ValueType> h_array;
  std::shared_ptr<ValueType> d_array;
  int len;

private:
  size_t size;
};

template <typename ValueType> RamArray<ValueType>::RamArray() {}

template <typename ValueType> RamArray<ValueType>::RamArray(int _len) {
  len = _len;
  size = len * sizeof(ValueType);
  h_array =
      std::shared_ptr<ValueType>(new ValueType[len], deleter<ValueType>());
  d_array = std::shared_ptr<ValueType>(nullptr, cudaDeleter<ValueType>());
  checkCudaError(cudaMalloc((void **)&d_array, size));
}

template <typename ValueType> void RamArray<ValueType>::create(int _len) {
  len = _len;
  size = len * sizeof(ValueType);
  h_array =
      std::shared_ptr<ValueType>(new ValueType[len], deleter<ValueType>());
  d_array = std::shared_ptr<ValueType>(nullptr, cudaDeleter<ValueType>());
  checkCudaError(cudaMalloc((void **)&d_array, size));
}

template <typename ValueType>
void RamArray<ValueType>::create(int _len, std::vector<ValueType> vec) {
  len = _len;
  size = len * sizeof(ValueType);
  h_array =
      std::shared_ptr<ValueType>(new ValueType[len], deleter<ValueType>());
  std::copy(vec.begin(), vec.end(), h_array.get());
  d_array = std::shared_ptr<ValueType>(nullptr, cudaDeleter<ValueType>());
  checkCudaError(cudaMalloc((void **)&d_array, size));
}

template <typename ValueType> RamArray<ValueType>::~RamArray() {}

template <typename ValueType> void RamArray<ValueType>::fill_random_h() {
  for (int i = 0; i < len; i++) {
    h_array.get()[i] = (ValueType)(std::rand() % 10) / 10;
  }
}

template <typename ValueType> void RamArray<ValueType>::fill_zero_h() {
  std::fill(h_array.get(), h_array.get() + len, 0x0);
}

template <typename ValueType> void RamArray<ValueType>::fill_default_one() {
  std::fill(h_array.get(), h_array.get() + len, 1);
}

template <typename ValueType> void RamArray<ValueType>::reset() {
  fill_zero_h();
  checkCudaError(cudaMemset(d_array.get(), 0, size));
}

template <typename ValueType> void RamArray<ValueType>::tocuda() {
  checkCudaError(cudaMemcpy((void *)d_array.get(), (void *)h_array.get(), size,
                            cudaMemcpyHostToDevice));
}

template <typename ValueType> void RamArray<ValueType>::tocpu() {
  checkCudaError(cudaMemcpy((void *)h_array.get(), (void *)d_array.get(), size,
                            cudaMemcpyDeviceToHost));
}
} // namespace util

#endif // RAM_ARRAY
