#ifndef GPUTIMER
#define GPUTIMER

#include "check.cuh"

namespace util {
class gpuTimer {
public:
  gpuTimer() {
    checkCudaError(cudaEventCreate(&t1));
    checkCudaError(cudaEventCreate(&t2));
  }
  void start() { cudaEventRecord(t1, 0); }
  void end() {
    cudaEventRecord(t2, 0);
    cudaEventSynchronize(t1);
    cudaEventSynchronize(t2);
  }
  float elapsed() {
    cudaEventElapsedTime(&time, t1, t2);
    return time;
  }

private:
  float time;
  cudaEvent_t t1, t2;
};
} // namespace util

#endif