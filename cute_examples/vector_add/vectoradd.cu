#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>
#include "cute/tensor.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "vector_add_kernel.cuh"

using namespace cute;

int main() {
  GPU_Clock gc;
  float time;

  constexpr int kElemPerThread = 8;
  device_init(0);
  thrust::device_vector<half> d_x, d_y, d_z;
  for (int kNum = 1024; kNum <= 1024*1024*16; kNum *= 2) {
    thrust::host_vector<half> h_x(kNum), h_y(kNum), h_z(kNum);
    dim3 dimBlock(kNum/kElemPerThread);

    for (int i=0; i<kNum; ++i) h_x[i] = static_cast<half>(i);
    for (int i=0; i<kNum; ++i) h_y[i] = static_cast<half>(i);
    for (int i=0; i<kNum; ++i) h_z[i] = static_cast<half>(-1);

    d_x = h_x, d_y = h_y, d_z = h_z;

    // Warm up
    vector_add_kernel<kElemPerThread><<<1, dimBlock>>>(d_z.data().get(), kNum, d_x.data().get(), d_y.data().get(), 1.0, 1.0, 0.0);

    gc.start();
    for(int i=0; i<100; ++i)
      vector_add_kernel<kElemPerThread><<<1, dimBlock>>>(d_z.data().get(), kNum, d_x.data().get(), d_y.data().get(), 1.0, 1.0, 0.0);
    time = gc.milliseconds();
    std::cout << "VectorAdd of " << kNum << " elements, Time: " << time/100 << " ms, Throughput: " 
              << (kNum * sizeof(half) * 3) / (time / 100 / 1e3) / 1e9 << " GB/s" << std::endl;
  }

  return 0;
}