#include "cuda_check.cuh"
#include "vector_add.cuh"
#include <iostream>
// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void vector_add(const float *a, const float *b, float *c, int n) {
  float *d_a, *d_b, *d_c;
  size_t bytes = n * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

  // 在内核启动后立即检查错误
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

void vector_add_cpu(const float *a, const float *b, float *c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}
