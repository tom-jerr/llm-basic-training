#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(const float *a, const float *b, float *c,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void elementwise_add_f32x4_kernel(const float *a, const float *b,
                                         float *c, int n) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < n) {
    float4 a4 = reinterpret_cast<const float4 *>(a)[idx / 4];
    float4 b4 = reinterpret_cast<const float4 *>(b)[idx / 4];
    float4 c4;
    c4.x = a4.x + b4.x;
    c4.y = a4.y + b4.y;
    c4.z = a4.z + b4.z;
    c4.w = a4.w + b4.w;
    reinterpret_cast<float4 *>(c)[idx / 4] = c4;
  }
}

__global__ void elementwise_add_f16_kernel(const half *a, const half *b, half *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = __hadd(a[idx], b[idx]);
  }
}


__global__ void elementwise_add_f16x8_kernel(const half *a, const half *b, half *c, int n) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx < n) {
    float4 a4 = reinterpret_cast<const float4 *>(a)[idx / 8];
    float4 b4 = reinterpret_cast<const float4 *>(b)[idx / 8];
    float4 c4;
    half2 *a2 = reinterpret_cast<half2 *>(&a4);
    half2 *b2 = reinterpret_cast<half2 *>(&b4);
    half2 *c2 = reinterpret_cast<half2 *>(&c4);
    c2[0] = __hadd2(a2[0], b2[0]);
    c2[1] = __hadd2(a2[1], b2[1]);
    c2[2] = __hadd2(a2[2], b2[2]);
    c2[3] = __hadd2(a2[3], b2[3]);
    reinterpret_cast<float4 *>(c)[idx / 8] = c4;
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_VECTOR_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_VECTOR_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_VECTOR_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_VECTOR_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_VECTOR_ADD(f16x8, torch::kHalf, half, 8)


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)

}