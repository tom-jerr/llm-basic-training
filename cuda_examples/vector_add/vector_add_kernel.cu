#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void vector_add_kernel(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void vector_add_kernel_vec4(float *a, float *b, float *c, int n) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < n) {                   // 确保不会越界
    float4 a_vec = FLOAT4(a[idx]); // 从正确的地址读取
    float4 b_vec = FLOAT4(b[idx]); // 从正确的地址读取
    float4 c_vec;
    c_vec.x = a_vec.x + b_vec.x;
    c_vec.y = a_vec.y + b_vec.y;
    c_vec.z = a_vec.z + b_vec.z;
    c_vec.w = a_vec.w + b_vec.w;
    FLOAT4(c[idx]) = c_vec;
  }
}

// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}

__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx < N) {
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

// PyTorch Wrappers

void vector_add_torch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
}

void vector_add_vec4_torch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.numel();
    int threads = 256 / 4;
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    vector_add_kernel_vec4<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
}

void vector_add_fp16_torch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elementwise_add_f16_kernel<<<blocks, threads>>>(reinterpret_cast<half*>(a.data_ptr<at::Half>()), reinterpret_cast<half*>(b.data_ptr<at::Half>()), reinterpret_cast<half*>(c.data_ptr<at::Half>()), n);
}

void vector_add_fp16x8_torch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.numel();
    int threads = 256;
    int elements_per_thread = 8;
    int blocks = (n + threads * elements_per_thread - 1) / (threads * elements_per_thread);
    elementwise_add_f16x8_kernel<<<blocks, threads>>>(reinterpret_cast<half*>(a.data_ptr<at::Half>()), reinterpret_cast<half*>(b.data_ptr<at::Half>()), reinterpret_cast<half*>(c.data_ptr<at::Half>()), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add_torch, "Vector Add FP32");
  m.def("vector_add_vec4", &vector_add_vec4_torch, "Vector Add FP32 Vec4");
  m.def("vector_add_fp16", &vector_add_fp16_torch, "Vector Add FP16");
  m.def("vector_add_fp16x8", &vector_add_fp16x8_torch, "Vector Add FP16x8");
}
