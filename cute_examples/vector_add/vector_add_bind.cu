#include <torch/extension.h>
#include "vector_add_kernel.cuh"

void vector_add_cuda(torch::Tensor z, torch::Tensor x, torch::Tensor y, float a, float b, float c) {
    int num = x.numel();
    const int kElemPerThread = 8;
    
    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int num_threads_needed = (num + kElemPerThread - 1) / kElemPerThread;
    int num_blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;

    vector_add_kernel<kElemPerThread><<<num_blocks, threads_per_block>>>(
        reinterpret_cast<half*>(z.data_ptr<at::Half>()),
        num,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(y.data_ptr<at::Half>()),
        __float2half(a),
        __float2half(b),
        __float2half(c)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add_cuda, "Vector Add (CUDA)");
}
