#include <torch/extension.h>
#include <cuda_fp16.h>

// Declaration of the CUDA launcher
void launch_gemm_simple(half* C_ptr, const half* A_ptr, const half* B_ptr, int m, int n, int k);

void gemm_custom(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kHalf, "A must be half");
    TORCH_CHECK(B.dtype() == torch::kHalf, "B must be half");
    TORCH_CHECK(C.dtype() == torch::kHalf, "C must be half");

    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(0); 

    TORCH_CHECK(B.size(1) == k, "B must have shape (n, k)");
    TORCH_CHECK(C.size(0) == m && C.size(1) == n, "C must have shape (m, n)");

    launch_gemm_simple(
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        m, n, k
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_custom", &gemm_custom, "Custom GEMM (CUDA)");
}
