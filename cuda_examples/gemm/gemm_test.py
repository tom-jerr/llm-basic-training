import torch
from torch.utils.cpp_extension import load
import time
import sys

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="gemm_lib",
    sources=[
        "gemm.cu",
        # "gemm_async.cu",
        # "gemm_wmma_tf32_stage.cu",
        "gemm_cublas.cu",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

def verify_correctness(func, a, b, c_ref, name, **kwargs):
    c = torch.zeros_like(c_ref)
    try:
        if kwargs.get('stages', 0) > 1:
            func(a, b, c, kwargs['stages'], kwargs.get('swizzle', False), kwargs.get('swizzle_stride', 1))
        else:
            func(a, b, c)
        
        torch.cuda.synchronize()
        
        if not torch.allclose(c, c_ref, atol=1e-2, rtol=1e-2):
            diff = (c - c_ref).abs().max()
            print(f"FAIL: {name}, max diff: {diff}")
        else:
            print(f"PASS: {name}")
    except Exception as e:
        print(f"ERROR: {name}, exception: {e}")


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    print(f"Testing with M={M}, N={N}, K={K}")
    
    a = torch.randn((M, K), dtype=torch.float).cuda()
    b = torch.randn((K, N), dtype=torch.float).cuda()
    c_ref = torch.matmul(a, b)
    c = torch.zeros_like(c_ref)

    # Verify Correctness
    print("-" * 20 + " Correctness Check " + "-" * 20)
    # verify_correctness(lib.gemm_t_8x8_sliced_k_f32x4, a, b, c_ref, "f32x4(t8x8sk)")
    # verify_correctness(lib.gemm_t_8x8_sliced_k_f32x4_bcf, a, b, c_ref, "f32x4(t8x8bcf)")
