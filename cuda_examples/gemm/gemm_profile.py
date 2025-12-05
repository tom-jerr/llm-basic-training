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
        "gemm_mma.cu",
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

def warmup(func, a, b,c, warmup=2):
  for _ in range(warmup):  
    func(a, b, c)
  torch.cuda.synchronize()

if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    print(f"Profiling with M={M}, N={N}, K={K}")
    
    a = torch.randn((M, K), dtype=torch.float).cuda()
    b = torch.randn((K, N), dtype=torch.float).cuda()
    c = torch.zeros((M, N), dtype=torch.float).cuda()

    # Warmup
    warmup(lib.gemm_cublas, a, b, c)
    warmup(lib.gemm_t_8x8_sliced_k_f32x4, a, b, c)
    warmup(lib.gemm_t_8x8_sliced_k_f32x4_bcf_offset, a, b, c)
    warmup(lib.gemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset, a, b, c)

    # gemm_wmma_m16n16k8_mma4x2_warp2x4_stages
    # args: a, b, c, stages, swizzle, swizzle_stride
    for _ in range(2):
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages(a, b, c, 2, False, 1)
    torch.cuda.synchronize()

    # sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem
    # args: a, b, c, stages, swizzle, swizzle_stride
    for _ in range(2):
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(a, b, c, 2, False, 1)
    torch.cuda.synchronize()

    # for _ in range(2):
    #     lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages(a, b, c, 3, False, 1)
    # torch.cuda.synchronize()

    # # sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem
    # # args: a, b, c, stages, swizzle, swizzle_stride
    # for _ in range(2):
    #     lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(a, b, c, 3, False, 1)
    # torch.cuda.synchronize()

    # for _ in range(2):
    #     lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages(a, b, c, 4, False, 1)
    # torch.cuda.synchronize()

    # # sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem
    # # args: a, b, c, stages, swizzle, swizzle_stride
    # for _ in range(2):
    #     lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(a, b, c, 4, False, 1)
    # torch.cuda.synchronize()


    # Profile
    lib.gemm_cublas(a, b, c)
    lib.gemm_t_8x8_sliced_k_f32x4(a, b, c)
    lib.gemm_t_8x8_sliced_k_f32x4_bcf_offset(a, b, c)
    lib.gemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset(a, b, c)
    lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages(a, b, c, 2, False, 1)
    lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(a, b, c, 2, False, 1)
    lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages(a, b, c, 3, False, 1)
    # lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(a, b, c, 3, False, 1)
    # lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages(a, b, c, 4, False, 1)
    # lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(a, b, c, 4, False, 1)
