import torch
from torch.utils.cpp_extension import load
import time

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="vector_add_lib_test",
    sources=["vector_add_kernel.cu"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

def verify_correctness(func, a, b, c_ref, name):
    c = torch.zeros_like(c_ref)
    try:
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
    N = 1024 * 1024 * 128 # 128M elements
    print(f"Testing with Vector size: {N}")
    
    # FP32
    a_fp32 = torch.randn(N, device='cuda', dtype=torch.float32)
    b_fp32 = torch.randn(N, device='cuda', dtype=torch.float32)
    c_ref_fp32 = a_fp32 + b_fp32
    c_fp32 = torch.zeros_like(c_ref_fp32)

    # FP16
    a_fp16 = torch.randn(N, device='cuda', dtype=torch.float16)
    b_fp16 = torch.randn(N, device='cuda', dtype=torch.float16)
    c_ref_fp16 = a_fp16 + b_fp16
    c_fp16 = torch.zeros_like(c_ref_fp16)

    # Verify Correctness
    print("-" * 20 + " Correctness Check " + "-" * 20)
    verify_correctness(lib.vector_add, a_fp32, b_fp32, c_ref_fp32, "vector_add (fp32)")
    verify_correctness(lib.vector_add_vec4, a_fp32, b_fp32, c_ref_fp32, "vector_add_vec4 (fp32)")
    verify_correctness(lib.vector_add_fp16, a_fp16, b_fp16, c_ref_fp16, "vector_add_fp16")
    verify_correctness(lib.vector_add_fp16x8, a_fp16, b_fp16, c_ref_fp16, "vector_add_fp16x8")


