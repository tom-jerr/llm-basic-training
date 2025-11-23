import torch
from torch.utils.cpp_extension import load
import time

# Load the CUDA kernel as a python module
# Reusing the same compilation flags as in test_benchmark.py
lib = load(
    name="vector_add_lib",
    sources=["vector_add_ops.cu"],
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

def check_implementation(func, a, b, c_ref, name, warmup=3):
    """
    Checks the correctness of a CUDA kernel implementation against a reference.
    Includes a warmup phase.
    """
    # Warm up
    # We use a temporary buffer for warmup to avoid side effects (though for add it doesn't matter much)
    warmup_out = torch.empty_like(c_ref)
    for _ in range(warmup):
        func(a, b, warmup_out)
    torch.cuda.synchronize()
    
    # Run actual test
    c_out = torch.zeros_like(c_ref)
    func(a, b, c_out)
    torch.cuda.synchronize()
    
    # Check correctness
    # Using slightly loose tolerance for FP16
    atol = 1e-3 if a.dtype == torch.float16 else 1e-5
    rtol = 1e-3 if a.dtype == torch.float16 else 1e-5
    
    if torch.allclose(c_out, c_ref, atol=atol, rtol=rtol):
        print(f"✅ {name} passed!")
    else:
        max_diff = (c_out - c_ref).abs().max().item()
        print(f"❌ {name} failed! Max diff: {max_diff}")

def main():
    S = 1024 * 10
    K = 1024 * 10
    print(f"Testing with shape ({S}, {K})...")
    
    # FP32 Test
    print("\n--- FP32 Tests ---")
    a_f32 = torch.randn((S, K)).cuda().float()
    b_f32 = torch.randn((S, K)).cuda().float()
    # warm up
    # torch.add(a_f32, b_f32)  
    c_ref_f32 = torch.add(a_f32, b_f32)
    
    # check_implementation(lib.elementwise_add_f32, a_f32, b_f32, c_ref_f32, "f32")
    check_implementation(lib.elementwise_add_f32x4, a_f32, b_f32, c_ref_f32, "f32x4")

    # FP16 Test
    print("\n--- FP16 Tests ---")
    a_f16 = a_f32.half()
    b_f16 = b_f32.half()
    # torch.add(a_f16, b_f16)  
    c_ref_f16 = torch.add(a_f16, b_f16)
    
    # check_implementation(lib.elementwise_add_f16, a_f16, b_f16, c_ref_f16, "f16")
    check_implementation(lib.elementwise_add_f16x8, a_f16, b_f16, c_ref_f16, "f16x8")
    # check_implementation(lib.elementwise_add_f16x8_pack, a_f16, b_f16, c_ref_f16, "f16x8_pack")

if __name__ == "__main__":
    main()
