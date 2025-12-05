import torch
from torch.utils.cpp_extension import load
import numpy as np
import os

# Set the working directory to the script's directory to ensure relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the CUDA kernel as a python module
lib = load(
    name="reduce_lib_test",
    sources=["reduce.cu"],
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

def test_reduce_correctness():
    print("Running Reduce Correctness Tests...")
    torch.manual_seed(0)
    
    # Test cases: (S, K)
    # S: number of rows (or outer dimension size)
    # K: number of columns (or inner dimension size)
    # Total elements N = S * K
    test_cases = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    kernels = [
        ("f32_f32 (Atomic)", lib.block_all_reduce_sum_f32_f32),
    ]

    print(f"{'Kernel':<25} {'Size (S, K)':<20} {'Status':<10} {'Max Diff':<15}")
    print("-" * 75)

    all_passed = True

    for S, K in test_cases:
        # Use float32 for testing
        x = torch.randn((S, K), device='cuda', dtype=torch.float32)
        
        # PyTorch reference result
        expected = torch.sum(x)
        
        for name, func in kernels:
            try:
                # Create a copy to ensure input isn't modified
                x_in = x.clone()
                
                # Run kernel
                output = func(x_in)
                
                # Check result
                # Reduce sum can have precision issues due to order of operations
                # We use a relatively loose tolerance for large reductions
                rtol = 1e-3
                atol = 1e-3
                
                # For very large sums, the absolute error might grow, so rely more on relative error
                # unless the sum is close to zero.
                
                if not torch.allclose(output, expected, rtol=rtol, atol=atol):
                    diff = torch.abs(output - expected).item()
                    print(f"{name:<25} {str((S, K)):<20} {'FAIL':<10} {diff:<15.6f}")
                    print(f"    Expected: {expected.item():.6f}, Got: {output.item():.6f}")
                    all_passed = False
                else:
                    # print(f"{name:<25} {str((S, K)):<20} {'PASS':<10} {'0.0':<15}")
                    pass # Reduce output noise, only print failures or summary
            except Exception as e:
                print(f"{name:<25} {str((S, K)):<20} {'ERROR':<10} {str(e)}")
                all_passed = False
    
    print("-" * 75)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED.")

if __name__ == "__main__":
    test_reduce_correctness()
