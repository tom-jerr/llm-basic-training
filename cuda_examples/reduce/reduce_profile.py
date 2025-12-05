import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="reduce_lib",
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
