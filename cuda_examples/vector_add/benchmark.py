import time
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="vector_add_lib",
    sources=["vector_add_kernel.cu"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

def run_benchmark(func, a, b, c, tag, warmup=10, iters=100):
    # Warmup
    for _ in range(warmup):
        func(a, b, c)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iters):
        func(a, b, c)
    torch.cuda.synchronize()
    end = time.time()
    
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    
    # Calculate Bandwidth (GB/s)
    # Read a, Read b, Write c. Total 3 * N * sizeof(type)
    element_size = a.element_size()
    num_elements = a.numel()
    total_bytes = 3 * num_elements * element_size
    bandwidth = total_bytes / (mean_time / 1000) / 1e9 # GB/s
    
    return {
        "tag": tag,
        "mean_time": mean_time,
        "bandwidth": bandwidth
    }

N = 1024 * 1024 * 128 # 128M elements
print(f"Vector size: {N}")

results = []

# FP32
a_fp32 = torch.randn(N, device='cuda', dtype=torch.float32)
b_fp32 = torch.randn(N, device='cuda', dtype=torch.float32)
c_fp32 = torch.empty_like(a_fp32)

results.append(run_benchmark(lambda a,b,c: torch.add(a, b, out=c), a_fp32, b_fp32, c_fp32, "torch.add (fp32)"))
results.append(run_benchmark(lib.vector_add, a_fp32, b_fp32, c_fp32, "vector_add (fp32)"))
results.append(run_benchmark(lib.vector_add_vec4, a_fp32, b_fp32, c_fp32, "vector_add_vec4 (fp32)"))

# FP16
a_fp16 = torch.randn(N, device='cuda', dtype=torch.float16)
b_fp16 = torch.randn(N, device='cuda', dtype=torch.float16)
c_fp16 = torch.empty_like(a_fp16)

results.append(run_benchmark(lambda a,b,c: torch.add(a, b, out=c), a_fp16, b_fp16, c_fp16, "torch.add (fp16)"))
results.append(run_benchmark(lib.vector_add_fp16, a_fp16, b_fp16, c_fp16, "vector_add_fp16"))
results.append(run_benchmark(lib.vector_add_fp16x8, a_fp16, b_fp16, c_fp16, "vector_add_fp16x8"))

print("-" * 75)
print(f"{'Name':<25} {'Time(ms)':<10} {'BW(GB/s)':<10} {'Speedup':<10}")

baseline_time = results[0]['mean_time']

for r in results:
    speedup = baseline_time / r['mean_time']
    print(f"{r['tag']:<25} {r['mean_time']:<10.4f} {r['bandwidth']:<10.2f} {speedup:<10.2f}")
print("-" * 75)
