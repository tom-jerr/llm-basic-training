import torch
import gemm_custom
import time
import matplotlib.pyplot as plt

def benchmark(m, n, k, num_repeats=100, num_warmup=10):
    a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    b = torch.randn(n, k, device='cuda', dtype=torch.float16)
    c = torch.empty(m, n, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(num_warmup):
        gemm_custom.gemm_custom(c, a, b)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_repeats):
        gemm_custom.gemm_custom(c, a, b)
    end.record()
    torch.cuda.synchronize()
    custom_time = start.elapsed_time(end) / num_repeats

    # PyTorch (cuBLAS)
    # PyTorch matmul: A @ B.T
    # Warmup
    for _ in range(num_warmup):
        torch.matmul(a, b.T)
    
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_repeats):
        torch.matmul(a, b.T)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_repeats
    
    return custom_time, torch_time

sizes = [1024 * i for i in range(1, 16, 4)]  # From 1K to 16K
custom_times = []
torch_times = []

print("Running benchmark...")
for s in sizes:
    m = n = k = s
    ct, tt = benchmark(m, n, k)
    custom_times.append(ct)
    torch_times.append(tt)
    print(f"Size {s}: Custom={ct:.3f}ms, PyTorch={tt:.3f}ms")

plt.figure()
plt.plot(sizes, custom_times, label='Custom GEMM')
plt.plot(sizes, torch_times, label='PyTorch (cuBLAS)')
plt.xlabel('Matrix Size (N=M=K)')
plt.ylabel('Time (ms)')
plt.title('GEMM Benchmark')
plt.legend()
plt.savefig('benchmark_result.png')
print("Benchmark plot saved to benchmark_result.png")
