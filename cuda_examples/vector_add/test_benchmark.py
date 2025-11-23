import time
from functools import partial
from typing import Optional
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
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


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 10,
    show_all: bool = False,
):
    # torch.dot vs custom dot_prod kernel
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


Ss = [256, 512, 1024, 2048, 4096]
Ks = [256, 512, 1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

results = defaultdict(list)

for S, K in SKs:
    N = S * K
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    a = torch.randn((S, K)).cuda().float().contiguous()
    b = torch.randn((S, K)).cuda().float().contiguous()
    c = torch.zeros_like(a).cuda().float().contiguous()
    
    _, t = run_benchmark(lib.elementwise_add_f32x4, a, b, "f32x4", c)
    results["f32x4"].append((N, t))
    
    _, t = run_benchmark(partial(torch.add, out=c), a, b, "f32_th")
    results["f32_th"].append((N, t))

    print("-" * 85)
    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()
    c_f16 = c.half().contiguous()
    
    _, t = run_benchmark(lib.elementwise_add_f16x8, a_f16, b_f16, "f16x8", c_f16)
    results["f16x8"].append((N, t))
    
    _, t = run_benchmark(partial(torch.add, out=c_f16), a_f16, b_f16, "f16_th")
    results["f16_th"].append((N, t))
    print("-" * 85)

# Plotting
plt.figure(figsize=(12, 8))
for label, data in results.items():
    data.sort() # Ensure sorted by N
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.plot(x, y, marker='o', label=label)

plt.xlabel("Total Elements (N)")
plt.ylabel("Time (ms)")
plt.title("Vector Add Benchmark Performance")
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.savefig("benchmark_result.png")
print("Benchmark result saved to benchmark_result.png")