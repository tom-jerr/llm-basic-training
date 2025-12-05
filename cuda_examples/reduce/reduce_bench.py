import time
import matplotlib.pyplot as plt
import numpy as np
import os

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


def run_benchmark(
    perf_func: callable,
    values: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
):
    for i in range(warmup):
        out = perf_func(values)  # warmup
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        out = perf_func(values)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    
    num_bytes = values.numel() * values.element_size()
    bandwidth = num_bytes / (mean_time / 1000) / 1e9 # GB/s
    
    return {
        "tag": tag,
        "mean_time": mean_time,
        "bandwidth": bandwidth,
        "out_val": out.item()
    }


def plot_results(all_results):
    labels = [f"{k[0]}x{k[1]}" for k in all_results.keys()]
    tags = sorted(list(set(r['tag'] for res in all_results.values() for r in res)))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    
    import matplotlib
    try:
        cmap = matplotlib.colormaps['tab10']
    except AttributeError:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(tags)))

    for i, tag in enumerate(tags):
        bw_vals = []
        for k in all_results.keys():
            val = 0
            for r in all_results[k]:
                if r['tag'] == tag:
                    val = r['bandwidth']
                    break
            bw_vals.append(val)
        ax.plot(x, bw_vals, marker='o', label=tag, color=colors[i])

    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Reduce Performance Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reduce_benchmark.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


Ss = [1024, 2048, 4096, 8192]
Ks = [1024, 2048, 4096, 8192]
SKs = [(S, K) for S in Ss for K in Ks]

all_results = {}

for S, K in SKs:
    print("-" * 100)
    print(f"S={S}, K={K}")
    values = torch.randn((S, K)).cuda().float()
    
    results = []
    results.append(run_benchmark(lib.block_all_reduce_sum_f32_f32, values, "f32f32"))
    results.append(run_benchmark(lib.block_all_reduce_sum_f32x4_f32, values, "f32x4f32"))
    results.append(run_benchmark(torch.sum, values, "f32f32_th"))
    
    # Find baseline (torch.sum)
    baseline_time = 0
    for r in results:
        if "th" in r["tag"]:
            baseline_time = r["mean_time"]
            break
    if baseline_time == 0:
        baseline_time = results[-1]["mean_time"]

    print(f"{'Name':<20} {'Time(ms)':<15} {'Bandwidth(GB/s)':<20} {'Speedup':<10} {'Value':<15}")
    for r in results:
        speedup = baseline_time / r["mean_time"] if r["mean_time"] > 0 else 0
        r["speedup"] = speedup
        print(f"{r['tag']:<20} {r['mean_time']:<15.4f} {r['bandwidth']:<20.2f} {speedup:<10.2f} {r['out_val']:<15.4f}")
        
    all_results[(S, K)] = results

plot_results(all_results)

    