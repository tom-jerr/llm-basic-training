import time
from functools import partial
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch.utils.cpp_extension import load

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

MAX_TFLOPS = -1


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    stages: int = -1,
    swizzle: bool = False,
    swizzle_stride: int = 1,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):

    global MAX_TFLOPS

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)

    if a.size(0) > 1024 or a.size(1) >= 1024 or b.size(1) > 1024:
        iters = 10

    if swizzle:
        # make swizzle stride as N/4 and multiples of 256
        swizzle_stride = int((int(N / 8) // 256) * 256)
        swizzle_stride = swizzle_stride if swizzle_stride >= 256 else 1
        swizzle = swizzle if swizzle_stride >= 256 else False
    else:
        swizzle_stride = 1  # means no thread block swizzle

    if stages:
        assert swizzle_stride is not None

    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten()[:2].detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}"[:10] for v in out_val]
    TFLOPS = (2 * M * N * K) * 1e-9 / (mean_time)
    swizzle_stride = "NOOP" if swizzle_stride == 1 else swizzle_stride

    return {
        "tag": tag,
        "out_val": out_val,
        "mean_time": mean_time,
        "swizzle_stride": swizzle_stride,
        "tflops": TFLOPS
    }


def plot_results(all_results):
    # all_results: dict of (M, N, K) -> list of result dicts

    # Prepare data for plotting
    # We will plot TFLOPS vs Problem Size (M=N=K usually, or just index)
    
    labels = []
    for k in all_results.keys():
        if k[0] == k[1] == k[2]:
            labels.append(str(k[0]))
        else:
            labels.append(f"{k[0]}x{k[1]}x{k[2]}")
    # Collect all unique tags
    tags = set()
    for res_list in all_results.values():
        for r in res_list:
            tags.add(r['tag'])
    tags = sorted(list(tags))
    
    x = np.arange(len(labels))  # the label locations
    
    # Increase figure size to accommodate bottom legend
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Using a distinct colormap
    import matplotlib
    try:
        cmap = matplotlib.colormaps['tab10']
    except AttributeError:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(tags)))
    for i, tag in enumerate(tags):
        tflops_vals = []
        for k in all_results.keys():
            # Find result for this tag in this config
            val = 0
            for r in all_results[k]:
                if r['tag'] == tag:
                    val = r['tflops']
                    break
            tflops_vals.append(val)
            
        ax.plot(x, tflops_vals, marker='o', linewidth=2, markersize=6,label=tag, color=colors[i])

    ax.set_ylabel('TFLOPS', fontsize=12)
    ax.set_xlabel('Matrix Size (M=N=K)', fontsize=12)
    ax.set_title('GEMM Performance Benchmark', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)
    # Legend at the bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3, fontsize=12)
    
    fig.tight_layout()
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gemm_benchmark_line.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


sizes = [256 * (2**i) for i in range(7)] # 256, 512, 1024, 2048, 4096, 8192, 16384
MAX_M, MAX_N, MAX_K = 16384, 16384, 16384
# pre allocate for fast profiling.
A = torch.randn((MAX_M, MAX_K), dtype=torch.float).cuda()
B = torch.randn((MAX_K, MAX_N), dtype=torch.float).cuda()
C = torch.randn((MAX_M, MAX_N), dtype=torch.float).cuda()
torch.cuda.synchronize()

all_results = {}

MNKs = [(s, s, s) for s in sizes]
for M, N, K in MNKs:
    MAX_TFLOPS = -1
    print("-" * 130)
    print(" " * 55 + f"M={M}, N={N}, K={K}")
    a = A[:M, :K].contiguous()
    b = B[:K, :N].contiguous()
    c = C[:M, :N].contiguous()
    torch.cuda.synchronize()

    results = []

    # CUDA Cores FP32
    # results.append(run_benchmark(lib.gemm_naive_f32, a, b, "f32(naive)", c))
    results.append(run_benchmark(lib.gemm_t_8x8_sliced_k_f32x4, a, b, "f32x4(t8x8sk)", c))
    # results.append(run_benchmark(lib.gemm_t_8x8_sliced_k_f32x4_bcf, a, b, "f32x4(t8x8bcf)", c))
    results.append(run_benchmark(lib.gemm_t_8x8_sliced_k_f32x4_bcf_offset, a, b, "f32x4(t8x8bcf_off)", c))
    results.append(run_benchmark(lib.gemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset, a, b, "f32x4(t8x8sk dbuf_offset)", c))
    results.append(run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th"))

    print("-" * 62 + "WMMA" + "-" * 64)
    # stage, thread block swizzle, dsmem
    results.append(run_benchmark(
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages,
        a,
        b,
        "tf32(mma2x4+warp2x4+stage3)",
        c,
        stages=3,
    ))
    results.append(run_benchmark(
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages,
        a,
        b,
        "tf32(mma2x4+warp2x4+stage2)",
        c,
        stages=2,
    ))

    results.append(run_benchmark(
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem,
        a,
        b,
        "tf32(mma2x4+...+stage3+dsmem)",
        c,
        stages=3,
    ))
    results.append(run_benchmark(
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem,
        a,
        b,
        "tf32(mma2x4+...+stage2+dsmem)",
        c,
        stages=2,
    ))

    results.append(run_benchmark(
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages,
        a,
        b,
        "tf32(mma2x4+...+stage3+swizzle)",
        c,
        stages=3,
        swizzle=True,
    ))
    results.append(run_benchmark(
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages,
        a,
        b,
        "tf32(mma2x4+...+stage2+swizzle)",
        c,
        stages=2,
        swizzle=True,
    ))

    results.append(run_benchmark(
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem,
        a,
        b,
        "tf32(...+stage3+dsmem+swizzle)",
        c,
        stages=3,
        swizzle=True,
    ))
    results.append(run_benchmark(
        lib.gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem,
        a,
        b,
        "tf32(...+stage2+dsmem+swizzle)",
        c,
        stages=2,
        swizzle=True,
    ))

    # results.append(run_benchmark(lib.gemm_cublas_tf32, a, b, "tf32(cublas+tf32)", c))
    torch.cuda.synchronize()

    # Print Table
    baseline_tflops = 0
    # Try to find f32_th as baseline
    for r in results:
        if "th" in r["tag"]:
            baseline_tflops = r["tflops"]
            break
    if baseline_tflops == 0:
        baseline_tflops = results[0]["tflops"]

    print(f"{'Name':<35} {'Time(ms)':<10} {'TFLOPS':<10} {'Speedup':<10} {'Swizzle':<10}")
    for r in results:
        speedup = r["tflops"] / baseline_tflops if baseline_tflops > 0 else 0
        print(f"{r['tag']:<35} {r['mean_time']:<10.4f} {r['tflops']:<10.2f} {speedup:<10.2f} {r['swizzle_stride']}")

    print("-" * 130)

    all_results[(M, N, K)] = results

# Calculate and print average speedup
print("\n" + "=" * 60)
print(f"{'Kernel Name':<35} {'Average Speedup':<15}")
print("-" * 60)

kernel_speedups = {}

for config, results in all_results.items():
    baseline_tflops = 0
    # Find baseline for this config
    for r in results:
        if "th" in r["tag"]:
            baseline_tflops = r["tflops"]
            break
    if baseline_tflops == 0 and len(results) > 0:
         baseline_tflops = results[0]["tflops"]
    
    if baseline_tflops > 0:
        for r in results:
            tag = r["tag"]
            speedup = r["tflops"] / baseline_tflops
            if tag not in kernel_speedups:
                kernel_speedups[tag] = []
            kernel_speedups[tag].append(speedup)

for tag in sorted(kernel_speedups.keys()):
    speedups = kernel_speedups[tag]
    avg_speedup = sum(speedups) / len(speedups)
    print(f"{tag:<35} {avg_speedup:<15.4f}")
print("=" * 60 + "\n")

plot_results(all_results)