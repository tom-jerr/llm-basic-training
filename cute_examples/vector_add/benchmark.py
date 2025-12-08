import torch
import vector_add_cuda
import time
import matplotlib.pyplot as plt

def vector_add_torch(x, y, a, b, c):
    return x * a + y * b + c

def benchmark(func, args, n_warmup=10, n_iters=100):
    # Warmup
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / n_iters

def main():
    sizes = [1024 * 1024 * i for i in range(1, 33, 4)] # 1MB to 32MB elements
    cuda_times = []
    torch_times = []
    
    device = torch.device("cuda")
    a = 1.0
    b = 1.0
    c = 0.0

    print("Running benchmark...")
    
    for num in sizes:
        print(f"Benchmarking size: {num}")
        x = torch.arange(num, dtype=torch.float16, device=device)
        y = torch.arange(num, dtype=torch.float16, device=device)
        z = torch.empty_like(x)
        
        # Correctness check (only once or for first size)
        if num == sizes[0]:
            vector_add_cuda.vector_add(z, x, y, a, b, c)
            ref = vector_add_torch(x, y, a, b, c)
            if not torch.allclose(z, ref, atol=1e-3, rtol=1e-3):
                print(f"Verification failed for size {num}!")
            else:
                print("Verification passed.")

        # Benchmark Custom CUDA
        t_cuda = benchmark(lambda: vector_add_cuda.vector_add(z, x, y, a, b, c), ())
        cuda_times.append(t_cuda * 1000) # ms
        
        # Benchmark PyTorch
        # Note: PyTorch implementation allocates memory for result, while CUDA one writes to existing z.
        # To be fair, we might want to include allocation time or pre-allocate for torch too if possible, 
        # but vector_add_torch returns a new tensor. 
        # For strict comparison of kernel speed, we should use torch.addcmul or similar in-place if possible,
        # but the current vector_add_torch is simple. We'll stick to it.
        t_torch = benchmark(lambda: vector_add_torch(x, y, a, b, c), ())
        torch_times.append(t_torch * 1000) # ms

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cuda_times, label='Custom CUDA', marker='o')
    plt.plot(sizes, torch_times, label='PyTorch', marker='x')
    plt.xlabel('Number of Elements')
    plt.ylabel('Time (ms)')
    plt.title('Vector Add Benchmark: Custom CUDA vs PyTorch')
    plt.legend()
    plt.grid(True)
    plt.savefig('benchmark_result.png')
    print("Benchmark finished. Result saved to benchmark_result.png")

if __name__ == "__main__":
    main()
