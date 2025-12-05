#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <algorithm>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// GEMM naive: compute one c[i,j]
// element per threads, all row major
__global__ void gemm_fp32_kernel(const float *a, const float *b, float *c, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M && n < N) {
    float psum = 0.0;
#pragma unroll
    for (int k = 0; k < K; k++) {
      // m row in a matrix, n col in b matrix
      psum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = psum;  // c[m,n]
  }
}

// GEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void gemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
  // [1] Block Tile: 32x32 的 block 处理 c 上一块32x32的元素计算
  // [2]     K Tile: 使用共享内存，并将 K 分块为 BK 大小的块
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;  // tid within the block

  // 保证相邻的线程 读取 相邻的内存地址
  int load_smem_a_m = tid / 32;                 // 0~31, tid / 32, tid / BM, threadIdx.y
  int load_smem_a_k = tid % 32;                 // 0~31, tid % 32, tid % BK, threadIdx.x
  int load_smem_b_k = tid / 32;                 // 0~31, tid / 32, tid / BK, threadIdx.y
  int load_smem_b_n = tid % 32;                 // 0~31, tid % 32, tid % BN, threadIdx.x
  int load_gmem_a_m = by * BM + load_smem_a_m;  // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n;  // global col of b and c

  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 计算每次加载到smem的A和B矩阵元素在全局内存中的列数和行数
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}

// gemm: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8,
          const int TN = 8>
__global__ void gemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c, int M, int N,
                                                 int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;  // tid within the block
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  // 0. 先计算shared memory中的索引
  // tid和需要加载的 smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2;                 // tid/2 (128/8)*(128/8)=256 threads per block,
                                               // tid/2->[0,128), BM=128 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;  // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32;        // tid/32, row of s_b 256/32=8 行 0~7
  int load_smem_b_n = (tid % 32) * 4;  // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  int load_gmem_a_m = by * BM + load_smem_a_m;  // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n;  // global col of b and c

  float r_c[TM][TN];
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 2. 计算每次加载到smem的A和B矩阵元素在全局内存中的列数和行数
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) =
        FLOAT4(a[load_gmem_a_addr]);  // load A to shared memory

    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) =
        FLOAT4(b[load_gmem_b_addr]);  // load B to shared memory
    __syncthreads();

    // 3. 计算线程负责的TMxTN个元素的部分乘加
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          int comp_smem_a_m = ty * TM + m;
          int comp_smem_b_n = tx * TN + n;
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

// 4. store output
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m =
        by * BM + ty * TM +
        m;  // 大分块的起始行号by * BM + 小分块的起始行号ty * TM + 小分块内部的相对行号 m
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      int store_gmem_c_n =
          bx * BN + tx * TN +
          n;  // 大分块的起始列号bx * BN + 小分块的起始列号tx * TN + 小分块内部的相对列号 n
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      c[store_gmem_c_addr] = r_c[m][n];
    }
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8,
          const int TN = 8, const int OFFSET = 0>
__global__ void gemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *a, float *b, float *c, const int M,
                                                     const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[BK][BM + OFFSET];  //我们需要的是 A[0][0], A[1][0], A[2][0],
                                          // A[3][0]（同一列的 4 个 M 值）。在 s_a[BM][BK] 中，这 4
                                          //个数的内存地址不连续（相隔 BK 个 float）。
  __shared__ float s_b[BK][BN + OFFSET];

  float r_load_a[TM / 2];  // 4
  float r_load_b[TN / 2];  // 4
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  int load_a_smem_m = tid / 2;  // tid / 2，(0,1,2,...,128)
  int load_a_smem_k = (tid & 1) << 2;
  int load_b_smem_k = tid / 32;         // 0~8
  int load_b_smem_n = (tid & 31) << 2;  // (0,4,8,12,...,124)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];      // e.g layer_0  b0
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];  // e.g layer_4  b0
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];  // e.g layer_8  b0
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];  // e.g layer_12 b0

    FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);

      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
      // conclusion: still have some bank conflicts, need 4 memory issues.

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    // sync per BK.
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8,
          const int TN = 8, const int OFFSET = 0>
__global__ void gemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(float *a, float *b, float *c, const int M,
                                                          const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];  // double buffer
  __shared__ float s_b[2][BK][BN + OFFSET];

  float r_load_a[TM / 2];
  float r_load_b[TN / 2];
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};
  // bm * bk (128 * 8)
  int load_a_smem_m = tid / 2;
  int load_a_smem_k = (tid & 1) << 2;
  int load_a_gmem_m = by * BM + load_a_smem_m;
  // bk * bn (8 * 128)
  int load_b_smem_k = tid / 32;
  int load_b_smem_n = (tid & 31) << 2;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  // before entering the bk loop, load the first block into buffer 0
  {
    int load_a_gmem_k = 0 * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = 0 * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    // from global memory to register
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);  // from register to shared memory
    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
  }
  // sync before entering the bk loop
  __syncthreads();
  // start the bk loop, bk = 1 we just use the data loaded before the loop
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int curr_buf = (bk - 1) & 1;
    int next_buf = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    // compute with curr_buf
#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // load from shared memory to register
      // to reduce bank conflict, we load 128 bit with TM / 2 or TN / 2 strides
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[curr_buf][tk][ty * TM / 2]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[curr_buf][tk][ty * TM / 2 + BM / 2]);

      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[curr_buf][tk][tx * TN / 2]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[curr_buf][tk][tx * TN / 2 + BN / 2]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    // load next block into next_buf
    s_a[next_buf][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[next_buf][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[next_buf][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[next_buf][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[next_buf][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();
  }
// now we must compute the last buffer
#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

  // store the result at four corner
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type))) {                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)             \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor size mismatch!");  \
  }

// gemm naive: compute one c[i,j] element per threads, all row major
void gemm_naive_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_fp32_kernel<<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                                    reinterpret_cast<float *>(b.data_ptr()),
                                    reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

// gemm: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major
void gemm_sliced_k_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_sliced_k_f32_kernel<BM, BN, BK><<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

// gemm: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
void gemm_t_8x8_sliced_k_f32x4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void gemm_t_8x8_sliced_k_f32x4_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void gemm_t_8x8_sliced_k_f32x4_bcf_offset(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int OFFSET = 4;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN, OFFSET><<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void gemm_t_8x8_sliced_k_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void gemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int OFFSET = 4;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN, OFFSET><<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}
// from gemm_cublas.cu
void gemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void gemm_cublas_tf32(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// from gemm_mma.cu
void gemm_wmma_m16n16k8_mma4x2_warp2x4_stages(torch::Tensor a, torch::Tensor b,
                                               torch::Tensor c, int stages,
                                               bool swizzle,
                                               int swizzle_stride);
void gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(torch::Tensor a,
                                                     torch::Tensor b,
                                                     torch::Tensor c,
                                                     int stages, bool swizzle,
                                                     int swizzle_stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CUDA Cores
  TORCH_BINDING_COMMON_EXTENSION(gemm_naive_f32)
  TORCH_BINDING_COMMON_EXTENSION(gemm_sliced_k_f32)
  TORCH_BINDING_COMMON_EXTENSION(gemm_t_8x8_sliced_k_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(gemm_t_8x8_sliced_k_f32x4_bcf)
  TORCH_BINDING_COMMON_EXTENSION(gemm_t_8x8_sliced_k_f32x4_bcf_offset)
  TORCH_BINDING_COMMON_EXTENSION(gemm_t_8x8_sliced_k_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(gemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset)
  // cuBLAS Tensor Cores
  TORCH_BINDING_COMMON_EXTENSION(gemm_cublas)
  TORCH_BINDING_COMMON_EXTENSION(gemm_cublas_tf32)
  // WMMA Tensor Cores
  TORCH_BINDING_COMMON_EXTENSION(gemm_wmma_m16n16k8_mma4x2_warp2x4_stages)
  TORCH_BINDING_COMMON_EXTENSION(gemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem)
}