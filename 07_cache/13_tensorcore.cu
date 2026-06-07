#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;
/*

実行
nvcc 13_tensorcore.cu -Xcompiler "-O3 -fopenmp" -lcublas

元のプログラムでやっていること

- 行列積C = A * Bを計算する

1. cuBLASを使って行列積を計算する(ベースライン)
2. 自作CUDAカーネルを使って行列積を計算する
3. 計算時間と性能を比較する

# main関数
- 行列のサイズを定義(m=10240, k=4096, n=8192)
- 行列A, B, C, C2をCUDAの統一メモリ
- メモリ確保と初期化
- cuBLASハンドルの作成
- cuBLASを使って行列積を計算し、時間を測定
- 自作CUDAカーネルを使って行列積を計算し、時間を測定
- 計算性能を表示
- 結果の誤差を計算して表示

# kernel関数
- 行列積を計算するCUDAカーネル
- 各ブロックは64x64のタイルを計算する
- 各スレッドは16x16のサブタイルを計算する
- wmma APIを使ってTensor Coreを利用して行列積を計算する
- ブロックごとに共有メモリにタイルをロードし、スレッドごとにフラグメントをロードして計算する
- 最後に結果をグローバルメモリにストアする
*/


/*方針
CUBLAS: 186508.66 Gflops, CUTLASS: 10218.12 Gflops
error: 0.003980
1. 64×64のタイル/ 2warpから128×128のタイル / 8 warpへ
CUBLAS: 186970.20 Gflops, CUTLASS: 9962.62 Gflops
error: 0.003980
2. shared memoryのサイズを増やす(k=16から32へ)(ここからGPUを確保した)
CUBLAS: 355551.90 Gflops, CUTLASS: 22191.98 Gflops
error: 0.003980
3. shared memoryのサイズを増やす(k=32から64へ)
CUBLAS: 355029.43 Gflops, CUTLASS: 22205.85 Gflops
error: 0.003980
4. shared memory paddingを入れる
CUBLAS: 355868.39 Gflops, CUTLASS: 26961.20 Gflops
error: 0.003980
5. shared memory からの B fragment load の削減
CUBLAS: 354092.24 Gflops, CUTLASS: 26831.83 Gflops
error: 0.003980
6. acc[2][4]からacc[2][2]へ
CUBLAS: 353918.69 Gflops, CUTLASS: 29574.17 Gflops
error: 0.003980
*/



__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       float *d_a, float *d_b, float *d_c) {

  const int BLOCK_M = 128;
  const int BLOCK_N = 64;
  const int BLOCK_K = 64;
  
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  int offset_a_m = BLOCK_M * blockIdx.x;
  int offset_b_n = BLOCK_N * blockIdx.y;

  int tid = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  int warp_m_base = (warp_id % 4) * 32;
  int warp_n_base = (warp_id / 4) * 32;

  __shared__ half block_a[BLOCK_K][BLOCK_M + 8];
  __shared__ half block_b[BLOCK_K][BLOCK_N + 8];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];

  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 2; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k = 0; k < dim_k; k += BLOCK_K) {
    __syncthreads();
    for (int idx = tid; idx < BLOCK_K * BLOCK_M; idx += blockDim.x) {
      int j = idx / BLOCK_M;
      int m = idx % BLOCK_M;

      int global_m = offset_a_m + m;
      int global_k = k + j;
      if(global_m < dim_m && global_k < dim_k)
        block_a[j][m] = __float2half(d_a[global_k * dim_m + global_m]);
      else
        block_a[j][m] = __float2half(0.0f);
    }
    for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += blockDim.x) {
      int j = idx / BLOCK_N;
      int n = idx % BLOCK_N;

      int global_n = offset_b_n + n;
      int global_k = k + j;

      if (global_n < dim_n && global_k < dim_k)
        block_b[j][n] = __float2half(d_b[global_n * dim_k + global_k]);
      else
        block_b[j][n] = __float2half(0.0f);
}


    __syncthreads();
    for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
      for (int c = 0; c < 2; c++) {
        int n_tile = warp_n_base + c * WMMA_N;
        wmma::load_matrix_sync(b_frag[c], &block_b[kk][n_tile], BLOCK_N+8);
      }
      for (int r = 0; r < 2; r++) {
        int m_tile = warp_m_base +r * WMMA_M;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
        wmma::load_matrix_sync(a_frag, &block_a[kk][m_tile], BLOCK_M+8);

        for (int c = 0; c < 2; c++) {
          wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
        }
    }
  }
  }

  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 2; c++) {
      int c_m = offset_a_m + warp_m_base + r * WMMA_M;
      int c_n = offset_b_n + warp_n_base + c * WMMA_N;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 A, CUDA_R_32F, m,
		 B, CUDA_R_32F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F_FAST_16F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  int tile_m = 128;
  int tile_n = 64;
  dim3 block = dim3(256);
  dim3 grid = dim3((m+tile_m-1)/tile_m, (n+tile_n-1)/tile_n);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m,
			      n,
			      k,
			      A,
			      B,
			      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
