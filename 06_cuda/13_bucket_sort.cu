#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>


// CUDAで実行されるGPU側の並列関数kernel
// keyの値を見てbucketの対応する位置をインクリメントしていく
__global__
void histogram_kernel(
    int* key,
    int* bucket,
    int n
){
    int idx =
      blockIdx.x * blockDim.x
      + threadIdx.x;

    if (idx < n) {
        atomicAdd(
          &bucket[key[idx]],
          1
        );
    }
}



int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");


  // GPUで並列化するため，bucketの更新で競合が起こらないように頑張る
  std::vector<int> bucket(range);

  // GPU側の配列
  int *d_key, *d_bucket;

  // GPU側の配列のメモリの確保
  cudaMalloc(&d_key, n*sizeof(int));
  cudaMalloc(&d_bucket, range*sizeof(int));

  // CPU側の配列をGPU側にコピー
  cudaMemcpy(d_key, key.data(), n*sizeof(int), cudaMemcpyHostToDevice);

  // GPU側の配列を初期化
  cudaMemset(d_bucket, 0, range*sizeof(int));


  // GPUを何個のスレッドで動かすか
  int threads = 256;
  // 必要なブロック数は？
  int blocks = (n + threads - 1) / threads;

  // GPU側の関数kernelを呼び出す
  histogram_kernel<<<blocks, threads>>>(
    d_key,
    d_bucket,
    n
  );

  // GPU側の処理が終わるのを待つ
  cudaDeviceSynchronize();

  // GPU側のbucketをCPU側にコピー
  cudaMemcpy(bucket.data(), d_bucket, range*sizeof(int), cudaMemcpyDeviceToHost);

  // GPU側の配列のメモリの解放
  cudaFree(d_key);
  cudaFree(d_bucket);

  for (int i=0, j=0; i<range; i++){
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }

  printf("\n");
  return 0;
}


  /*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
*/

/*
original output
3 1 2 0 3 0 1 2 4 1 2 2 0 4 3 1 0 1 2 1 1 3 2 4 2 0 2 3 2 0 4 2 2 3 4 2 3 1 1 2 4 3 1 4 4 2 3 4 0 0 
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4
*/

/*
[ud06891@r3n11 06_cuda]$ nvcc 13_bucket_sort.cu
[ud06891@r3n11 06_cuda]$ ./a.out
3 1 2 0 3 0 1 2 4 1 2 2 0 4 3 1 0 1 2 1 1 3 2 4 2 0 2 3 2 0 4 2 2 3 4 2 3 1 1 2 4 3 1 4 4 2 3 4 0 0 
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 
*/

