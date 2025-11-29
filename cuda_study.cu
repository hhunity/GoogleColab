%%writefile test.cu
#include <stdio.h>

__global__ void gpu_kernel(int *data) {
    //全体のthread番号
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("[%d]Hello from GPU! thread %d blockIdx %d blockDim %d gridDim %d\n",global_idx,threadIdx.x,blockIdx.x,blockDim.x,gridDim.x);
    //thread   ブロック内のスレッド番号
    //blockIdx グリッド内でのブロック番号
    //blockDim １ブロックのスレッド数
    //gridDim ブロック数

    data[global_idx] *= 2;
}

int main() {
    printf("Launching a simple GPU kernel...\n");

    const int N = 10;
    int h_data[N] = {1,2,3,4,5,6,7,8,9,10};
    int* d_data;

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N*sizeof(int), cudaMemcpyHostToDevice);

    gpu_kernel<<<2, N/2>>>(d_data); //1Block, 5Thradで起動。引数はなし
    cudaDeviceSynchronize(); // GPUの処理が完了するまで待機
    cudaMemcpy(h_data, d_data, N*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++) {
        printf("[%d]=%d",i,h_data[i]);
    }

    printf("\nGPU kernel finished. Program exiting.\n");
    return 0;
}