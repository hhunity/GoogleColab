#include <cuda_runtime.h>
#include <iostream>

bool is_cuda_available()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount error: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    return deviceCount > 0;
}

int main() {
    printf("Launching a simple GPU kernel...\n");

    if (is_cuda_available()) {
        std::cout << "CUDA is available.\n";
    } else {
        std::cout << "CUDA is NOT available.\n";
        return;
    }

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

