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

int main()
{
    if (is_cuda_available()) {
        std::cout << "CUDA is available.\n";
    } else {
        std::cout << "CUDA is NOT available.\n";
    }
}

int count = 0;
cudaGetDeviceCount(&count);

for (int i = 0; i < count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "GPU " << i << ": " << prop.name << "\n";
    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << "\n";
}

cudaError_t err = cudaFree(0);  // コンテキスト作成
if (err != cudaSuccess) ...

void* p = nullptr;
cudaError_t err = cudaMalloc(&p, 1024);
cudaFree(p);

int driver, runtime;
cudaDriverGetVersion(&driver);
cudaRuntimeGetVersion(&runtime);


