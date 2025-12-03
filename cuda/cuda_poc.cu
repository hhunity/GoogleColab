// myproc.cu
#include "myproc.h"
#include <cuda_runtime.h>
#include <cufft.h>

struct CudaHandle {
    int width, height;
    // デバイスバッファ
    uint8_t* d_src;
    uint8_t* d_dst;
    float*   d_mag;
    float*   d_sobel;
    float*   d_corr;
    cufftComplex* d_fft1;
    cufftComplex* d_fft2;
    cufftComplex* d_fft_p;
    // ピーク検出用バッファなど…
    Peak* d_block_peaks;
    Peak* d_tmp_peaks;
    Peak* d_final_peak;
    Centroid* d_centroid;
    // cuFFT プランやストリーム
    cufftHandle fft_plan;
    cufftHandle ifft_plan;
    cudaStream_t stream;
    int peak_threads, peak_blocks, reduce_blocks;
};

CudaHandle* create_cuda_handle(int width, int height) {
    auto* h = new CudaHandle{};
    h->width = width; h->height = height;
    // cudaStreamCreate, cudaMalloc(必要分を全て確保), cufftPlan2d など
    return h;
}

void destroy_cuda_handle(CudaHandle* h) {
    if (!h) return;
    // cudaFree / cufftDestroy / cudaStreamDestroy
    delete h;
}

bool run_phase_correlation(CudaHandle* h,
                           const uint8_t* img_host,
                           const float* fft2_host /*半分幅 or フル幅*/ ) {
    // H2D 転送 (h->stream)
    // rotate/sobel/FFT/IFFT/peak reduce…
    // D2H は結果だけ
    return true;
}
