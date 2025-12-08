%%writefile cuda_phase.cu
// 半精度 cuFFT Xt 版
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cufftXt.h>
#include "pfm_io.h"
#include "pgm_io.h"
#include "cuda_kernels_half.cuh"

// input/output helperは省略（ベンチ用）

int main(int argc, char** argv) {
    int split_x = (argc >= 2) ? std::stoi(argv[1]) : 1;
    int split_y = (argc >= 3) ? std::stoi(argv[2]) : 1;
    int iter    = (argc >= 4) ? std::stoi(argv[3]) : 1;
    if (split_x < 1) split_x = 1;
    if (split_y < 1) split_y = 1;

    cudaStream_t streams[2];
    CHECK_CUDA(cudaStreamCreate(&streams[0]));
    CHECK_CUDA(cudaStreamCreate(&streams[1]));

    Image img1, img2;
    if (!read_pgm("img_1.pgm", img1)) return 1;
    if (!read_pgm("img_2.pgm", img2)) return 1;
    if (img1.width != img2.width || img1.height != img2.height) {
        std::fprintf(stderr, "Input sizes mismatch\n");
        return 1;
    }
    if (img1.width % split_x != 0 || img1.height % split_y != 0) {
        std::fprintf(stderr, "Image size must be divisible by split_x, split_y\n");
        return 1;
    }

    const int tile_w = img1.width / split_x;
    const int tile_h = img1.height / split_y;
    const int tile_pixels = tile_w * tile_h;
    const int batch = split_x * split_y;
    if (tile_w > MAX_TILE_DIM || tile_h > MAX_TILE_DIM) {
        std::fprintf(stderr, "tile too large for constant hann table\n");
        return 1;
    }
    std::vector<float> hann_x(tile_w), hann_y(tile_h);
    if (tile_w > 1) for (int x = 0; x < tile_w; ++x) hann_x[x] = 0.5f * (1.0f - cosf(2.0f * 3.1415926535f * x / (tile_w - 1)));
    else hann_x[0] = 1.0f;
    if (tile_h > 1) for (int y = 0; y < tile_h; ++y) hann_y[y] = 0.5f * (1.0f - cosf(2.0f * 3.1415926535f * y / (tile_h - 1)));
    else hann_y[0] = 1.0f;
    CHECK_CUDA(cudaMemcpyToSymbol(c_hann_x, hann_x.data(), static_cast<size_t>(tile_w) * sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_hann_y, hann_y.data(), static_cast<size_t>(tile_h) * sizeof(float)));

    size_t batch_pixels = static_cast<size_t>(tile_pixels) * batch;
    unsigned char* d_img1_full[2] = {nullptr, nullptr};
    unsigned char* d_img2_full[2] = {nullptr, nullptr};
    float* d_img1_f[2] = {nullptr, nullptr};
    float* d_img2_f[2] = {nullptr, nullptr};
    __half2* d_fft1[2] = {nullptr, nullptr};
    __half2* d_fft2[2] = {nullptr, nullptr};
    __half2* d_fft_p[2] = {nullptr, nullptr};
    cufftComplex* d_fft1_f[2] = {nullptr, nullptr};
    cufftComplex* d_fft2_f[2] = {nullptr, nullptr};
    cufftComplex* d_fft_p_f[2] = {nullptr, nullptr};
    float* d_corr[2] = {nullptr, nullptr};
    Peak* d_block_peaks[2] = {nullptr, nullptr};
    Peak* d_final_peaks[2] = {nullptr, nullptr};
    Centroid* d_centroids[2] = {nullptr, nullptr};

    int peak_threads = 256;
    int peak_blocks = (tile_pixels + peak_threads - 1) / peak_threads;

    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cudaMalloc(&d_img1_full[i], static_cast<size_t>(img1.width) * img1.height * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&d_img2_full[i], static_cast<size_t>(img2.width) * img2.height * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&d_img1_f[i], static_cast<size_t>(img1.width) * img1.height * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_img2_f[i], static_cast<size_t>(img2.width) * img2.height * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_fft1[i], batch_pixels * sizeof(__half2)));
        CHECK_CUDA(cudaMalloc(&d_fft2[i], batch_pixels * sizeof(__half2)));
        CHECK_CUDA(cudaMalloc(&d_fft_p[i], batch_pixels * sizeof(__half2)));
        CHECK_CUDA(cudaMalloc(&d_fft1_f[i], batch_pixels * sizeof(cufftComplex)));
        CHECK_CUDA(cudaMalloc(&d_fft2_f[i], batch_pixels * sizeof(cufftComplex)));
        CHECK_CUDA(cudaMalloc(&d_fft_p_f[i], batch_pixels * sizeof(cufftComplex)));
        CHECK_CUDA(cudaMalloc(&d_corr[i], batch_pixels * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_block_peaks[i], peak_blocks * sizeof(Peak)));
        CHECK_CUDA(cudaMalloc(&d_final_peaks[i], static_cast<size_t>(batch) * sizeof(Peak)));
        CHECK_CUDA(cudaMalloc(&d_centroids[i], static_cast<size_t>(batch) * sizeof(Centroid)));
    }

    cufftHandle fft_plan[2], ifft_plan[2];
    cufftHandle ifft_plan_f[2];
    void* d_work[2] = {nullptr, nullptr};
    long long n[2] = {tile_h, tile_w};
    long long inembed[2] = {tile_h, tile_w};
    long long onembed[2] = {tile_h, tile_w};
    long long istride = 1, ostride = 1;
    long long idist = tile_w * tile_h;
    long long odist = tile_w * tile_h;
    for (int i = 0; i < 2; ++i) {
        size_t work_fwd = 0, work_inv = 0;
        cufftCreate(&fft_plan[i]);
        cufftCreate(&ifft_plan[i]);
        cufftPlanMany(&ifft_plan_f[i], 2, reinterpret_cast<int*>(n),
                      reinterpret_cast<int*>(inembed), static_cast<int>(istride), static_cast<int>(idist),
                      reinterpret_cast<int*>(onembed), static_cast<int>(ostride), static_cast<int>(odist),
                      CUFFT_C2C, batch);
        cufftSetAutoAllocation(fft_plan[i], 0);
        cufftSetAutoAllocation(ifft_plan[i], 0);
        cufftXtMakePlanMany(fft_plan[i], 2, n,
                            inembed, istride, idist, CUDA_C_16F,
                            onembed, ostride, odist, CUDA_C_16F,
                            batch, &work_fwd, CUDA_C_16F);
        cufftXtMakePlanMany(ifft_plan[i], 2, n,
                            inembed, istride, idist, CUDA_C_16F,
                            onembed, ostride, odist, CUDA_C_16F,
                            batch, &work_inv, CUDA_C_16F);
        size_t work = std::max(work_fwd, work_inv);
        if (work > 0) {
            CHECK_CUDA(cudaMalloc(&d_work[i], work));
            cufftSetWorkArea(fft_plan[i], d_work[i]);
            cufftSetWorkArea(ifft_plan[i], d_work[i]);
        }
        cufftSetStream(fft_plan[i], streams[i]);
        cufftSetStream(ifft_plan[i], streams[i]);
        cufftSetStream(ifft_plan_f[i], streams[i]);
    }

    dim3 block(16, 16);
    dim3 grid((tile_w + block.x - 1) / block.x, (tile_h + block.y - 1) / block.y);
    dim3 grid3(grid.x, grid.y, batch);
    dim3 block2(16, 16);
    dim3 grid2((img1.width + block2.x - 1) / block2.x,
               (img1.height + block2.y - 1) / block2.y);

    Peak* peaks_host = nullptr;
    Centroid* cent_host = nullptr;
    CHECK_CUDA(cudaHostAlloc(&peaks_host, static_cast<size_t>(batch) * sizeof(Peak), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&cent_host, static_cast<size_t>(batch) * sizeof(Centroid), cudaHostAllocDefault));

    // img2 は固定: u8 -> float -> pack half -> FFT
    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d_img2_full[i], img2.data.data(),
                                   static_cast<size_t>(img2.width) * img2.height * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice, streams[i]));
        u8_to_float_window<<<grid2, block2, 0, streams[i]>>>(d_img2_full[i], d_img2_f[i], img2.width, img2.height);
        pack_tiles_to_half_complex<<<grid3, block, 0, streams[i]>>>(d_img2_f[i], d_fft2[i],
                                                                    img2.width, img2.height,
                                                                    tile_w, tile_h, split_x, split_y);
        if (cufftXtExec(fft_plan[i], d_fft2[i], d_fft2[i], CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftXtExec forward failed\n");
            return 1;
        }
        int blocks_c = (static_cast<int>(batch_pixels) + 255) / 256;
        half2_to_float2<<<blocks_c, 256, 0, streams[i]>>>(d_fft2[i], d_fft2_f[i], static_cast<int>(batch_pixels));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iter; ++i) {
        int buf = i & 1;
        cudaStream_t stream = streams[buf];
        CHECK_CUDA(cudaMemcpyAsync(d_img1_full[buf], img1.data.data(),
                                   static_cast<size_t>(img1.width) * img1.height * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice, stream));
        u8_to_float_window<<<grid2, block2, 0, stream>>>(d_img1_full[buf], d_img1_f[buf], img1.width, img1.height);
        pack_tiles_to_half_complex<<<grid3, block, 0, stream>>>(d_img1_f[buf], d_fft1[buf],
                                                                img1.width, img1.height,
                                                                tile_w, tile_h, split_x, split_y);
        if (cufftXtExec(fft_plan[buf], d_fft1[buf], d_fft1[buf], CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftXtExec forward failed\n");
            return 1;
        }
        int blocks_c = (static_cast<int>(batch_pixels) + 255) / 256;
        half2_to_float2<<<blocks_c, 256, 0, stream>>>(d_fft1[buf], d_fft1_f[buf], static_cast<int>(batch_pixels));
        int threads = 256;
        int blocks = (static_cast<int>(batch_pixels) + threads - 1) / threads;
        complex_mul_conj_normalize_f<<<blocks, threads, 0, stream>>>(d_fft1_f[buf], d_fft2_f[buf], d_fft_p_f[buf],
                                                                     static_cast<int>(batch_pixels), 1e-2f);
        if (cufftExecC2C(ifft_plan_f[buf], d_fft_p_f[buf], d_fft_p_f[buf], CUFFT_INVERSE) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2C inverse failed\n");
            return 1;
        }
        // ホスト側で response を (tile_w*tile_h) で割っているので、ここではスケールせずそのまま
        float inv_scale = 1.0f;
        complex_real_scale_shift_batch_f<<<grid3, block, 0, stream>>>(d_fft_p_f[buf], d_corr[buf],
                                                                      tile_w, tile_h, batch, inv_scale);
        size_t shared_bytes = peak_threads * (sizeof(float) + sizeof(int));
        for (int b = 0; b < batch; ++b) {
            const float* corr_tile = d_corr[buf] + static_cast<size_t>(b) * tile_pixels;
            block_peak<<<peak_blocks, peak_threads, shared_bytes, stream>>>(corr_tile, tile_pixels, d_block_peaks[buf]);
            CHECK_CUDA(cudaMemsetAsync(d_centroids[buf] + b, 0, sizeof(Centroid), stream));
            final_peak_and_centroid<<<1, peak_threads, shared_bytes, stream>>>(d_block_peaks[buf], peak_blocks,
                                                                               corr_tile, tile_w, tile_h,
                                                                               d_final_peaks[buf] + b, d_centroids[buf] + b);
        }
        int prev = buf ^ 1;
        if (i > 0) {
            CHECK_CUDA(cudaMemcpyAsync(peaks_host, d_final_peaks[prev], static_cast<size_t>(batch) * sizeof(Peak),
                                   cudaMemcpyDeviceToHost, streams[prev]));
            CHECK_CUDA(cudaMemcpyAsync(cent_host, d_centroids[prev], static_cast<size_t>(batch) * sizeof(Centroid),
                                   cudaMemcpyDeviceToHost, streams[prev]));
            CHECK_CUDA(cudaStreamSynchronize(streams[prev]));
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Total %.3f ms, avg %.3f ms/frame over %d iterations\n", total_ms, total_ms / iter, iter);

    int last = (iter - 1) & 1;
    CHECK_CUDA(cudaMemcpyAsync(peaks_host, d_final_peaks[last], static_cast<size_t>(batch) * sizeof(Peak),
                                   cudaMemcpyDeviceToHost, streams[last]));
    CHECK_CUDA(cudaMemcpyAsync(cent_host, d_centroids[last], static_cast<size_t>(batch) * sizeof(Centroid),
                                   cudaMemcpyDeviceToHost, streams[last]));
    CHECK_CUDA(cudaStreamSynchronize(streams[last]));
    double center_x = static_cast<double>(tile_w) / 2.0;
    double center_y = static_cast<double>(tile_h) / 2.0;
    for (int b = 0; b < batch; ++b) {
        int tile_idx = b;
        int tile_tx = tile_idx % split_x;
        int tile_ty = tile_idx / split_x;
        Peak pk = peaks_host[b];
        Centroid ct = cent_host[b];
        double peak_x = pk.idx % tile_w;
        double peak_y = pk.idx / tile_w;
        double t_x = (ct.m00 > 0.0f) ? (ct.m10 / ct.m00) : peak_x;
        double t_y = (ct.m00 > 0.0f) ? (ct.m01 / ct.m00) : peak_y;
        double shift_x = center_x - t_x;
        double shift_y = center_y - t_y;
        if (shift_x > center_x) shift_x -= tile_w;
        if (shift_y > center_y) shift_y -= tile_h;
        double response = (ct.m00 > 0.0f) ? (ct.m00 / static_cast<float>(tile_w * tile_h)) : 0.0;
        std::printf("tile %d (%d,%d): peak=(%.0f,%.0f,val=%.6f) subpix=(%.4f,%.4f) shift=(%.4f,%.4f) m00=%.6f response=%.6f\n",
                    b, tile_tx, tile_ty,
                    peak_x, peak_y, pk.val,
                    t_x, t_y, shift_x, shift_y, ct.m00, response);
    }

    // 後始末
    for (int i = 0; i < 2; ++i) {
        cufftDestroy(fft_plan[i]);
        cufftDestroy(ifft_plan[i]);
        cufftDestroy(ifft_plan_f[i]);
        if (d_work[i]) cudaFree(d_work[i]);
        cudaFree(d_img1_full[i]);
        cudaFree(d_img2_full[i]);
        cudaFree(d_img1_f[i]);
        cudaFree(d_img2_f[i]);
        cudaFree(d_fft1[i]);
        cudaFree(d_fft2[i]);
        cudaFree(d_fft_p[i]);
        cudaFree(d_fft1_f[i]);
        cudaFree(d_fft2_f[i]);
        cudaFree(d_fft_p_f[i]);
        cudaFree(d_corr[i]);
        cudaFree(d_block_peaks[i]);
        cudaFree(d_final_peaks[i]);
        cudaFree(d_centroids[i]);
    }
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFreeHost(peaks_host);
    cudaFreeHost(cent_host);
    return 0;
}
