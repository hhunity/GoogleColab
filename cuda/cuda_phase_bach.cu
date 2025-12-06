%%writefile cuda_phase.cu
// 入力PFM 2枚をタイル分割して Phase Correlation を計算するサンプル
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cufft.h>
#include "pfm_io.h"
#include "cuda_kernels.cuh"

void save_pfm(const std::string& path, const cufftComplex* src, int width, int height, int c = 3) {
    std::vector<cufftComplex> host(static_cast<size_t>(width) * height);
    CHECK_CUDA(cudaMemcpy(host.data(), src, host.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    PFMImage out;
    out.width = width;
    out.height = height;
    out.channels = c;
    out.data.resize(static_cast<size_t>(width) * height * c);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = static_cast<size_t>(y) * width + x;
            if (c == 3) {
                out.data[idx * 3 + 0] = host[idx].x;
                out.data[idx * 3 + 1] = host[idx].y;
                out.data[idx * 3 + 2] = 0.0f;
            } else {
                out.data[idx] = host[idx].x;
            }
        }
    }
    write_pfm(path, out);
}

void save_pfm_real(const std::string& path, const float* src, int width, int height) {
    std::vector<float> host(static_cast<size_t>(width) * height);
    CHECK_CUDA(cudaMemcpy(host.data(), src, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    PFMImage out;
    out.width = width;
    out.height = height;
    out.channels = 1;
    out.data = std::move(host);
    write_pfm(path, out);
}

int main(int argc, char** argv) {
    // 使い方: ./cuda_phase_bach [split_x=1] [split_y=1]
    int split_x = (argc >= 2) ? std::stoi(argv[1]) : 1;
    int split_y = (argc >= 3) ? std::stoi(argv[2]) : 1;
    if (split_x < 1) split_x = 1;
    if (split_y < 1) split_y = 1;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // input img (PFM top-down)
    PFMImage img1, img2;
    if (!read_pfm("img_1.pfm", img1)) return 1;
    if (!read_pfm("img_2.pfm", img2)) return 1;
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

    // デバイスバッファ（タイル全体分）
    float* d_img1 = nullptr;
    float* d_img2 = nullptr;
    float* d_corr = nullptr;
    Peak* d_block_peaks = nullptr;
    Peak* d_tmp_peaks = nullptr;
    Peak* d_final_peak = nullptr;
    Centroid* d_centroid = nullptr;
    size_t batch_pixels = static_cast<size_t>(tile_pixels) * batch;
    CHECK_CUDA(cudaMalloc(&d_img1, batch_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_img2, batch_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_corr, batch_pixels * sizeof(float)));
    int peak_threads = 256;
    int peak_blocks = (tile_pixels + peak_threads - 1) / peak_threads;
    CHECK_CUDA(cudaMalloc(&d_block_peaks, peak_blocks * sizeof(Peak)));
    int reduce_blocks = (peak_blocks + peak_threads - 1) / peak_threads;
    if (reduce_blocks < 1) reduce_blocks = 1;
    CHECK_CUDA(cudaMalloc(&d_tmp_peaks, reduce_blocks * sizeof(Peak)));
    CHECK_CUDA(cudaMalloc(&d_final_peak, sizeof(Peak)));
    CHECK_CUDA(cudaMalloc(&d_centroid, sizeof(Centroid)));

    // cuFFT plan (C2C) batched
    cufftHandle fft_plan, ifft_plan;
    // size_t fft_elems_full = static_cast<size_t>(tile_w) * tile_h;
    cufftComplex* d_fft1 = nullptr;
    cufftComplex* d_fft2 = nullptr;
    cufftComplex* d_fft_p = nullptr;
    CHECK_CUDA(cudaMalloc(&d_fft1, batch_pixels * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_fft2, batch_pixels * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_fft_p, batch_pixels * sizeof(cufftComplex)));

    int n[2] = {tile_h, tile_w};
    int inembed[2] = {tile_h, tile_w};
    int onembed[2] = {tile_h, tile_w};
    int istride = 1, ostride = 1;
    int idist = tile_w * tile_h;
    int odist = tile_w * tile_h;
    if (cufftPlanMany(&fft_plan, 2, n,
                      inembed, istride, idist,
                      onembed, ostride, odist,
                      CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlanMany failed\n");
        return 1;
    }
    if (cufftPlanMany(&ifft_plan, 2, n,
                      inembed, istride, idist,
                      onembed, ostride, odist,
                      CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlanMany (C2C inverse) failed\n");
        return 1;
    }
    cufftSetStream(fft_plan, stream);
    cufftSetStream(ifft_plan, stream);

    dim3 block(16, 16);
    dim3 grid((tile_w + block.x - 1) / block.x, (tile_h + block.y - 1) / block.y);
    dim3 grid3(grid.x, grid.y, batch);

    std::vector<Peak> peaks_host(batch);
    std::vector<Centroid> cent_host(batch);

    // ホスト側で分割した順に並べ替えてからまとめて転送（batch x H x W 連続）
    // バッチ０→バッチ１→バッチ２と連続に並び替え
    for (int ty = 0; ty < split_y; ++ty) {
        for (int tx = 0; tx < split_x; ++tx) {
            int b = ty * split_x + tx;
            const float* src1 = img1.data.data() + static_cast<size_t>(ty * tile_h) * img1.width + tx * tile_w;
            const float* src2 = img2.data.data() + static_cast<size_t>(ty * tile_h) * img2.width + tx * tile_w;
            float* dst1 = d_img1 + static_cast<size_t>(b) * tile_pixels;
            float* dst2 = d_img2 + static_cast<size_t>(b) * tile_pixels;
            // ピッチ付きコピーで元画像のストライドを保ったままタイルを転送
            CHECK_CUDA(cudaMemcpy2DAsync(dst1, tile_w * sizeof(float),
                                         src1, img1.width * sizeof(float),
                                         tile_w * sizeof(float), tile_h,
                                         cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpy2DAsync(dst2, tile_w * sizeof(float),
                                         src2, img2.width * sizeof(float),
                                         tile_w * sizeof(float), tile_h,
                                         cudaMemcpyHostToDevice, stream));
        }
    }

    // 実→複素へ変換
    float_to_complex_batch<<<grid3, block, 0, stream>>>(d_img1, d_fft1, tile_w, tile_h, batch);
    float_to_complex_batch<<<grid3, block, 0, stream>>>(d_img2, d_fft2, tile_w, tile_h, batch);

    // FFT (C2C forward) batched
    if (cufftExecC2C(fft_plan, d_fft1, d_fft1, CUFFT_FORWARD) != CUFFT_SUCCESS ||
        cufftExecC2C(fft_plan, d_fft2, d_fft2, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftExecC2C forward failed\n");
        return 1;
    }

    // 相互スペクトル（バッチ一括）
    int threads = 256;
    int blocks = (static_cast<int>(batch_pixels) + threads - 1) / threads;
    complex_mul_conj<<<blocks, threads, 0, stream>>>(d_fft1, d_fft2, d_fft_p, static_cast<int>(batch_pixels));
    normalize_phase<<<blocks, threads, 0, stream>>>(d_fft_p, static_cast<int>(batch_pixels), 1e-16f);

    // IFFT (C2C inverse) batched（バッチ一括）
    if (cufftExecC2C(ifft_plan, d_fft_p, d_fft_p, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftExecC2C inverse failed\n");
        return 1;
    }

    // スケール＆中心シフト（バッチ）
    float inv_scale = 1.0f;
    complex_real_scale_shift_batch<<<grid3, block, 0, stream>>>(d_fft_p, d_corr, tile_w, tile_h, batch, inv_scale);

    // 各タイルのピーク検出
    size_t shared_bytes = peak_threads * (sizeof(float) + sizeof(int));
    for (int b = 0; b < batch; ++b) {
        const float* corr_tile = d_corr + static_cast<size_t>(b) * tile_pixels;
        block_peak<<<peak_blocks, peak_threads, shared_bytes, stream>>>(corr_tile, tile_pixels, d_block_peaks);
        reduce_peak<<<reduce_blocks, peak_threads, shared_bytes, stream>>>(d_block_peaks, peak_blocks, d_tmp_peaks);
        reduce_peak<<<1, peak_threads, shared_bytes, stream>>>(d_tmp_peaks, reduce_blocks, d_final_peak);
        CHECK_CUDA(cudaMemsetAsync(d_centroid, 0, sizeof(Centroid), stream));
        centroid5x5<<<1, 25, 0, stream>>>(corr_tile, tile_w, tile_h, d_final_peak, d_centroid);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(&peaks_host[b], d_final_peak, sizeof(Peak), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&cent_host[b], d_centroid, sizeof(Centroid), cudaMemcpyDeviceToHost));
    }

    double center_x = static_cast<double>(tile_w) / 2.0;
    double center_y = static_cast<double>(tile_h) / 2.0;
    for (int b = 0; b < batch; ++b) {
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
        double response = ct.m00 / (tile_w * tile_h);
        std::printf("tile %d (%d,%d): peak=(%.0f,%.0f) subpix=(%.4f,%.4f) shift=(%.4f,%.4f) response=%.6f\n",
                    b, b % split_x, b / split_x,
                    peak_x, peak_y, t_x, t_y, shift_x, shift_y, response);
    }

    cudaStreamDestroy(stream);
    cufftDestroy(fft_plan);
    cufftDestroy(ifft_plan);
    cudaFree(d_fft1);
    cudaFree(d_fft2);
    cudaFree(d_fft_p);
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_corr);
    cudaFree(d_block_peaks);
    cudaFree(d_tmp_peaks);
    cudaFree(d_final_peak);
    cudaFree(d_centroid);
    return 0;
}
