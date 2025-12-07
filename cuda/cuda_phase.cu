%%writefile cuda_phase.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cufft.h>   // cuFFTを使用してFFTを計算
#include "pfm_io.h"  // FFT結果をPFMで保存（OpenCV DFT互換のフル複素）
#include "cuda_kernels.cuh"

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

void save_pfm(std::string path,cufftComplex* complex,int width,int height,int c = 3) {

    std::vector<cufftComplex> fft_host(width*height);

    CHECK_CUDA(cudaMemcpy(fft_host.data(), complex, fft_host.size()* sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    PFMImage fft_pfm;
    fft_pfm.width  = width;
    fft_pfm.height = height;
    fft_pfm.channels = c; // R,I,ダミー
    fft_pfm.data.resize(static_cast<size_t>(width) * height * c);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = static_cast<size_t>(y) * width + x;
            if(c==3) {
                fft_pfm.data[idx * 3 + 0] = fft_host[idx].x;
                fft_pfm.data[idx * 3 + 1] = fft_host[idx].y;
                fft_pfm.data[idx * 3 + 2] = 0.0f;
            }else{
                fft_pfm.data[idx] = fft_host[idx].x;
            }
        }
    }
    write_pfm(path, fft_pfm);
}

int main(int argc, char** argv) {
    int iter    = (argc >= 4) ? std::stoi(argv[3]) : 1;
    int debug   = (argc >= 5) ? std::stoi(argv[4]) : 0;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // input img
    PFMImage img_1;
    if (!read_pfm("img_1.pfm", img_1)) return 1;
    PFMImage img_2;
    if (!read_pfm("img_2.pfm", img_2)) return 1;

    // output fft
    const int fft_w = img_1.width;
    const int fft_h = img_1.height;
    float* d_pfm_f = nullptr;   // FFT2入力用
    Peak* d_block_peaks = nullptr;
    Peak* d_tmp_peaks = nullptr;
    Peak* d_final_peak = nullptr;
    Centroid* d_centroid = nullptr;
    CHECK_CUDA(cudaMalloc(&d_pfm_f, img_1.data.size() * sizeof(float)));
    int total_pixels = img_1.width * img_1.height;
    int peak_threads = 256;
    int peak_blocks = (total_pixels + peak_threads - 1) / peak_threads;
    CHECK_CUDA(cudaMalloc(&d_block_peaks, peak_blocks * sizeof(Peak)));
    int reduce_blocks = (peak_blocks + peak_threads - 1) / peak_threads;
    if (reduce_blocks < 1) reduce_blocks = 1;
    CHECK_CUDA(cudaMalloc(&d_tmp_peaks, reduce_blocks * sizeof(Peak)));
    CHECK_CUDA(cudaMalloc(&d_final_peak, sizeof(Peak)));
    CHECK_CUDA(cudaMalloc(&d_centroid, sizeof(Centroid)));
    // Sobel on the rotated image (d_dst) — 常に実行
    dim3 block2(16, 16);
    dim3 grid2((img_1.width + block2.x - 1) / block2.x,
                (img_1.height + block2.y - 1) / block2.y);
    dim3 grid_full((fft_w + block2.x - 1) / block2.x,
                    (fft_h + block2.y - 1) / block2.y);

    // define FFFT woking memory & cuFFT plane（C2C フル複素）
    cufftHandle fft_plan;
    size_t fft_full_size  = static_cast<size_t>(fft_w) * fft_h;
    float*d_img1_f = nullptr; 
    float*d_img2_f = nullptr; 
    cufftComplex* d_fft1_full = nullptr;
    cufftComplex* d_fft2_full = nullptr;
    cufftComplex* d_fft_p_full = nullptr; // multiply結果
    CHECK_CUDA(cudaMalloc(&d_img1_f, fft_full_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_img2_f, fft_full_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fft_p_full,    fft_full_size * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_fft1_full,     fft_full_size * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_fft2_full,     fft_full_size * sizeof(cufftComplex)));
    if (cufftPlan2d(&fft_plan, fft_h, fft_w, CUFFT_C2C) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlan2d failed\n");
        return 1;
    }

    cufftSetStream(fft_plan, stream);

    for(int i = 0 ; i < iter ; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_img2_f, img_2.data.data(), img_2.data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        float_to_complex<<<grid2, block2, 0, stream>>>(d_img2_f, d_fft2_full, fft_w, fft_h);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        auto t_start = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaMemcpyAsync(d_img1_f, img_1.data.data(), img_1.data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

        // 実数→複素に詰めてから FFT 実行（C2C フル複素）
        float_to_complex<<<grid2, block2, 0, stream>>>(d_img1_f, d_fft1_full, fft_w, fft_h);
        if (cufftExecC2C(fft_plan, d_fft1_full, d_fft1_full, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2C forward failed\n");
            return 1;
        }
        if(debug) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            save_pfm("cuda_fft1_full.pfm", d_fft1_full, fft_w, fft_h);
            if (cufftExecC2C(fft_plan, d_fft1_full, d_fft_p_full, CUFFT_INVERSE) != CUFFT_SUCCESS) {
                std::fprintf(stderr, "cufftExecC2C inverse failed\n");
                return 1;
            }
            CHECK_CUDA(cudaStreamSynchronize(stream));
            save_pfm("cuda_ifft1_full.pfm", d_fft_p_full, fft_w, fft_h,1);
        }
        if (cufftExecC2C(fft_plan, d_fft2_full, d_fft2_full, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2C forward failed\n");
            return 1;
        }
        if(debug) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            save_pfm("cuda_fft2_full.pfm", d_fft2_full, fft_w, fft_h);
        }
        // P = FFT1 * conj(FFT2)
        int threads = 256;
        int blocks = (static_cast<int>(fft_full_size) + threads - 1) / threads;
        complex_mul_conj<<<blocks, threads, 0, stream>>>(d_fft1_full, d_fft2_full, d_fft_p_full, static_cast<int>(fft_full_size));
        normalize_phase<<<blocks, threads, 0, stream>>>(d_fft_p_full, static_cast<int>(fft_full_size), 2.220446049250313e-16);
        // IFFTで相関ピークを得る（フル複素 C2C）
        if (cufftExecC2C(fft_plan, d_fft_p_full, d_fft_p_full, CUFFT_INVERSE) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2C inverse failed\n");
            return 1;
        }
        
        if(debug) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            save_pfm("cuda_C.pfm", d_fft_p_full, fft_w, fft_h,1);
        }
        // スケール＆シフト（DC中心）、複素→実
        // opencvのphaseCorrの内部も1/(wxh)されてないものが出るので、スケールは１で良い
        float inv_scale = 1.0f;
        complex_real_scale_shift<<<grid2, block2, 0, stream>>>(d_fft_p_full, d_pfm_f, img_1.width, img_1.height, inv_scale);
        // 相関出力をPFMに保存（ループ最後にホストへコピー）

        if(debug) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            save_pfm_real("cuda_C_shift.pfm", d_pfm_f, fft_w, fft_h);
        }
        // GPUでピークとサブピクセル重心を計算
        block_peak<<<peak_blocks, peak_threads, peak_threads * (sizeof(float) + sizeof(int)), stream>>>(d_pfm_f, total_pixels, d_block_peaks);
        reduce_peak<<<reduce_blocks, peak_threads, peak_threads * (sizeof(float) + sizeof(int)), stream>>>(d_block_peaks, peak_blocks, d_tmp_peaks);
        reduce_peak<<<1, peak_threads, peak_threads * (sizeof(float) + sizeof(int)), stream>>>(d_tmp_peaks, reduce_blocks, d_final_peak);
        CHECK_CUDA(cudaMemsetAsync(d_centroid, 0, sizeof(Centroid), stream));

        centroid5x5<<<1, 25, 0, stream>>>(d_pfm_f, img_1.width, img_1.height, d_final_peak, d_centroid);
        
        Peak peak_host{};
        Centroid cent_host{};
        CHECK_CUDA(cudaMemcpyAsync(&peak_host, d_final_peak, sizeof(Peak), cudaMemcpyDeviceToHost,stream));
        CHECK_CUDA(cudaMemcpyAsync(&cent_host, d_centroid, sizeof(Centroid), cudaMemcpyDeviceToHost,stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        auto t_end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        float fft_ms = 0.0f, peak_ms = 0.0f;
        // CHECK_CUDA(cudaEventElapsedTime(&fft_ms, ev_fft_start, ev_fft_end));
        // CHECK_CUDA(cudaEventElapsedTime(&peak_ms, ev_peak_start, ev_peak_end));
        std::printf("[%d/%d] 処理時間: %.3f ms fft %.3f ms peak %.3f ms \n",i,iter,elapsed_ms,fft_ms,peak_ms);

        // ピーク・重心結果をホストに戻して表示
        double peak_x = peak_host.idx % img_1.width;
        double peak_y = peak_host.idx / img_1.width;
        double t_x = (cent_host.m00 > 0.0f) ? (cent_host.m10 / cent_host.m00) : peak_x;
        double t_y = (cent_host.m00 > 0.0f) ? (cent_host.m01 / cent_host.m00) : peak_y;
        // d_pfm_f は scale_and_shift で 1/(W*H) を掛け済みなので、さらにサイズで割らない
        // double response = peak_host.val/(img_1.width * img_1.height);
        // peak_hostは本当のピークのみ。m00が5x5の集合和なのでこれがopencvと一致する
        double response = cent_host.m00/(img_1.width * img_1.height); 
        double center_x = static_cast<double>(img_1.width)  / 2.0;
        double center_y = static_cast<double>(img_1.height) / 2.0;
        double shift_x = center_x - t_x;
        double shift_y = center_y - t_y;
        if (shift_x > center_x) shift_x -= img_1.width;
        if (shift_y > center_y) shift_y -= img_1.height;
        std::printf("phaseCorrelate peak=(%.0f,%.0f) subpix=(%.4f,%.4f) shift=(%.4f,%.4f) response=%.6f\n",
                    peak_x, peak_y, t_x, t_y, shift_x, shift_y, response);
    }
    
    cudaStreamDestroy(stream);
    cufftDestroy(fft_plan);
    cudaFree(d_fft1_full);
    cudaFree(d_fft2_full);
    cudaFree(d_fft_p_full);
    cudaFree(d_img1_f);
    cudaFree(d_img2_f);
    cudaFree(d_pfm_f);
    cudaFree(d_block_peaks);
    cudaFree(d_tmp_peaks);
    cudaFree(d_final_peak);
    cudaFree(d_centroid);
    return 0;
}
