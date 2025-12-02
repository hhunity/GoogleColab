%%writefile rot_sobel.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cufft.h>   // cuFFTを使用してFFTを計算
#include "pgm_io.h"
#include "pfm_io.h"  // FFT結果をPFMで保存（OpenCV DFT互換のフル複素）

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",              \
                         cudaGetErrorString(err__), err__, __FILE__, __LINE__);\
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Rotate an image (grayscale) around the top-left origin by `angle_rad`.
// Bilinear sampling with constant border (0) to match OpenCV warpAffine defaults.
__global__ void rotate_origin(const unsigned char* src, unsigned char* dst,
                              int width, int height, float angle_rad,
                              unsigned char border_value = 0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Inverse rotation to fetch from the source (avoids holes in dst).
    float c = cosf(angle_rad);
    float s = sinf(angle_rad);
    float src_x =  c * x + s * y;
    float src_y = -s * x + c * y;

    // Bilinear interpolation like OpenCV's default for warpAffine.
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    float dx = src_x - x0;
    float dy = src_y - y0;

    float accum = 0.0f;
    float wsum = 0.0f;
    // Iterate neighbors (x0,y0) to (x0+1, y0+1)
    for (int j = 0; j <= 1; ++j) {
        int yy = y0 + j;
        if (yy < 0 || yy >= height) continue;
        float wy = (j == 0) ? (1.0f - dy) : dy;
        for (int i = 0; i <= 1; ++i) {
            int xx = x0 + i;
            if (xx < 0 || xx >= width) continue;
            float wx = (i == 0) ? (1.0f - dx) : dx;
            float w = wx * wy;
            accum += w * static_cast<float>(src[yy * width + xx]);
            wsum += w;
        }
    }
    float out_val;
    if (wsum > 0.0f) {
        out_val = accum / wsum;
    } else {
        out_val = static_cast<float>(border_value);
    }
    // Clamp to 0-255
    if (out_val < 0.0f) out_val = 0.0f;
    if (out_val > 255.0f) out_val = 255.0f;
    dst[y * width + x] = static_cast<unsigned char>(out_val + 0.5f);
}

__device__ __forceinline__ int reflect101(int p, int len) {
    // OpenCV BORDER_REFLECT_101
    if (len == 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) p = -p;
        if (p >= len) p = 2 * len - p - 2;
    }
    return p;
}

// Sobel filter + magnitude fused (OpenCV ksize=3, scale=1, BORDER_DEFAULT).
// Outputs 8-bit magnitude (clamped) directly.
__global__ void sobel3x3_mag(const unsigned char* src, float* mag_out,
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sumx = 0.0f;
    float sumy = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        int yy = reflect101(y + ky, height);
        for (int kx = -1; kx <= 1; ++kx) {
            int xx = reflect101(x + kx, width);
            float v = static_cast<float>(src[yy * width + xx]);
            float kx_coeff = (kx == 0) ? 0.0f : (kx == -1 ? -1.0f : 1.0f);
            float ky_coeff = (ky == 0) ? 0.0f : (ky == -1 ? -1.0f : 1.0f);
            float weightx = ((ky == 0) ? 2.0f : 1.0f) * kx_coeff;
            float weighty = ((kx == 0) ? 2.0f : 1.0f) * ky_coeff;
            sumx += weightx * v;
            sumy += weighty * v;
        }
    }
    float mag = sqrtf(sumx * sumx + sumy * sumy);
    mag_out[y * width + x] = mag;
}

// u8 -> float 変換（R2C入力用）＋2Dハン窓を適用してリークを低減
__global__ void u8_to_float_window(float* src, float* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const float pi = 3.1415926535f;
    float wx = 1.0f;
    float wy = 1.0f;
    if (width > 1) {
        wx = 0.5f * (1.0f - cosf(2.0f * pi * x / (width - 1)));
    }
    if (height > 1) {
        wy = 0.5f * (1.0f - cosf(2.0f * pi * y / (height - 1)));
    }
    float v = static_cast<float>(src[y * width + x]) * wx * wy;
    dst[y * width + x] = v;
}

// パックされたR2C出力（幅 = width/2+1）を、OpenCV DFT互換のフル複素(width)に左右対称展開する。
// src: height x (width/2+1) 複素, dst: height x width 複素
__global__ void unpack_half_to_full(const cufftComplex* src, cufftComplex* dst,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_cols = width / 2 + 1;
    if (x >= width || y >= height) return;
    // 左半分はそのまま
    if (x < half_cols) {
        size_t idx_src = static_cast<size_t>(y) * half_cols + x;
        size_t idx_dst = static_cast<size_t>(y) * width + x;
        dst[idx_dst] = src[idx_src];
    } else { // 右半分は共役対称から復元
        int src_x = width - x;
        size_t idx_src = static_cast<size_t>(y) * half_cols + src_x;
        size_t idx_dst = static_cast<size_t>(y) * width + x;
        cufftComplex v = src[idx_src];
        v.y = -v.y;
        dst[idx_dst] = v;
    }
}

// 複素乗算: out = a * conj(b)
__global__ void complex_mul_conj(const cufftComplex* a, const cufftComplex* b,
                                 cufftComplex* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float ar = a[idx].x, ai = a[idx].y;
    float br = b[idx].x, bi = -b[idx].y; // conj(b)
    out[idx].x = ar * br - ai * bi;
    out[idx].y = ar * bi + ai * br;
}

// フェーズ相関用に振幅で正規化: out = in / |in| (ゼロ除算回避)
__global__ void normalize_phase(cufftComplex* inout, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float ar = inout[idx].x;
    float ai = inout[idx].y;
    float mag = sqrtf(ar * ar + ai * ai);
    if (mag < eps) mag = eps;
    inout[idx].x = ar / mag;
    inout[idx].y = ai / mag;
}

// ifft後の実数配列を1/(W*H)でスケールしつつ中心シフトして出力
__global__ void scale_and_shift(const float* src, float* dst, int width, int height, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int sx = (x + width / 2) % width;
    int sy = (y + height / 2) % height;
    dst[y * width + x] = src[sy * width + sx] * scale;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <input.pgm> [output_rot.pgm=rotated.pgm] [angle_deg=30] [iters=1]\n",
                     argv[0]);
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = (argc >= 3) ? argv[2] : "rotated.pgm";
    float angle_deg = (argc >= 4) ? std::stof(argv[3]) : 30.0f;
    float angle_rad = angle_deg * 3.1415926535f / 180.0f;
    int iters = (argc >= 5) ? std::stoi(argv[4]) : 1;
    if (iters < 1) iters = 1;
    
    // 非同期用ストリーム（cudaStreamCreate: 同一ストリーム内は順序保証。別ストリームを使えば転送と計算を重ねられる）
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // input img
    Image img;
    if (!read_pgm(in_path, img)) return 1;
    PFMImage pfm_in;
    std::string pfm_path = "out.pgm_fft_opencv2_0.pfm";
    if (pfm_path.empty() || !read_pfm(pfm_path, pfm_in)) {
        std::fprintf(stderr, "Failed to read PFM input (%s)\n", pfm_path.c_str());
        return 1;
    }
    int half_cols_candidate = img.width / 2 + 1;
    bool width_ok = (pfm_in.width == img.width) || (pfm_in.width == half_cols_candidate);
    if (!width_ok || pfm_in.height != img.height ||
        (pfm_in.channels != 1 && pfm_in.channels != 2 && pfm_in.channels != 3)) {
        std::fprintf(stderr, "PFM input must be W,H=(%d,%d) or W/2+1,H with channels=1/2/3 (real[/imag])\n",
                     img.width, img.height);
        return 1;
    }
    
    // output fft
    const int fft_w = img.width;
    const int fft_h = img.height;
    // size_t fft_full_elems = static_cast<size_t>(fft_w) * fft_h;
    // std::vector<float> fft_host(fft_full_elems);
    PFMImage corr_pfm;
    corr_pfm.width = img.width;
    corr_pfm.height = img.height;
    corr_pfm.channels = 1;
    corr_pfm.data.resize(static_cast<size_t>(img.width) * img.height);

    // 入力/出力のホストバッファをピン留めしてH2D/D2H転送を高速化
    // ※std::vectorをresizeして再確保させるとポインタが変わり登録が無効になるので、サイズ固定で使い回す。
    CHECK_CUDA(cudaHostRegister(img.data.data(), img.data.size(), cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(corr_pfm.data.data(), corr_pfm.data.size() * sizeof(float), cudaHostRegisterDefault));

    // define cuda woking memory for rotate & sobel
    unsigned char *d_src = nullptr, *d_dst = nullptr;
    float  *d_mag_f = nullptr, *d_sobel_f = nullptr; 
    float* d_pfm_f = nullptr;   // FFT2入力用
    float* d_fft_half = nullptr,*d_fft_full = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, img.data.size()));
    CHECK_CUDA(cudaMalloc(&d_dst, img.data.size()));
    CHECK_CUDA(cudaMalloc(&d_mag_f  , img.data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sobel_f, img.data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pfm_f, img.data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fft_half, img.data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fft_full, img.data.size() * sizeof(float)));

    // define FFFT woking memory & cuFFT plane（R2C 半分出力）とフル複素へ展開
    // cuFFTプランとバッファ（R2C）。出力サイズは height * (width/2 + 1) のcomplex。
    cufftHandle fft_plan;
    const int fft_out_cols = fft_w / 2 + 1;
    size_t fft_elems = static_cast<size_t>(fft_h) * fft_out_cols;
    cufftComplex* d_fft1 = nullptr;
    cufftComplex* d_fft2 = nullptr;
    cufftComplex* d_fft_p = nullptr; // multiply結果
    CHECK_CUDA(cudaMalloc(&d_fft1, fft_elems * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_fft2, fft_elems * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_fft_p, fft_elems * sizeof(cufftComplex)));
    if (cufftPlan2d(&fft_plan, fft_h, fft_w, CUFFT_R2C) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlan2d failed\n");
        return 1;
    }

    cufftSetStream(fft_plan, stream);
    // 逆FFT用プラン（C2R）
    cufftHandle ifft_plan;
    if (cufftPlan2d(&ifft_plan, fft_h, fft_w, CUFFT_C2R) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlan2d (C2R) failed\n");
        return 1;
    }
    cufftSetStream(ifft_plan, stream);

    // CUDAイベントでカーネル時間を個別計測（cudaEventRecord: 指定ストリーム上のタイムスタンプを記録）
    cudaEvent_t ev_rot_start, ev_rot_end, ev_sobel_start, ev_sobel_end;
    CHECK_CUDA(cudaEventCreate(&ev_rot_start));
    CHECK_CUDA(cudaEventCreate(&ev_rot_end));
    CHECK_CUDA(cudaEventCreate(&ev_sobel_start));
    CHECK_CUDA(cudaEventCreate(&ev_sobel_end));

    double total_ms = 0.0;
    float total_rot_ms = 0.0f, total_sobel_ms = 0.0f;

    // FFT2（PFM入力）を1回だけ前計算:
    // OpenCV DFTはフル複素幅=img.widthを出すが、cuFFT R2Cは幅=img.width/2+1にパックされる。
    // PFMがフル幅なら左半分を切り出し、既に半分幅ならそのまま詰める。
    {
        const int pfm_cols = pfm_in.width;
        const int copy_cols = (pfm_cols == img.width) ? fft_out_cols : pfm_cols;
        if (copy_cols < fft_out_cols) {
            std::fprintf(stderr, "PFM width too small: %d (need at least %d)\n", pfm_cols, fft_out_cols);
            return 1;
        }
        std::vector<cufftComplex> fft2_host(fft_elems);
        for (int y = 0; y < fft_h; ++y) {
            for (int x = 0; x < fft_out_cols; ++x) {
                size_t src_idx = static_cast<size_t>(fft_h-y-1) * pfm_cols + x; // 左半分を参照
                size_t dst_idx = static_cast<size_t>(y) * fft_out_cols + x; // R2C半分
                float re = 0.0f, im = 0.0f;
                if (pfm_in.channels == 1) {
                    re = pfm_in.data[src_idx];
                } else {
                    re = pfm_in.data[src_idx * pfm_in.channels + 0];
                    im = pfm_in.data[src_idx * pfm_in.channels + 1];
                }
                fft2_host[dst_idx].x = re;
                fft2_host[dst_idx].y = im;
            }
        }
        CHECK_CUDA(cudaMemcpy(d_fft2, fft2_host.data(), fft_elems * sizeof(cufftComplex),cudaMemcpyHostToDevice));
    }

    // 初回はコンテキスト起動やJITで遅くなりがち。ループで回すと2回目以降は速くなる（ウォームアップ効果）。
    for (int iter = 0; iter < iters; ++iter) {
        auto t_start = std::chrono::steady_clock::now();

        // cudaMemcpyAsync: 非同期転送。ここでは同一ストリームで順序づけているので転送完了後にカーネルが走る。
        CHECK_CUDA(cudaMemcpyAsync(d_src, img.data.data(), img.data.size(),cudaMemcpyHostToDevice, stream));

        dim3 block(16, 16);
        dim3 grid((img.width + block.x - 1) / block.x, (img.height + block.y - 1) / block.y);
        CHECK_CUDA(cudaEventRecord(ev_rot_start, stream));
        rotate_origin<<<grid, block, 0, stream>>>(d_src, d_dst, img.width, img.height, angle_rad);
        CHECK_CUDA(cudaEventRecord(ev_rot_end, stream));

        // Sobel on the rotated image (d_dst) — 常に実行
        dim3 block2(16, 16);
        dim3 grid2((img.width + block2.x - 1) / block2.x,
                   (img.height + block2.y - 1) / block2.y);
        CHECK_CUDA(cudaEventRecord(ev_sobel_start, stream));
        sobel3x3_mag<<<grid2, block2, 0, stream>>>(d_dst, d_mag_f, img.width, img.height);
        CHECK_CUDA(cudaEventRecord(ev_sobel_end, stream));

        // Sobel結果(u8)をFFT入力用にfloatへ変換し、ハン窓を適用してからFFT（R2C）
        u8_to_float_window<<<grid2, block2, 0, stream>>>(d_mag_f, d_sobel_f, img.width, img.height);
        // FFT実行（R2C 半分出力）
        if (cufftExecR2C(fft_plan, reinterpret_cast<cufftReal*>(d_sobel_f), d_fft1) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecR2C failed\n");
            return 1;
        }
        
        // P = FFT1 * conj(FFT2)
        int threads = 256;
        int blocks = (fft_elems + threads - 1) / threads;
        complex_mul_conj<<<blocks, threads, 0, stream>>>(d_fft1, d_fft2, d_fft_p, static_cast<int>(fft_elems));
        normalize_phase<<<blocks, threads, 0, stream>>>(d_fft_p, static_cast<int>(fft_elems), 1e-8f);
        // IFFTで相関ピークを得る
        if (cufftExecC2R(ifft_plan, d_fft_p, reinterpret_cast<cufftReal*>(d_sobel_f)) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2R failed\n");
            return 1;
        }

        // スケール＆シフト（DC中心）
        float inv_scale = 1.0f / (img.width * img.height);
        scale_and_shift<<<grid2, block2, 0, stream>>>(d_sobel_f, d_pfm_f, img.width, img.height, inv_scale);
        // 相関出力をPFMに保存（ループ最後にホストへコピー）

        // 半分出力をフル複素に展開（OpenCV DFT互換のサイズ）※処理時間に含める
        // dim3 block3(16, 16);
        // dim3 grid3((img.width + block3.x - 1) / block3.x, (img.height + block3.y - 1) / block3.y);
        // unpack_half_to_full<<<grid3, block3, 0, stream>>>(d_fft_half, d_fft_full, img.width, img.height);

        // 出力転送も非同期。ストリーム上で順序づけされているのでカーネル完了後に転送される。
        CHECK_CUDA(cudaMemcpyAsync(corr_pfm.data.data(), d_pfm_f, corr_pfm.data.size()  * sizeof(float), cudaMemcpyDeviceToHost,stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float rot_ms = 0.0f, sobel_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&rot_ms, ev_rot_start, ev_rot_end));
        CHECK_CUDA(cudaEventElapsedTime(&sobel_ms, ev_sobel_start, ev_sobel_end));
        auto t_end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        total_ms += elapsed_ms;
        total_rot_ms += rot_ms;
        total_sobel_ms += sobel_ms;

        std::printf("[iter %d/%d] 128-157行相当の処理時間: %.3f ms (rotate kernel: %.3f ms, sobel kernel: %.3f ms)\n",
                    iter + 1, iters, elapsed_ms, rot_ms, sobel_ms);
    }

    double avg_ms = total_ms / iters;
    float avg_rot_ms = total_rot_ms / iters;
    float avg_sobel_ms = total_sobel_ms / iters;
    std::printf("平均: %.3f ms (rotate kernel: %.3f ms, sobel kernel: %.3f ms) over %d iters\n",
                avg_ms, avg_rot_ms, avg_sobel_ms, iters);

    //Sobel結果をホストへコピー
    PFMImage sobel_out;
    sobel_out.width    = img.width;
    sobel_out.height   = img.height;
    sobel_out.channels = 1;
    sobel_out.data.resize(static_cast<size_t>(img.width) * img.height);
    CHECK_CUDA(cudaMemcpy(sobel_out.data.data(), d_mag_f, sobel_out.data.size()*sizeof(float),cudaMemcpyDeviceToHost));
    
    // std::string fft_out_path = out_path + "_fft_cuda.pfm";
    // if (!write_pfm(fft_out_path, fft_pfm)) return 1;
    // std::printf("Wrote FFT (complex, OpenCV DFT-like, PFM) to %s\n", fft_out_path.c_str());

    std::string sobel_out_path = out_path + "_sobel_cuda.pfm";
    if (!write_pfm(sobel_out_path, sobel_out)) return 1;
    std::printf("Wrote Sobel magnitude to %s\n", sobel_out_path.c_str());

    // 相関出力（空間ドメイン）をPFMで保存（正規化なし）
    // PFMImage corr_pfm;
    // corr_pfm.width = img.width;
    // corr_pfm.height = img.height;
    // corr_pfm.channels = 1;
    // corr_pfm.data.resize(static_cast<size_t>(img.width) * img.height);
    // CHECK_CUDA(cudaMemcpy(corr_pfm.data.data(), d_pfm_f, corr_pfm.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::string corr_out_path = out_path + "_pcorr.pfm";
    if (!write_pfm(corr_out_path, corr_pfm)) return 1;
    std::printf("Wrote phase correlation (PFM) to %s\n", corr_out_path.c_str());

    cudaHostUnregister(img.data.data());
    cudaHostUnregister(sobel_out.data.data());
    cudaEventDestroy(ev_rot_start);
    cudaEventDestroy(ev_rot_end);
    cudaEventDestroy(ev_sobel_start);
    cudaEventDestroy(ev_sobel_end);
    cudaStreamDestroy(stream);
    cufftDestroy(fft_plan);

    cudaFree(d_fft_full);
    cudaFree(d_fft_half);
    cudaFree(d_sobel_f);
    cudaFree(d_mag_f);
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
