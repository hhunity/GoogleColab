%%writefile rot_sobel.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include "pgm_io.h"

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
__global__ void sobel3x3_mag(const unsigned char* src, unsigned char* mag_out,
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
    if (mag > 255.0f) mag = 255.0f;
    mag_out[y * width + x] = static_cast<unsigned char>(mag + 0.5f);
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

    Image img;
    if (!read_pgm(in_path, img)) return 1;
    Image out{img.width, img.height, std::vector<unsigned char>(img.width * img.height)};
    Image sobel_out{img.width, img.height, std::vector<unsigned char>(out.data.size())};
    // 入力/出力のホストバッファをピン留めしてH2D/D2H転送を高速化
    // ※std::vectorをresizeして再確保させるとポインタが変わり登録が無効になるので、サイズ固定で使い回す。
    CHECK_CUDA(cudaHostRegister(img.data.data(), img.data.size(), cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(sobel_out.data.data(), sobel_out.data.size(), cudaHostRegisterDefault));

    unsigned char *d_src = nullptr, *d_dst = nullptr, *d_mag = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, img.data.size()));
    CHECK_CUDA(cudaMalloc(&d_dst, out.data.size()));
    // Sobel用の出力バッファも計測前に確保し、計測からメモリアロケーションを除外
    CHECK_CUDA(cudaMalloc(&d_mag, out.data.size()));
    // 非同期用ストリーム（cudaStreamCreate: 同一ストリーム内は順序保証。別ストリームを使えば転送と計算を重ねられる）
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    // ウォームアップ: cudaFree(0)でコンテキストを強制生成（cudaMallocでも起きるが明示しておく）
    CHECK_CUDA(cudaFree(0));

    // CUDAイベントでカーネル時間を個別計測（cudaEventRecord: 指定ストリーム上のタイムスタンプを記録）
    cudaEvent_t ev_rot_start, ev_rot_end, ev_sobel_start, ev_sobel_end;
    CHECK_CUDA(cudaEventCreate(&ev_rot_start));
    CHECK_CUDA(cudaEventCreate(&ev_rot_end));
    CHECK_CUDA(cudaEventCreate(&ev_sobel_start));
    CHECK_CUDA(cudaEventCreate(&ev_sobel_end));

    double total_ms = 0.0;
    float total_rot_ms = 0.0f, total_sobel_ms = 0.0f;

    // 初回はコンテキスト起動やJITで遅くなりがち。ループで回すと2回目以降は速くなる（ウォームアップ効果）。
    for (int iter = 0; iter < iters; ++iter) {
        auto t_start = std::chrono::steady_clock::now();

        // cudaMemcpyAsync: 非同期転送。ここでは同一ストリームで順序づけているので転送完了後にカーネルが走る。
        CHECK_CUDA(cudaMemcpyAsync(d_src, img.data.data(), img.data.size(),
                                   cudaMemcpyHostToDevice, stream));

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
        sobel3x3_mag<<<grid2, block2, 0, stream>>>(d_dst, d_mag, img.width, img.height);
        CHECK_CUDA(cudaEventRecord(ev_sobel_end, stream));

        // 出力転送も非同期。ストリーム上で順序づけされているのでカーネル完了後に転送される。
        CHECK_CUDA(cudaMemcpyAsync(sobel_out.data.data(), d_mag, sobel_out.data.size(),
                                   cudaMemcpyDeviceToHost, stream));
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

    std::string sobel_out_path = out_path + "_sobel.pgm";
    if (!write_pgm(sobel_out_path, sobel_out)) return 1;
    std::printf("Wrote Sobel magnitude to %s\n", sobel_out_path.c_str());

    cudaHostUnregister(img.data.data());
    cudaHostUnregister(sobel_out.data.data());
    cudaEventDestroy(ev_rot_start);
    cudaEventDestroy(ev_rot_end);
    cudaEventDestroy(ev_sobel_start);
    cudaEventDestroy(ev_sobel_end);
    cudaStreamDestroy(stream);

    cudaFree(d_mag);
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
