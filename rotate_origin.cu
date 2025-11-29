#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <input.pgm> [output.pgm=rotated.pgm] [angle_deg=30]\n",
                     argv[0]);
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = (argc >= 3) ? argv[2] : "rotated.pgm";
    float angle_deg = (argc >= 4) ? std::stof(argv[3]) : 30.0f;
    float angle_rad = angle_deg * 3.1415926535f / 180.0f;

    Image img;
    if (!read_pgm(in_path, img)) return 1;
    Image out{img.width, img.height, std::vector<unsigned char>(img.width * img.height)};

    unsigned char *d_src = nullptr, *d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, img.data.size()));
    CHECK_CUDA(cudaMalloc(&d_dst, out.data.size()));
    CHECK_CUDA(cudaMemcpy(d_src, img.data.data(), img.data.size(), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((img.width + block.x - 1) / block.x, (img.height + block.y - 1) / block.y);
    rotate_origin<<<grid, block>>>(d_src, d_dst, img.width, img.height, angle_rad);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out.data.data(), d_dst, out.data.size(), cudaMemcpyDeviceToHost));

    cudaFree(d_src);
    cudaFree(d_dst);

    if (!write_pgm(out_path, out)) return 1;
    std::printf("Wrote rotated image to %s (angle %.2f deg)\n", out_path.c_str(), angle_deg);
    return 0;
}
