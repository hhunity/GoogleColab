%%writefile cuda_kernels_half.cuh
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cufftXt.h>

struct Peak {
    float val;
    int idx;
};

struct Centroid {
    float m00;
    float m10;
    float m01;
};

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",              \
                         cudaGetErrorString(err__), err__, __FILE__, __LINE__);\
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// タイル内ハン窓（1次元）を事前計算して常駐させる
#define MAX_TILE_DIM 4096
__constant__ float c_hann_x[MAX_TILE_DIM];
__constant__ float c_hann_y[MAX_TILE_DIM];

// u8 → float 窓
__global__ void u8_to_float_window(const unsigned char* src, float* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const float pi = 3.1415926535f;
    float wx = 1.0f, wy = 1.0f;
    if (width > 1)  wx = 0.5f * (1.0f - cosf(2.0f * pi * x / (width - 1)));
    if (height > 1) wy = 0.5f * (1.0f - cosf(2.0f * pi * y / (height - 1)));
    dst[static_cast<size_t>(y) * width + x] = static_cast<float>(src[y * width + x]) * wx * wy;
}

// pack + 実→半精度複素を一度に行う（タイル内で2Dハン窓を適用）
__global__ void pack_tiles_to_half_complex(const float* src, __half2* dst,
                                           int width, int height,
                                           int tile_w, int tile_h,
                                           int split_x, int split_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    int batch = split_x * split_y;
    if (b >= batch || x >= tile_w || y >= tile_h) return;
    int tx = b % split_x;
    int ty = b / split_x;
    int src_x = tx * tile_w + x;
    int src_y = ty * tile_h + y;
    if (src_x >= width || src_y >= height) return;
    size_t src_idx = static_cast<size_t>(src_y) * width + src_x;
    size_t dst_idx = static_cast<size_t>(b) * tile_w * tile_h + static_cast<size_t>(y) * tile_w + x;
    float wx = c_hann_x[x];
    float wy = c_hann_y[y];
    float val = src[src_idx] * wx * wy;
    dst[dst_idx] = __halves2half2(__float2half(val), __float2half(0.0f));
}

__global__ void pack_tiles_to_half_complex_masked(const float* src, const int* tile_indices, int num_tiles,
                                                  __half2* dst,
                                                  int width, int height,
                                                  int tile_w, int tile_h,
                                                  int split_x, int split_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (b >= num_tiles || x >= tile_w || y >= tile_h) return;
    int tile_idx = tile_indices[b];
    int tx = tile_idx % split_x;
    int ty = tile_idx / split_x;
    int src_x = tx * tile_w + x;
    int src_y = ty * tile_h + y;
    if (src_x >= width || src_y >= height) return;
    size_t src_idx = static_cast<size_t>(src_y) * width + src_x;
    size_t dst_idx = static_cast<size_t>(b) * tile_w * tile_h + static_cast<size_t>(y) * tile_w + x;
    float wx = c_hann_x[x];
    float wy = c_hann_y[y];
    float val = src[src_idx] * wx * wy;
    dst[dst_idx] = __halves2half2(__float2half(val), __float2half(0.0f));
}
//#define DEBUG_CMPLX_PRINT
// half複素: out = a * conj(b) / |a*conj(b)| (eps回避)
__global__ void complex_mul_conj_normalize_half(const __half2* a, const __half2* b,
                                                __half2* out, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    __half2 ha = a[idx];
    __half2 hb = b[idx];
    float ar = __half2float(__low2half(ha));
    float ai = __half2float(__high2half(ha));
    float br = __half2float(__low2half(hb));
    float bi = -__half2float(__high2half(hb));
    float xr = ar * br - ai * bi;
    float xi = ar * bi + ai * br;
    float mag = sqrtf(xr * xr + xi * xi);
    if (mag < eps) mag = eps;
    out[idx] = __halves2half2(__float2half(xr / mag), __float2half(xi / mag));
#ifdef DEBUG_CMPLX_PRINT
    if (idx < 8 && blockIdx.x == 0) {
        printf("cmn_half idx=%d ar=%.6f ai=%.6f br=%.6f bi=%.6f xr=%.6f xi=%.6f mag=%.6f\n",
               idx, ar, ai, br, -bi, xr, xi, mag);
    }
#endif
}

// half複素 -> 実(float)で中心シフトしつつスケール
__global__ void complex_real_scale_shift_batch_half(const __half2* src, float* dst,
                                                    int width, int height, int batch,
                                                    float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (x >= width || y >= height || b >= batch) return;
    int sx = (x + width / 2) % width;
    int sy = (y + height / 2) % height;
    size_t base = static_cast<size_t>(b) * width * height;
    size_t idx_src = base + static_cast<size_t>(sy) * width + sx;
    __half2 v = src[idx_src];
    float real = __half2float(__low2half(v));
    dst[base + static_cast<size_t>(y) * width + x] = real * scale;
}

// block_peak と reduce_peak は float 入力を前提（ピーク検出は従来通り）
__global__ void block_peak(const float* src, int n, Peak* block_out) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = (idx < n) ? src[idx] : -1e30f;
    int id = idx;
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float vv = __shfl_down_sync(mask, v, offset);
        int iid = __shfl_down_sync(mask, id, offset);
        if (vv > v) {
            v = vv;
            id = iid;
        }
    }
    __shared__ float warp_val[32];
    __shared__ int warp_idx[32];
    int lane = tid & (warpSize - 1);
    int warp = tid / warpSize;
    if (lane == 0) {
        warp_val[warp] = v;
        warp_idx[warp] = id;
    }
    __syncthreads();
    if (warp == 0) {
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        float val = (lane < warp_count) ? warp_val[lane] : -1e30f;
        int idx2 = (lane < warp_count) ? warp_idx[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float vv = __shfl_down_sync(mask, val, offset);
            int iid = __shfl_down_sync(mask, idx2, offset);
            if (vv > val) {
                val = vv;
                idx2 = iid;
            }
        }
        if (lane == 0) {
            block_out[blockIdx.x].val = val;
            block_out[blockIdx.x].idx = idx2;
        }
    }
}

// half2 -> float2 変換
__global__ void half2_to_float2(const __half2* src, cufftComplex* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    __half2 v = src[idx];
    dst[idx].x = __half2float(__low2half(v));
    dst[idx].y = __half2float(__high2half(v));
}

// float 複素: out = a * conj(b) / |a*conj(b)| (eps 回避)
__global__ void complex_mul_conj_normalize_f(const cufftComplex* a, const cufftComplex* b,
                                             cufftComplex* out, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float ar = a[idx].x, ai = a[idx].y;
    float br = b[idx].x, bi = -b[idx].y;
    float xr = ar * br - ai * bi;
    float xi = ar * bi + ai * br;
    float mag = sqrtf(xr * xr + xi * xi);
    if (mag < eps) mag = eps;
    out[idx].x = xr / mag;
    out[idx].y = xi / mag;
}

// float 複素 -> 実(float)で中心シフトしつつスケール
__global__ void complex_real_scale_shift_batch_f(const cufftComplex* src, float* dst,
                                                 int width, int height, int batch,
                                                 float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (x >= width || y >= height || b >= batch) return;
    int sx = (x + width / 2) % width;
    int sy = (y + height / 2) % height;
    size_t base = static_cast<size_t>(b) * width * height;
    size_t idx_src = base + static_cast<size_t>(sy) * width + sx;
    const cufftComplex v = src[idx_src];
    dst[base + static_cast<size_t>(y) * width + x] = v.x * scale;
}

__global__ void reduce_peak(const Peak* src, int n, Peak* out) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = -1e30f;
    int id = 0;
    if (idx < n) {
        v = src[idx].val;
        id = src[idx].idx;
    }
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float vv = __shfl_down_sync(mask, v, offset);
        int iid = __shfl_down_sync(mask, id, offset);
        if (vv > v) {
            v = vv;
            id = iid;
        }
    }
    __shared__ float warp_val[32];
    __shared__ int warp_idx[32];
    int lane = tid & (warpSize - 1);
    int warp = tid / warpSize;
    if (lane == 0) {
        warp_val[warp] = v;
        warp_idx[warp] = id;
    }
    __syncthreads();
    if (warp == 0) {
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        float val = (lane < warp_count) ? warp_val[lane] : -1e30f;
        int idx2 = (lane < warp_count) ? warp_idx[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float vv = __shfl_down_sync(mask, val, offset);
            int iid = __shfl_down_sync(mask, idx2, offset);
            if (vv > val) {
                val = vv;
                idx2 = iid;
            }
        }
        if (lane == 0) {
            out[blockIdx.x].val = val;
            out[blockIdx.x].idx = idx2;
        }
    }
}

__global__ void centroid5x5(const float* src, int width, int height, const Peak* peak, Centroid* out) {
    int tid = threadIdx.x;
    if (tid >= 25) return;
    int peak_idx = peak->idx;
    int px = peak_idx % width;
    int py = peak_idx / width;
    int dx = tid % 5 - 2;
    int dy = tid / 5 - 2;
    int x = px + dx;
    int y = py + dy;
    float v = 0.0f;
    if (x >= 0 && x < width && y >= 0 && y < height) {
        v = src[static_cast<size_t>(y) * width + x];
    }
    atomicAdd(&out->m00, v);
    atomicAdd(&out->m10, v * static_cast<float>(x));
    atomicAdd(&out->m01, v * static_cast<float>(y));
}

// block_peak->reduce + centroid を一つのカーネルにまとめる
__global__ void final_peak_and_centroid(const Peak* block_peaks, int block_count,
                                        const float* corr, int width, int height,
                                        Peak* out_peak, Centroid* out_centroid) {
    int tid = threadIdx.x;
    float best_val = -1e30f;
    int best_idx = 0;
    for (int i = tid; i < block_count; i += blockDim.x) {
        float v = block_peaks[i].val;
        if (v > best_val) {
            best_val = v;
            best_idx = block_peaks[i].idx;
        }
    }
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float v = __shfl_down_sync(mask, best_val, offset);
        int id = __shfl_down_sync(mask, best_idx, offset);
        if (v > best_val) {
            best_val = v;
            best_idx = id;
        }
    }
    __shared__ float warp_val[32];
    __shared__ int warp_idx[32];
    int lane = tid & (warpSize - 1);
    int warp = tid / warpSize;
    if (lane == 0) {
        warp_val[warp] = best_val;
        warp_idx[warp] = best_idx;
    }
    __syncthreads();
    if (warp == 0) {
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        float v = (lane < warp_count) ? warp_val[lane] : -1e30f;
        int id = (lane < warp_count) ? warp_idx[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float vv = __shfl_down_sync(mask, v, offset);
            int iid = __shfl_down_sync(mask, id, offset);
            if (vv > v) {
                v = vv;
                id = iid;
            }
        }
        if (lane == 0) {
            out_peak->val = v;
            out_peak->idx = id;
        }
    }
    __syncthreads();
    if (tid == 0) {
        int peak_idx = out_peak->idx;
        int px = peak_idx % width;
        int py = peak_idx / width;
        float m00 = 0.0f, m10 = 0.0f, m01 = 0.0f;
        for (int dy = -2; dy <= 2; ++dy) {
            int y = py + dy;
            if (y < 0 || y >= height) continue;
            for (int dx = -2; dx <= 2; ++dx) {
                int x = px + dx;
                if (x < 0 || x >= width) continue;
                float v = corr[static_cast<size_t>(y) * width + x];
                m00 += v;
                m10 += v * static_cast<float>(x);
                m01 += v * static_cast<float>(y);
            }
        }
        out_centroid->m00 = m00;
        out_centroid->m10 = m10;
        out_centroid->m01 = m01;
    }
}
