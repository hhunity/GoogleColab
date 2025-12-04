%%writefile cuda_kernels.cuh
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cufft.h>   // cuFFTを使用してFFTを計算

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

// 相関画像からブロックごとの最大値とインデックスを求める。
// ざっくり流れ:
//  (1) グローバルメモリ(src)から共有メモリ(s_mem)へ読み込み
//      └ グローバルは遅いが容量大、共有は速いがブロック内専用キャッシュ的なもの
//      └ グローバルメモリ: GPU全体で共有される大きなメモリ。帯域は高いがレイテンシが大きい。
//      └ 共有メモリ: 同じブロックのスレッドでだけ共有できる小さな高速メモリ（SM内キャッシュに近い）。グローバルから読み込んだ値をここに置くことで、そのブロック内のスレッド間で高速に使い回せる。
//      └ 流れ: src[idx] を各スレッドが s_mem[tid] に書く → __syncthreads() で全員が書き終わるのを待つ → 以降の計算は共有メモリ上のデータを使う。
//  (2) __syncthreads でブロック内全スレッドのロード完了を待機
//      └ __syncthreads は「同じブロックの全スレッドがここまで来るまで待つ」バリア
//  (3) 木構造のリダクションで最大値とその位置を見つける
//      └ 配列の最大値（や和など）を並列に求める典型手法。要素数を半分ずつ潰していくので「木構造」。
//      └ 例（block_peak）: 共有メモリに blockDim.x 個の値が入っている状態からスタート。最初はオフセット=blockDim.x/2。
//      └ tid < offset のスレッドだけが「自分」と「自分+offset」を比較し、大きい方を自分に残す。
//      └ オフセットを半分にして繰り返す（blockDim.x/2 → blockDim.x/4 → … → 1）。各ステップの前後で __syncthreads() しないと、他スレッドがまだ更新中の値を読んでしまう。
//      └ 最終的に tid=0 にブロック内の最大値とそのインデックスが残る。
//      └ 半分ずつ潰す: offset=blockDim/2 から 1 まで、tid<offset が (自分 vs 自分+offset) を比較
//      └ 各段の前後で __syncthreads を入れ、他スレッドがまだ書き込み中の値を読まないようにする
//      └ educe_peak も同じパターンで、入力が “ブロック代表の配列” になっているだけ。
//  __syncthreads とは
//      └ 同一ブロック内の全スレッドがこの地点に到達するまで待つバリア。これ以前の共有メモリ書き込みが全員完了してから次に進むための安全装置。
//      └ リダクションの各ステップごとに入れているのは、更新が終わっていないデータを他スレッドが読まないようにするため。
__global__ void block_peak(const float* src, int n, Peak* block_out) {
    extern __shared__ float s_mem[];
    float* s_val = s_mem;
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = -1e30f;
    int id = idx;
    if (idx < n) {
        v = src[idx];
    }
    s_val[tid] = v; // 共有メモリに値と元インデックスを詰める（速いローカルキャッシュとして使う）
    s_idx[tid] = id;
    __syncthreads(); // 全スレッドのロード完了を同期（これ以前のデータ書き込みが全員済むのを待つ）
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) { // 木構造の最大値リダクション（半分ずつ潰す）
        if (tid < offset) {
            if (s_val[tid + offset] > s_val[tid]) {
                s_val[tid] = s_val[tid + offset];
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads(); // 次段に進む前に全スレッドを揃える（同期しないと別スレッドがまだ更新中かもしれない）
    }
    if (tid == 0) {
        block_out[blockIdx.x].val = s_val[0];
        block_out[blockIdx.x].idx = s_idx[0];
    }
}

// ブロック最大値の配列をさらに縮約して全体の最大値を求める（手順は block_peak と同じ）。
// ここでも共有メモリに一旦集め、__syncthreads でバリアを挟みつつ木構造リダクションで1要素まで畳む。
__global__ void reduce_peak(const Peak* src, int n, Peak* out) {
    extern __shared__ float s_mem[];
    float* s_val = s_mem;
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = -1e30f;
    int id = 0;
    if (idx < n) {
        v = src[idx].val;
        id = src[idx].idx;
    }
    s_val[tid] = v; // 共有メモリに値と元インデックスを詰める
    s_idx[tid] = id;
    __syncthreads(); // __syncthreads は同一ブロックの全スレッドが揃うまで待つ
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) { // 木構造の最大値リダクション
        if (tid < offset) {
            if (s_val[tid + offset] > s_val[tid]) {
                s_val[tid] = s_val[tid + offset];
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads(); // 次段へ進む前に全スレッドを同期
    }
    if (tid == 0) {
        out[blockIdx.x].val = s_val[0];
        out[blockIdx.x].idx = s_idx[0];
    }
}

// 5x5窓で重み付き重心のモーメントを取る（25スレッドだけを使用）
// 各スレッドが1画素を担当し、atomicAdd で m00/m10/m01 を累積。
// m00 = 重み総和, m10 = x*重みの総和, m01 = y*重みの総和 → 最後に m10/m00, m01/m00 で重心。
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