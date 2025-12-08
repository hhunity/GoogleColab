%%writefile cuda_phase.cu
// 入力PFM 2枚をタイル分割して Phase Correlation を計算するサンプル
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cufft.h>
#include "pfm_io.h"
#include "pgm_io.h"
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

// 元画像（全体）からタイルごとにバッチバッファへ詰める
__global__ void pack_tiles(const float* src, float* dst,
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
    dst[dst_idx] = src[src_idx];
}

// マスク付き: 有効タイルリストを使って詰める（実数）
__global__ void pack_tiles_masked(const float* src, float* dst,
                                  const int* tile_indices, int num_tiles,
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
    dst[dst_idx] = src[src_idx];
}

// pack + 実→複素を一度に行う
__global__ void pack_tiles_to_complex(const float* src, cufftComplex* dst,
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
    dst[dst_idx].x = src[src_idx];
    dst[dst_idx].y = 0.0f;
}

// マスク付き: 有効タイルリストを使って複素へ詰める
__global__ void pack_tiles_to_complex_masked(const float* src, const int* tile_indices, int num_tiles,
                                             cufftComplex* dst,
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
    dst[dst_idx].x = src[src_idx];
    dst[dst_idx].y = 0.0f;
}

// block_peak->reduce + centroid を一つのカーネルにまとめる
__global__ void final_peak_and_centroid(const Peak* block_peaks, int block_count,
                                        const float* corr, int width, int height,
                                        Peak* out_peak, Centroid* out_centroid) {
    extern __shared__ float s_mem[];
    float* s_val = s_mem;
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);
    int tid = threadIdx.x;

    // 複数blockの代表値から最大を探す（ストライド付きで走査）
    float best_val = -1e30f;
    int best_idx = 0;
    for (int i = tid; i < block_count; i += blockDim.x) {
        float v = block_peaks[i].val;
        if (v > best_val) {
            best_val = v;
            best_idx = block_peaks[i].idx;
        }
    }
    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    // block内リダクションで全体最大を決定
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_val[tid + offset] > s_val[tid]) {
                s_val[tid] = s_val[tid + offset];
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_peak->val = s_val[0];
        out_peak->idx = s_idx[0];
        int peak_idx = s_idx[0];
        int px = peak_idx % width;
        int py = peak_idx / width;
        float m00 = 0.0f, m10 = 0.0f, m01 = 0.0f;
        // 5x5周辺で重心計算
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

// d_corr を介さず、センタシフト＋ピーク＋重心まで一括で行う
__global__ void block_peak_shifted(const cufftComplex* src, int width, int height,
                                   size_t tile_offset, Peak* block_out) {
    extern __shared__ float s_mem[];
    float* s_val = s_mem;
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = -1e30f;
    int id = idx;
    if (idx < width * height) {
        int x = idx % width;
        int y = idx / width;
        int sx = (x + width / 2) % width;
        int sy = (y + height / 2) % height;
        size_t pos = tile_offset + static_cast<size_t>(sy) * width + sx;
        v = src[pos].x; // IFFT出力の実部
    }
    s_val[tid] = v;
    s_idx[tid] = id;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_val[tid + offset] > s_val[tid]) {
                s_val[tid] = s_val[tid + offset];
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_out[blockIdx.x].val = s_val[0];
        block_out[blockIdx.x].idx = s_idx[0];
    }
}

__global__ void final_peak_and_centroid_shifted(const Peak* block_peaks, int block_count,
                                                const cufftComplex* fft, int width, int height,
                                                size_t tile_offset, float scale,
                                                Peak* out_peak, Centroid* out_centroid) {
    extern __shared__ float s_mem[];
    float* s_val = s_mem;
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);
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
    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_val[tid + offset] > s_val[tid]) {
                s_val[tid] = s_val[tid + offset];
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_peak->val = s_val[0];
        out_peak->idx = s_idx[0];
        int peak_idx = s_idx[0];
        int px = peak_idx % width;
        int py = peak_idx / width;
        float m00 = 0.0f, m10 = 0.0f, m01 = 0.0f;
        for (int dy = -2; dy <= 2; ++dy) {
            int y = py + dy;
            if (y < 0 || y >= height) continue;
            for (int dx = -2; dx <= 2; ++dx) {
                int x = px + dx;
                if (x < 0 || x >= width) continue;
                int sx = (x + width / 2) % width;
                int sy = (y + height / 2) % height;
                size_t pos = tile_offset + static_cast<size_t>(sy) * width + sx;
                float v = fft[pos].x * scale;
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

//#define USE_FUSED_CORRLESS_PEAK
//#define USE_TILE_MASK
int main(int argc, char** argv) {
    // 使い方: ./cuda_phase_bach [split_x=1] [split_y=1]
    int split_x = (argc >= 2) ? std::stoi(argv[1]) : 1;
    int split_y = (argc >= 3) ? std::stoi(argv[2]) : 1;
    int iter    = (argc >= 4) ? std::stoi(argv[3]) : 1;
    int debug   = (argc >= 5) ? std::stoi(argv[4]) : 0;
    if (split_x < 1) split_x = 1;
    if (split_y < 1) split_y = 1;

    cudaStream_t streams[2];
    cudaStream_t h2d_stream;
    CHECK_CUDA(cudaStreamCreate(&streams[0]));
    CHECK_CUDA(cudaStreamCreate(&streams[1]));
    CHECK_CUDA(cudaStreamCreate(&h2d_stream));

    // input img (PFM top-down)
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
    CHECK_CUDA(cudaHostRegister(img1.data.data(),
                                img1.data.size() * sizeof(unsigned char),
                                cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(img2.data.data(),
                                img2.data.size() * sizeof(unsigned char),
                                cudaHostRegisterDefault));

    const int tile_w = img1.width / split_x;
    const int tile_h = img1.height / split_y;
    const int tile_pixels = tile_w * tile_h;
    std::vector<int> active_tiles;
    active_tiles.reserve(split_x * split_y);

#ifdef USE_TILE_MASK
    // ここで使用するタイルをマスクで指定（true が処理対象）
    auto use_tile = [](int tx, int ty) {
        static const bool mask[8][3] = {
            {true, true,false,},
            {true, true,false,},
            {true, true,false,},
            {true, true,false,},
            {true, true,false,},
            {true, true,false,},
            {true, true,false,},
            {true, true,false,}
        };
        if (ty < 3 && tx < 8) return mask[ty][tx];
        return false;
    };
    for (int ty = 0; ty < split_y; ++ty) {
        for (int tx = 0; tx < split_x; ++tx) {
            if (use_tile(tx, ty)) {
                active_tiles.push_back(ty * split_x + tx);
            }
        }
    }
#else
    for (int ty = 0; ty < split_y; ++ty) {
        for (int tx = 0; tx < split_x; ++tx) {
            active_tiles.push_back(ty * split_x + tx);
        }
    }
#endif

    const int batch = static_cast<int>(active_tiles.size());
    if (batch == 0) {
        std::fprintf(stderr, "No active tiles selected\n");
        return 1;
    }

    // デバイスバッファ（タイル全体分）
    unsigned char* d_img1_full[2] = {nullptr, nullptr};
    unsigned char* d_img2_full[2] = {nullptr, nullptr};
    unsigned char* d_sobel_out[2]     = {nullptr, nullptr};;
    float* d_mag_f[2]   = {nullptr, nullptr};;
    float* d_sobel_f[2] = {nullptr, nullptr};;
    float* d_img1[2] = {nullptr, nullptr}; // デバッグ用にバッチ連続の実数タイルを保持
    float* d_corr[2] = {nullptr, nullptr};
    Peak* d_block_peaks[2] = {nullptr, nullptr};
    Peak* d_final_peaks[2] = {nullptr, nullptr};
    Centroid* d_centroids[2] = {nullptr, nullptr};
    int peak_threads = 256;
    int peak_blocks = (tile_pixels + peak_threads - 1) / peak_threads;
    size_t batch_pixels = static_cast<size_t>(tile_pixels) * batch;
    for (int b = 0; b < 2; ++b) {
        CHECK_CUDA(cudaMalloc(&d_img1_full[b], static_cast<size_t>(img1.width) * img1.height * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&d_img2_full[b], static_cast<size_t>(img2.width) * img2.height * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&d_sobel_out[b], static_cast<size_t>(img1.width) * img1.height * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&d_mag_f[b], static_cast<size_t>(img1.width) * img1.height * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sobel_f[b], static_cast<size_t>(img1.width) * img1.height * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_img1[b], batch_pixels * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_corr[b], batch_pixels * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_block_peaks[b], peak_blocks * sizeof(Peak)));
        CHECK_CUDA(cudaMalloc(&d_final_peaks[b], static_cast<size_t>(batch) * sizeof(Peak)));
        CHECK_CUDA(cudaMalloc(&d_centroids[b], static_cast<size_t>(batch) * sizeof(Centroid)));
    }
    // cuFFT plan (C2C) batched
    cufftHandle fft_plan[2], ifft_plan[2];
    // size_t fft_elems_full = static_cast<size_t>(tile_w) * tile_h;
    cufftComplex* d_fft1[2] = {nullptr, nullptr};
    cufftComplex* d_fft2[2] = {nullptr, nullptr};
    cufftComplex* d_fft_p[2] = {nullptr, nullptr};
    for (int b = 0; b < 2; ++b) {
        CHECK_CUDA(cudaMalloc(&d_fft1[b], batch_pixels * sizeof(cufftComplex)));
        CHECK_CUDA(cudaMalloc(&d_fft2[b], batch_pixels * sizeof(cufftComplex)));
        CHECK_CUDA(cudaMalloc(&d_fft_p[b], batch_pixels * sizeof(cufftComplex)));
    }
#ifdef USE_TILE_MASK
    int* d_active_tiles = nullptr;
    CHECK_CUDA(cudaMalloc(&d_active_tiles, static_cast<size_t>(batch) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_active_tiles, active_tiles.data(), static_cast<size_t>(batch) * sizeof(int), cudaMemcpyHostToDevice));
#endif

    int n[2] = {tile_h, tile_w};
    int inembed[2] = {tile_h, tile_w};
    int onembed[2] = {tile_h, tile_w};
    int istride = 1, ostride = 1;
    int idist = tile_w * tile_h;
    int odist = tile_w * tile_h;

    void* d_cufft_work[2] = {nullptr, nullptr};
    for (int b = 0; b < 2; ++b) {
        size_t work_size_fwd = 0, work_size_inv = 0;
        cufftCreate(&fft_plan[b]);
        cufftCreate(&ifft_plan[b]);
        cufftSetAutoAllocation(fft_plan[b], 0);
        cufftSetAutoAllocation(ifft_plan[b], 0);
        if (cufftMakePlanMany(fft_plan[b], 2, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, batch, &work_size_fwd) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftMakePlanMany failed\n");
            return 1;
        }
        if (cufftMakePlanMany(ifft_plan[b], 2, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, batch, &work_size_inv) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftMakePlanMany (C2C inverse) failed\n");
            return 1;
        }
        size_t work_size = std::max(work_size_fwd, work_size_inv);
        if (work_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_cufft_work[b], work_size));
            cufftSetWorkArea(fft_plan[b], d_cufft_work[b]);
            cufftSetWorkArea(ifft_plan[b], d_cufft_work[b]);
        }
        cufftSetStream(fft_plan[b], streams[b]);
        cufftSetStream(ifft_plan[b], streams[b]);
    }

    dim3 block(16, 16);
    dim3 grid((tile_w + block.x - 1) / block.x, (tile_h + block.y - 1) / block.y);
    dim3 grid3(grid.x, grid.y, batch);
    dim3 block2(16, 16);
    dim3 grid2((img1.width + block2.x - 1) / block2.x,
                (img1.height + block2.y - 1) / block2.y);
    Peak* peaks_host[2] = {nullptr, nullptr};
    Centroid* cent_host[2] = {nullptr, nullptr};
    CHECK_CUDA(cudaHostAlloc(&peaks_host[0], static_cast<size_t>(batch) * sizeof(Peak), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&peaks_host[1], static_cast<size_t>(batch) * sizeof(Peak), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&cent_host[0], static_cast<size_t>(batch) * sizeof(Centroid), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&cent_host[1], static_cast<size_t>(batch) * sizeof(Centroid), cudaHostAllocDefault));

    cudaEvent_t h2d_ready[2];
    CHECK_CUDA(cudaEventCreate(&h2d_ready[0]));
    CHECK_CUDA(cudaEventCreate(&h2d_ready[1]));

    for (int i = 0; i < 2; i++) {
        cudaStream_t stream = streams[i];
        CHECK_CUDA(cudaMemcpyAsync(d_img2_full[i], img2.data.data(),
                                   static_cast<size_t>(img2.width) * img2.height * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice, stream));
        rotate_origin<<<grid2, block2, 0, stream>>>(d_img2_full[i], d_sobel_out[i], img1.width, img1.height, 0.0);
        sobel3x3_mag<<<grid2, block2, 0, stream>>>(d_sobel_out[i], d_mag_f[i], img1.width, img1.height);
        float_hann_window<<<grid2, block2, 0, stream>>>(d_mag_f[i], d_sobel_f[i], img1.width, img1.height);
#ifdef USE_TILE_MASK
        pack_tiles_to_complex_masked<<<grid3, block, 0, stream>>>(d_img2_full[i], d_active_tiles, batch,
                                                                  d_fft2[i],
                                                                  img2.width, img2.height,
                                                                  tile_w, tile_h, split_x, split_y);
#else
        pack_tiles_to_complex<<<grid3, block, 0, stream>>>(d_sobel_f[i], d_fft2[i],
                                                           img2.width, img2.height,
                                                           tile_w, tile_h, split_x, split_y);
        if(1) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            // save_pfm_real("cuda_rotate_out.pfm", d_sobel_out,img1.width, img1.height);
            save_pfm_real("b_cuda_sobel_out.pfm", d_mag_f[i],img1.width, img1.height);
            save_pfm_real("b_cuda_hann_window_out.pfm", d_sobel_f[i], img1.width, img1.height);
            save_pfm("b_cuda_tiles_out.pfm", d_fft2[i], tile_w, tile_h*batch,1);
        }
#endif
        if (cufftExecC2C(fft_plan[i], d_fft2[i], d_fft2[i], CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2C forward failed\n");
            return 1;
        }
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    auto t_all_start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < iter; i++) {
        int buf = i & 1;
        cudaStream_t stream = streams[buf];
        // H2D は専用ストリームで前倒しし、イベントで待つ
        CHECK_CUDA(cudaMemcpyAsync(d_img1_full[buf], img1.data.data(),
                                   static_cast<size_t>(img1.width) * img1.height * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice, stream));
        // //このストリーム上のここまでで、イベントが完了しtら転送完了フラグを立てる
        // CHECK_CUDA(cudaEventRecord(h2d_ready[buf], h2d_stream));
        // //転送完了待ち
        // CHECK_CUDA(cudaStreamWaitEvent(stream, h2d_ready[buf], 0));

#ifdef USE_TILE_MASK
        pack_tiles_to_complex_masked<<<grid3, block, 0, stream>>>(d_img1_full[buf], d_active_tiles, batch,
                                                                  d_fft1[buf],
                                                                  img1.width, img1.height,
                                                                  tile_w, tile_h, split_x, split_y);
#else
        rotate_origin<<<grid2, block2, 0, stream>>>(d_img1_full[buf], d_sobel_out[buf], img1.width, img1.height, 0.0);
        sobel3x3_mag<<<grid2, block2, 0, stream>>>(d_sobel_out[buf], d_mag_f[buf], img1.width, img1.height);
        float_hann_window<<<grid2, block2, 0, stream>>>(d_mag_f[buf], d_sobel_f[buf], img1.width, img1.height);
        pack_tiles_to_complex<<<grid3, block, 0, stream>>>(d_sobel_f[buf], d_fft1[buf],
                                                           img1.width, img1.height,
                                                           tile_w, tile_h, split_x, split_y);
        if(1) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            // save_pfm_real("cuda_rotate_out.pfm", d_sobel_out,img1.width, img1.height);
            save_pfm_real("cuda_sobel_out.pfm", d_mag_f[buf],img1.width, img1.height);
            save_pfm_real("cuda_hann_window_out.pfm", d_sobel_f[buf], img1.width, img1.height);
            save_pfm("cuda_tiles_out.pfm", d_fft1[buf], tile_w, tile_h*batch,1);
        }
#endif
        if (cufftExecC2C(fft_plan[buf], d_fft1[buf], d_fft1[buf], CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2C forward failed\n");
            return 1;
        }

        int threads = 256;
        int blocks = (static_cast<int>(batch_pixels) + threads - 1) / threads;
        complex_mul_conj_normalize<<<blocks, threads, 0, stream>>>(d_fft1[buf], d_fft2[buf], d_fft_p[buf],
                                                                   static_cast<int>(batch_pixels), 1e-16f);
        if (cufftExecC2C(ifft_plan[buf], d_fft_p[buf], d_fft_p[buf], CUFFT_INVERSE) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2C inverse failed\n");
            return 1;
        }

        float inv_scale = 1.0f;
#ifdef USE_FUSED_CORRLESS_PEAK
        size_t shared_bytes = peak_threads * (sizeof(float) + sizeof(int));
        for (int b = 0; b < batch; ++b) {
            size_t tile_offset = static_cast<size_t>(b) * tile_pixels;
            block_peak_shifted<<<peak_blocks, peak_threads, shared_bytes, stream>>>(d_fft_p[buf], tile_w, tile_h,
                                                                                    tile_offset, d_block_peaks[buf]);
            final_peak_and_centroid_shifted<<<1, peak_threads, shared_bytes, stream>>>(d_block_peaks[buf], peak_blocks,
                                                                                       d_fft_p[buf], tile_w, tile_h,
                                                                                       tile_offset, inv_scale,
                                                                                       d_final_peaks[buf] + b, d_centroids[buf] + b);
        }
#else
        size_t shared_bytes = peak_threads * (sizeof(float) + sizeof(int));
        complex_real_scale_shift_batch<<<grid3, block, 0, stream>>>(d_fft_p[buf], d_corr[buf], tile_w, tile_h, batch, inv_scale);
        for (int b = 0; b < batch; ++b) {
            const float* corr_tile = d_corr[buf] + static_cast<size_t>(b) * tile_pixels;
            block_peak<<<peak_blocks, peak_threads, shared_bytes, stream>>>(corr_tile, tile_pixels, d_block_peaks[buf]);
            final_peak_and_centroid<<<1, peak_threads, shared_bytes, stream>>>(d_block_peaks[buf], peak_blocks,
                                                                               corr_tile, tile_w, tile_h,
                                                                               d_final_peaks[buf] + b, d_centroids[buf] + b);
        }
#endif

        int prev = buf ^ 1;
        if (i > 0) {
            CHECK_CUDA(cudaMemcpyAsync(peaks_host[prev], d_final_peaks[prev], static_cast<size_t>(batch) * sizeof(Peak),
                                   cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaMemcpyAsync(cent_host[prev], d_centroids[prev], static_cast<size_t>(batch) * sizeof(Centroid),
                                   cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(streams[prev]));

            double center_x = static_cast<double>(tile_w) / 2.0;
            double center_y = static_cast<double>(tile_h) / 2.0;
            // for (int b = 0; b < batch; ++b) {
            //     int tile_idx = active_tiles[b];
            //     int tile_tx = tile_idx % split_x;
            //     int tile_ty = tile_idx / split_x;
            //     Peak pk = peaks_host[prev][b];
            //     Centroid ct = cent_host[prev][b];
            //     double peak_x = pk.idx % tile_w;
            //     double peak_y = pk.idx / tile_w;
            //     double t_x = (ct.m00 > 0.0f) ? (ct.m10 / ct.m00) : peak_x;
            //     double t_y = (ct.m00 > 0.0f) ? (ct.m01 / ct.m00) : peak_y;
            //     double shift_x = center_x - t_x;
            //     double shift_y = center_y - t_y;
            //     if (shift_x > center_x) shift_x -= tile_w;
            //     if (shift_y > center_y) shift_y -= tile_h;
            //     double response = ct.m00 / (tile_w * tile_h);
            //     std::printf("tile %d (%d,%d): peak=(%.0f,%.0f) subpix=(%.4f,%.4f) shift=(%.4f,%.4f) response=%.6f\n",
            //                 b, tile_tx, tile_ty,
            //                 peak_x, peak_y, t_x, t_y, shift_x, shift_y, response);
            // }
        }
    }
    
    auto t_all_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_all_end - t_all_start).count();
    std::printf("Total %.3f ms, avg %.3f ms/frame over %d iterations\n", total_ms, total_ms / iter, iter);

    int last = (iter - 1) & 1;
    CHECK_CUDA(cudaStreamSynchronize(streams[last]));
    double center_x = static_cast<double>(tile_w) / 2.0;
    double center_y = static_cast<double>(tile_h) / 2.0;
    for (int b = 0; b < batch; ++b) {
        int tile_idx = active_tiles[b];
        int tile_tx = tile_idx % split_x;
        int tile_ty = tile_idx / split_x;
        Peak pk = peaks_host[last][b];
        Centroid ct = cent_host[last][b];
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
                    b, tile_tx, tile_ty,
                    peak_x, peak_y, t_x, t_y, shift_x, shift_y, response);
    }

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(h2d_stream);
    cudaEventDestroy(h2d_ready[0]);
    cudaEventDestroy(h2d_ready[1]);
    for (int b = 0; b < 2; ++b) {
        cufftDestroy(fft_plan[b]);
        cufftDestroy(ifft_plan[b]);
        cudaFree(d_sobel_out[b]);
        cudaFree(d_mag_f[b]);
        cudaFree(d_sobel_f[b]);
        cudaFree(d_fft1[b]);
        cudaFree(d_fft2[b]);
        cudaFree(d_fft_p[b]);
        cudaFree(d_img1_full[b]);
        cudaFree(d_img2_full[b]);
        cudaFree(d_img1[b]);
        cudaFree(d_corr[b]);
        cudaFree(d_block_peaks[b]);
        cudaFree(d_final_peaks[b]);
        cudaFree(d_centroids[b]);
    }
#ifdef USE_TILE_MASK
    cudaFree(d_active_tiles);
#endif
    cudaFreeHost(peaks_host[0]);
    cudaFreeHost(peaks_host[1]);
    cudaFreeHost(cent_host[0]);
    cudaFreeHost(cent_host[1]);
    if (d_cufft_work[0]) cudaFree(d_cufft_work[0]);
    if (d_cufft_work[1]) cudaFree(d_cufft_work[1]);
    CHECK_CUDA(cudaHostUnregister(img1.data.data()));
    CHECK_CUDA(cudaHostUnregister(img2.data.data()));
    return 0;
}
