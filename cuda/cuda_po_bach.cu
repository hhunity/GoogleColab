%%writefile rot_sobel.cu
// GPUリソース概算（幅=W, 高さ=H の場合）
// - デバイスバッファ:
//   d_src/d_dst: W*H バイト (u8)
//   d_mag_f/d_sobel_f/d_pfm_f: W*H*4 バイト (float) × batch
//   d_fft1/d_fft2/d_fft_p: H*(W/2+1)*8 バイト (cufftComplex) × batch
//   d_block_peaks: ceil(W*H/256)*sizeof(Peak)（1要素約8バイト）
//   d_tmp_peaks: ceil(ceil(W*H/256)/256)*sizeof(Peak)
//   d_final_peak: 8バイト, d_centroid: 12バイト
//   （cuFFT内部ワーク領域は cuFFT が管理）
//   例: 256x256 の場合
//     u8: 約 2*64 KB
//     float群: 約 5*256 KB
//     complex群: 約 3*0.26 MB
//     peakバッファ: 数KB → 合計でおおよそ 2 MB 弱 + cuFFTワーク
// - ホスト固定メモリ: 入力 img.data (W*H バイト)
// - 1イテレーションで起動するカーネル（順番）:
//   rotate_origin, sobel3x3_mag, u8_to_float_window,
//   complex_mul_conj, normalize_phase, scale_and_shift,
//   block_peak, reduce_peak(1), reduce_peak(2), centroid5x5
//   + cuFFT呼び出し: cufftExecR2C, cufftExecC2R
//   スレッド・ブロック設定の例:
//     画像系: block=(16,16), grid=(ceil(W/16), ceil(H/16))
//     peak系: block_peak/reduce_peak = 256スレッド、grid=ceil((W*H)/256)（2段目はそのさらにceil/256）
//     centroid5x5 = block=25スレッド, grid=1
//   ワープ/スレッド/グリッドの考え方:
//     warpは32スレッド単位。画像系 block=(16,16)=256スレッド→8ワープ/ブロック。block_peak も同様に256スレッド→8ワープ。
//     gridは「ブロックで画像全体をカバーする数」で、ceil(W/block.x) × ceil(H/block.y)。
//     peak系のgridはピクセル数を256で割ったもの（2段目はさらに256で割る）、centroid5x5は1ブロックのみ。
//   GPUごとの並列度（block=256 の理論上限、実際はレジスタ/共有メモリ使用量で減る点に注意）:
//     T400(推定SM数≈12): 最大 8 block/SM → 約 96 block 同時実行の上限
//     RTX A2000 12GB(SM=26): 最大 8 block/SM → 約 208 block 同時実行の上限
//     RTX A4000(SM=48): 最大 8 block/SM → 約 384 block 同時実行の上限
//     ※1 SMあたりハード上限は 2048 スレッド/SM, 16 block/SM。レジスタや共有メモリの使用量次第でこれより減る。
// - __syncthreads: ブロック内全スレッドが到達するまで待つバリア。
//   リダクション中に入れて、他スレッドがまだ共有メモリを更新中の値を読まないようにする。
// スレッド: 最小の実行単位。
// ワープ: 32スレッドの束。スケジューラはワープ単位で実行。block=(16,16)=256スレッドなら 8ワープ/ブロック。
// ブロック: 同期や共有メモリを共有する単位。グリッド内に多数並ぶ。SM（Streaming Multiprocessor）がブロックを順次受け持つ。
// グリッド: そのカーネル起動で投げられる全ブロック集合。上記では画像系グリッドが 256 ブロック、peak系は 256→1→1 ブロック。
// SM: GPU のコア群。1SM で複数ブロックを同時に実行（上限はブロック内スレッド数や共有メモリ/レジスタ使用量で決まる）。ワープは SM 内でスケジューリングされる。
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cufft.h>   // cuFFTを使用してFFTを計算
#include "pgm_io.h"
#include "pfm_io.h"  // FFT結果をPFMで保存（OpenCV DFT互換のフル複素）
#include "cuda_kernels.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <input.pgm> [output_rot.pgm=rotated.pgm] [angle_deg=30] [iters=1] [split_x=1] [split_y=1]\n",
                     argv[0]);
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = (argc >= 3) ? argv[2] : "rotated.pgm";
    float angle_deg = (argc >= 4) ? std::stof(argv[3]) : 30.0f;
    float angle_rad = angle_deg * 3.1415926535f / 180.0f;
    int iters = (argc >= 5) ? std::stoi(argv[4]) : 1;
    if (iters < 1) iters = 1;
    int split_x = (argc >= 6) ? std::stoi(argv[5]) : 1;
    int split_y = (argc >= 7) ? std::stoi(argv[6]) : 1;
    if (split_x < 1) split_x = 1;
    if (split_y < 1) split_y = 1;
    int batch = split_x * split_y;
    
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
    int half_cols_candidate_full = img.width / 2 + 1;
    bool width_ok = (pfm_in.width == img.width) || (pfm_in.width == half_cols_candidate_full);
    if (!width_ok || pfm_in.height != img.height ||
        (pfm_in.channels != 1 && pfm_in.channels != 2 && pfm_in.channels != 3)) {
        std::fprintf(stderr, "PFM input must be W,H=(%d,%d) or W/2+1,H with channels=1/2/3 (real[/imag])\n",
                     img.width, img.height);
        return 1;
    }
    
    // 画像を縦横に分割（コピーなしで各タイルを扱うため、幅・高さが分割数で割り切れる前提）
    if (img.height % split_y != 0 || img.width % split_x != 0) {
        std::fprintf(stderr, "image size (%d,%d) not divisible by split (%d,%d)\n", img.width, img.height, split_x, split_y);
        return 1;
    }
    const int tile_w = img.width / split_x;
    const int tile_h = img.height / split_y;
    const int tile_pixels = tile_w * tile_h;
    size_t img_bytes = img.data.size();
    // 入力をピン留めしてH2D転送を高速化（複製せず1枚のみ）
    CHECK_CUDA(cudaHostRegister(img.data.data(), img.data.size(), cudaHostRegisterDefault));

    // output fft
    const int fft_w = tile_w;
    const int fft_h = tile_h;
    const int total_pixels = tile_pixels;
    const size_t batch_pixels = static_cast<size_t>(tile_pixels) * batch;

    // define cuda woking memory for rotate & sobel
    unsigned char *d_src = nullptr, *d_dst = nullptr; // タイルを連続配置
    float  *d_mag_f = nullptr, *d_sobel_f = nullptr; 
    float* d_pfm_f = nullptr;   // IFFT出力＋シフト後
    Peak* d_block_peaks = nullptr;
    Peak* d_tmp_peaks = nullptr;
    Peak* d_final_peak = nullptr;
    Centroid* d_centroid = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, batch_pixels * sizeof(unsigned char))); // u8 なので1バイト/ピクセル
    CHECK_CUDA(cudaMalloc(&d_dst, batch_pixels * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_mag_f  , batch_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sobel_f, batch_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pfm_f  , batch_pixels * sizeof(float)));
    int peak_threads = 256;
    int peak_blocks = (total_pixels + peak_threads - 1) / peak_threads;
    CHECK_CUDA(cudaMalloc(&d_block_peaks, static_cast<size_t>(peak_blocks) * batch * sizeof(Peak)));
    int reduce_blocks = (peak_blocks + peak_threads - 1) / peak_threads;
    if (reduce_blocks < 1) reduce_blocks = 1;
    CHECK_CUDA(cudaMalloc(&d_tmp_peaks, static_cast<size_t>(reduce_blocks) * batch * sizeof(Peak)));
    CHECK_CUDA(cudaMalloc(&d_final_peak, batch * sizeof(Peak)));
    CHECK_CUDA(cudaMalloc(&d_centroid, batch * sizeof(Centroid)));

    // define FFFT woking memory & cuFFT plane（R2C 半分出力）とフル複素へ展開
    // cuFFTプランとバッファ（R2C）。出力サイズは height * (width/2 + 1) のcomplex。
    cufftHandle fft_plan;
    const int fft_out_cols = fft_w / 2 + 1;
    size_t fft_elems = static_cast<size_t>(fft_h) * fft_out_cols;
    size_t batch_fft_elems = fft_elems * batch;
    cufftComplex* d_fft1 = nullptr;
    cufftComplex* d_fft2 = nullptr;
    cufftComplex* d_fft_p = nullptr; // multiply結果
    CHECK_CUDA(cudaMalloc(&d_fft1, batch_fft_elems * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_fft2, batch_fft_elems * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_fft_p, batch_fft_elems * sizeof(cufftComplex)));
    int n[2] = {fft_h, fft_w};
    int inembed[2] = {fft_h, fft_w};
    int onembed[2] = {fft_h, fft_out_cols};
    int idist = fft_h * fft_w;
    int odist = fft_h * fft_out_cols;
    if (cufftPlanMany(&fft_plan, 2, n,
                      inembed, 1, idist,
                      onembed, 1, odist,
                      CUFFT_R2C, batch) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlanMany (R2C) failed\n");
        return 1;
    }

    cufftSetStream(fft_plan, stream);
    // 逆FFT用プラン（C2R）
    cufftHandle ifft_plan;
    if (cufftPlanMany(&ifft_plan, 2, n,
                      onembed, 1, odist,
                      inembed, 1, idist,
                      CUFFT_C2R, batch) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cufftPlanMany (C2R) failed\n");
        return 1;
    }
    cufftSetStream(ifft_plan, stream);

    // CUDAイベントでカーネル時間を個別計測（cudaEventRecord: 指定ストリーム上のタイムスタンプを記録）
    cudaEvent_t ev_rot_start, ev_rot_end, ev_sobel_start, ev_sobel_end;
    cudaEvent_t ev_fft_start, ev_fft_end, ev_ifft_start, ev_ifft_end,ev_peek_start, ev_peek_end;

    CHECK_CUDA(cudaEventCreate(&ev_rot_start));
    CHECK_CUDA(cudaEventCreate(&ev_rot_end));
    CHECK_CUDA(cudaEventCreate(&ev_sobel_start));
    CHECK_CUDA(cudaEventCreate(&ev_sobel_end));
    CHECK_CUDA(cudaEventCreate(&ev_fft_start));
    CHECK_CUDA(cudaEventCreate(&ev_fft_end));
    CHECK_CUDA(cudaEventCreate(&ev_ifft_start));
    CHECK_CUDA(cudaEventCreate(&ev_ifft_end));
    CHECK_CUDA(cudaEventCreate(&ev_peek_start));
    CHECK_CUDA(cudaEventCreate(&ev_peek_end));

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
        std::vector<cufftComplex> fft2_host(batch_fft_elems);
        for (int b = 0; b < batch; ++b) {
            int tile_x = b % split_x;
            int tile_y = b / split_x;
            cufftComplex* dst_base = fft2_host.data() + static_cast<size_t>(b) * fft_elems;
            int start_x = tile_x * tile_w;
            int start_y = tile_y * tile_h;
            for (int y = 0; y < fft_h; ++y) {
                for (int x = 0; x < fft_out_cols; ++x) {
                    int src_y = (pfm_in.height - 1) - (start_y + y); // 上下反転
                    int src_x = start_x + x;
                    if (src_x >= pfm_cols) {
                        std::fprintf(stderr, "PFM width too small for split (need at least %d columns, have %d)\n",
                                     start_x + fft_out_cols, pfm_cols);
                        return 1;
                    }
                    size_t src_idx = static_cast<size_t>(src_y) * pfm_cols + src_x;
                    size_t dst_idx = static_cast<size_t>(y) * fft_out_cols + x; // R2C半分
                    float re = 0.0f, im = 0.0f;
                    if (pfm_in.channels == 1) {
                        re = pfm_in.data[src_idx];
                    } else {
                        re = pfm_in.data[src_idx * pfm_in.channels + 0];
                        im = pfm_in.data[src_idx * pfm_in.channels + 1];
                    }
                    dst_base[dst_idx].x = re;
                    dst_base[dst_idx].y = im;
                }
            }
        }
        CHECK_CUDA(cudaMemcpy(d_fft2, fft2_host.data(), batch_fft_elems * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    }

    // 初回はコンテキスト起動やJITで遅くなりがち。ループで回すと2回目以降は速くなる（ウォームアップ効果）。
    std::vector<Peak> peaks_host(batch);
    std::vector<Centroid> centroids_host(batch);
    double center_x = static_cast<double>(tile_w)  / 2.0;
    double center_y = static_cast<double>(tile_h) / 2.0;
    for (int iter = 0; iter < iters; ++iter) {
        auto t_start = std::chrono::steady_clock::now();

        // cudaMemcpyAsync: 非同期転送。ここでは同一ストリームで順序づけているので転送完了後にカーネルが走る。
        CHECK_CUDA(cudaEventRecord(ev_rot_start, stream));

        // H2D: 各タイルを個別にコピー（幅=tile_w, 高さ=tile_h, ピッチ=img.width）
        for (int b = 0; b < batch; ++b) {
            int tile_x = b % split_x;
            int tile_y = b / split_x;
            const unsigned char* src_ptr = img.data.data() + (tile_y * tile_h * img.width) + tile_x * tile_w;
            unsigned char* dst_ptr = d_src + static_cast<size_t>(b) * tile_pixels;
            CHECK_CUDA(cudaMemcpy2DAsync(dst_ptr, tile_w * sizeof(unsigned char),
                                         src_ptr, img.width * sizeof(unsigned char),
                                         tile_w * sizeof(unsigned char), tile_h,
                                         cudaMemcpyHostToDevice, stream));
        }

        // rotate each batch
        dim3 block(16, 16);
        dim3 grid((tile_w + block.x - 1) / block.x, (tile_h + block.y - 1) / block.y);
        for (int b = 0; b < batch; ++b) {
            size_t byte_off = static_cast<size_t>(b) * tile_pixels;
            rotate_origin<<<grid, block, 0, stream>>>(d_src + byte_off, d_dst + byte_off, tile_w, tile_h, angle_rad);
        }
        CHECK_CUDA(cudaEventRecord(ev_rot_end, stream));

        // Sobel on the rotated image (d_dst) — バッチ分まわす
        dim3 block2(16, 16);
        dim3 grid2((tile_w + block2.x - 1) / block2.x,
                   (tile_h + block2.y - 1) / block2.y);
        CHECK_CUDA(cudaEventRecord(ev_sobel_start, stream));
        for (int b = 0; b < batch; ++b) {
            size_t byte_off = static_cast<size_t>(b) * tile_pixels;
            size_t float_off = static_cast<size_t>(b) * total_pixels;
            sobel3x3_mag<<<grid2, block2, 0, stream>>>(d_dst + byte_off, d_mag_f + float_off, tile_w, tile_h);
            u8_to_float_window<<<grid2, block2, 0, stream>>>(d_mag_f + float_off, d_sobel_f + float_off, tile_w, tile_h);
        }
        CHECK_CUDA(cudaEventRecord(ev_sobel_end, stream));

        
        CHECK_CUDA(cudaEventRecord(ev_fft_start, stream));
        // FFT実行（R2C 半分出力、PlanManyでバッチ処理）
        if (cufftExecR2C(fft_plan, reinterpret_cast<cufftReal*>(d_sobel_f), d_fft1) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecR2C failed\n");
            return 1;
        }
        CHECK_CUDA(cudaEventRecord(ev_fft_end, stream));
        
        // P = FFT1 * conj(FFT2)
        int threads = 256;
        int blocks = (static_cast<int>(batch_fft_elems) + threads - 1) / threads;
        CHECK_CUDA(cudaEventRecord(ev_ifft_start, stream));
        complex_mul_conj<<<blocks, threads, 0, stream>>>(d_fft1, d_fft2, d_fft_p, static_cast<int>(batch_fft_elems));
        normalize_phase<<<blocks, threads, 0, stream>>>(d_fft_p, static_cast<int>(batch_fft_elems), 1e-8f);
        // IFFTで相関ピークを得る
        if (cufftExecC2R(ifft_plan, d_fft_p, reinterpret_cast<cufftReal*>(d_sobel_f)) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2R failed\n");
            return 1;
        }
        CHECK_CUDA(cudaEventRecord(ev_ifft_end, stream));

        CHECK_CUDA(cudaEventRecord(ev_peek_start, stream));
        // スケール＆シフト（DC中心）を各バッチに適用
        float inv_scale = 1.0f / (tile_w * tile_h);
        for (int b = 0; b < batch; ++b) {
            size_t float_off = static_cast<size_t>(b) * total_pixels;
            scale_and_shift<<<grid2, block2, 0, stream>>>(d_sobel_f + float_off, d_pfm_f + float_off, tile_w, tile_h, inv_scale);
        }
        CHECK_CUDA(cudaGetLastError());

        // GPUでピークとサブピクセル重心を計算（バッチごと）
        size_t shared_bytes = peak_threads * (sizeof(float) + sizeof(int));
        CHECK_CUDA(cudaMemsetAsync(d_centroid, 0, batch * sizeof(Centroid), stream));
        for (int b = 0; b < batch; ++b) {
            size_t float_off = static_cast<size_t>(b) * total_pixels;
            const float* corr_plane = d_pfm_f + float_off;
            Peak* block_out = d_block_peaks + static_cast<size_t>(b) * peak_blocks;
            Peak* tmp_out   = d_tmp_peaks + static_cast<size_t>(b) * reduce_blocks;
            Peak* final_out = d_final_peak + b;
            block_peak<<<peak_blocks, peak_threads, shared_bytes, stream>>>(corr_plane, total_pixels, block_out);
            CHECK_CUDA(cudaGetLastError());
            reduce_peak<<<reduce_blocks, peak_threads, shared_bytes, stream>>>(block_out, peak_blocks, tmp_out);
            CHECK_CUDA(cudaGetLastError());
            reduce_peak<<<1, peak_threads, shared_bytes, stream>>>(tmp_out, reduce_blocks, final_out);
            CHECK_CUDA(cudaGetLastError());
            centroid5x5<<<1, 25, 0, stream>>>(corr_plane, tile_w, tile_h, final_out, d_centroid + b);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaEventRecord(ev_peek_end, stream));
        
        // 以降の計測・コピーが走る前にストリーム完了を待つ
        CHECK_CUDA(cudaStreamSynchronize(stream));

        CHECK_CUDA(cudaMemcpy(peaks_host.data(), d_final_peak, batch * sizeof(Peak), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(centroids_host.data(), d_centroid, batch * sizeof(Centroid), cudaMemcpyDeviceToHost));

        float rot_ms = 0.0f, sobel_ms = 0.0f,peek_ms=0.0f,fft_ms=0.0f,ifft_ms=0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&rot_ms, ev_rot_start, ev_rot_end));
        CHECK_CUDA(cudaEventElapsedTime(&sobel_ms, ev_sobel_start, ev_sobel_end));
        CHECK_CUDA(cudaEventElapsedTime(&fft_ms,  ev_fft_start, ev_fft_end));
        CHECK_CUDA(cudaEventElapsedTime(&ifft_ms, ev_ifft_start, ev_ifft_end));
        CHECK_CUDA(cudaEventElapsedTime(&peek_ms, ev_peek_start, ev_peek_end));

        auto t_end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        total_ms += elapsed_ms;
        total_rot_ms += rot_ms;
        total_sobel_ms += sobel_ms;

        std::printf("[iter %d/%d] 処理時間: %.3f ms (rotate kernel: %.3f ms, sobel kernel: %.3f ms)\n",
                    iter + 1, iters, elapsed_ms, rot_ms, sobel_ms);
        std::printf("-----------  fft %.3f ms ifft %.3f ms, peek %.3f ms)\n",
                    fft_ms, ifft_ms, peek_ms);

        // ピーク・重心結果をホストに戻して表示（バッチ平均も出す）
        double sum_shift_x = 0.0;
        double sum_shift_y = 0.0;
        double sum_response = 0.0;
        for (int b = 0; b < batch; ++b) {
            Peak peak_host = peaks_host[b];
            Centroid cent_host = centroids_host[b];
            double peak_x = peak_host.idx % tile_w;
            double peak_y = peak_host.idx / tile_w;
            double t_x = (cent_host.m00 > 0.0f) ? (cent_host.m10 / cent_host.m00) : peak_x;
            double t_y = (cent_host.m00 > 0.0f) ? (cent_host.m01 / cent_host.m00) : peak_y;
            // d_pfm_f は scale_and_shift で 1/(W*H) を掛け済みなので、さらにサイズで割らない
            double response = peak_host.val;
            double shift_x = t_x - center_x;
            double shift_y = t_y - center_y;
            if (shift_x > center_x) shift_x -= tile_w;
            if (shift_y > center_y) shift_y -= tile_h;
            sum_shift_x += shift_x;
            sum_shift_y += shift_y;
            sum_response += response;
            std::printf("batch %d: peak=(%.0f,%.0f) subpix=(%.4f,%.4f) shift=(%.4f,%.4f) response=%.6f\n",
                        b, peak_x, peak_y, t_x, t_y, shift_x, shift_y, response);
        }
        std::printf("batch average shift=(%.4f,%.4f) avg response=%.6f over %d batches\n",
                    sum_shift_x / batch, sum_shift_y / batch, sum_response / batch, batch);
    }

    double avg_ms = total_ms / iters;
    float avg_rot_ms = total_rot_ms / iters;
    float avg_sobel_ms = total_sobel_ms / iters;
    std::printf("平均: %.3f ms (rotate kernel: %.3f ms, sobel kernel: %.3f ms) over %d iters\n",
                avg_ms, avg_rot_ms, avg_sobel_ms, iters);

    //Sobel結果をホストへコピー
    PFMImage sobel_out;
    sobel_out.width    = tile_w;
    sobel_out.height   = tile_h;
    sobel_out.channels = 1;
    sobel_out.data.resize(static_cast<size_t>(tile_w) * tile_h);
    // バッチの先頭だけをダンプ
    CHECK_CUDA(cudaMemcpy(sobel_out.data.data(), d_mag_f, sobel_out.data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // std::string fft_out_path = out_path + "_fft_cuda.pfm";
    // if (!write_pfm(fft_out_path, fft_pfm)) return 1;
    // std::printf("Wrote FFT (complex, OpenCV DFT-like, PFM) to %s\n", fft_out_path.c_str());

    std::string sobel_out_path = out_path + "_sobel_cuda.pfm";
    if (!write_pfm(sobel_out_path, sobel_out)) return 1;
    std::printf("Wrote Sobel magnitude to %s\n", sobel_out_path.c_str());
    
    cudaHostUnregister(img.data.data());
    cudaEventDestroy(ev_rot_start);
    cudaEventDestroy(ev_rot_end);
    cudaEventDestroy(ev_sobel_start);
    cudaEventDestroy(ev_sobel_end);
    cudaEventDestroy(ev_fft_start);
    cudaEventDestroy(ev_fft_end);
    cudaEventDestroy(ev_ifft_start);
    cudaEventDestroy(ev_ifft_end);
    cudaEventDestroy(ev_peek_start);
    cudaEventDestroy(ev_peek_end);
    cudaStreamDestroy(stream);
    cufftDestroy(fft_plan);
    cufftDestroy(ifft_plan);

    cudaFree(d_fft1);
    cudaFree(d_fft2);
    cudaFree(d_fft_p);
    cudaFree(d_sobel_f);
    cudaFree(d_mag_f);
    cudaFree(d_pfm_f);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_block_peaks);
    cudaFree(d_tmp_peaks);
    cudaFree(d_final_peak);
    cudaFree(d_centroid);
    return 0;
}
