%%writefile rot_sobel.cu
// GPUリソース概算（幅=W, 高さ=H の場合）
// - デバイスバッファ:
//   d_src/d_dst: W*H バイト (u8)
//   d_mag_f/d_sobel_f/d_pfm_f/d_fft_half/d_fft_full: W*H*4 バイト (float)
//   d_fft1/d_fft2/d_fft_p: H*(W/2+1)*8 バイト (cufftComplex)
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
    // 入力をピン留めしてH2D転送を高速化
    CHECK_CUDA(cudaHostRegister(img.data.data(), img.data.size(), cudaHostRegisterDefault));

    // define cuda woking memory for rotate & sobel
    unsigned char *d_src = nullptr, *d_dst = nullptr;
    float  *d_mag_f = nullptr, *d_sobel_f = nullptr; 
    float* d_pfm_f = nullptr;   // FFT2入力用
    float* d_fft_half = nullptr,*d_fft_full = nullptr;
    Peak* d_block_peaks = nullptr;
    Peak* d_tmp_peaks = nullptr;
    Peak* d_final_peak = nullptr;
    Centroid* d_centroid = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, img.data.size()));
    CHECK_CUDA(cudaMalloc(&d_dst, img.data.size()));
    CHECK_CUDA(cudaMalloc(&d_mag_f  , img.data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sobel_f, img.data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pfm_f, img.data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fft_half, img.data.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fft_full, img.data.size() * sizeof(float)));
    int total_pixels = img.width * img.height;
    int peak_threads = 256;
    int peak_blocks = (total_pixels + peak_threads - 1) / peak_threads;
    CHECK_CUDA(cudaMalloc(&d_block_peaks, peak_blocks * sizeof(Peak)));
    int reduce_blocks = (peak_blocks + peak_threads - 1) / peak_threads;
    if (reduce_blocks < 1) reduce_blocks = 1;
    CHECK_CUDA(cudaMalloc(&d_tmp_peaks, reduce_blocks * sizeof(Peak)));
    CHECK_CUDA(cudaMalloc(&d_final_peak, sizeof(Peak)));
    CHECK_CUDA(cudaMalloc(&d_centroid, sizeof(Centroid)));

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
        CHECK_CUDA(cudaEventRecord(ev_rot_start, stream));

        CHECK_CUDA(cudaMemcpyAsync(d_src, img.data.data(), img.data.size(),cudaMemcpyHostToDevice, stream));

        dim3 block(16, 16);
        dim3 grid((img.width + block.x - 1) / block.x, (img.height + block.y - 1) / block.y);
        rotate_origin<<<grid, block, 0, stream>>>(d_src, d_dst, img.width, img.height, angle_rad);
        CHECK_CUDA(cudaEventRecord(ev_rot_end, stream));

        // Sobel on the rotated image (d_dst) — 常に実行
        dim3 block2(16, 16);
        dim3 grid2((img.width + block2.x - 1) / block2.x,
                   (img.height + block2.y - 1) / block2.y);
        CHECK_CUDA(cudaEventRecord(ev_sobel_start, stream));
        sobel3x3_mag<<<grid2, block2, 0, stream>>>(d_dst, d_mag_f, img.width, img.height);
        CHECK_CUDA(cudaEventRecord(ev_sobel_end, stream));

        
        CHECK_CUDA(cudaEventRecord(ev_fft_start, stream));
        // Sobel結果(u8)をFFT入力用にfloatへ変換し、ハン窓を適用してからFFT（R2C）
        u8_to_float_window<<<grid2, block2, 0, stream>>>(d_mag_f, d_sobel_f, img.width, img.height);
        // FFT実行（R2C 半分出力）
        if (cufftExecR2C(fft_plan, reinterpret_cast<cufftReal*>(d_sobel_f), d_fft1) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecR2C failed\n");
            return 1;
        }
        CHECK_CUDA(cudaEventRecord(ev_fft_end, stream));
        
        // P = FFT1 * conj(FFT2)
        int threads = 256;
        int blocks = (fft_elems + threads - 1) / threads;
        CHECK_CUDA(cudaEventRecord(ev_ifft_start, stream));
        complex_mul_conj<<<blocks, threads, 0, stream>>>(d_fft1, d_fft2, d_fft_p, static_cast<int>(fft_elems));
        normalize_phase<<<blocks, threads, 0, stream>>>(d_fft_p, static_cast<int>(fft_elems), 1e-8f);
        // IFFTで相関ピークを得る
        if (cufftExecC2R(ifft_plan, d_fft_p, reinterpret_cast<cufftReal*>(d_sobel_f)) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "cufftExecC2R failed\n");
            return 1;
        }
        CHECK_CUDA(cudaEventRecord(ev_ifft_end, stream));

        CHECK_CUDA(cudaEventRecord(ev_peek_start, stream));
        // スケール＆シフト（DC中心）
        float inv_scale = 1.0f / (img.width * img.height);
        scale_and_shift<<<grid2, block2, 0, stream>>>(d_sobel_f, d_pfm_f, img.width, img.height, inv_scale);
        // 相関出力をPFMに保存（ループ最後にホストへコピー）
        CHECK_CUDA(cudaGetLastError());

        // GPUでピークとサブピクセル重心を計算
        block_peak<<<peak_blocks, peak_threads, peak_threads * (sizeof(float) + sizeof(int)), stream>>>(d_pfm_f, total_pixels, d_block_peaks);
        CHECK_CUDA(cudaGetLastError());
        reduce_peak<<<reduce_blocks, peak_threads, peak_threads * (sizeof(float) + sizeof(int)), stream>>>(d_block_peaks, peak_blocks, d_tmp_peaks);
        CHECK_CUDA(cudaGetLastError());
        reduce_peak<<<1, peak_threads, peak_threads * (sizeof(float) + sizeof(int)), stream>>>(d_tmp_peaks, reduce_blocks, d_final_peak);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemsetAsync(d_centroid, 0, sizeof(Centroid), stream));
        centroid5x5<<<1, 25, 0, stream>>>(d_pfm_f, img.width, img.height, d_final_peak, d_centroid);
        CHECK_CUDA(cudaGetLastError());
        
        // 以降の計測・コピーが走る前にストリーム完了を待つ
        CHECK_CUDA(cudaStreamSynchronize(stream));

        Peak peak_host{};
        Centroid cent_host{};
        CHECK_CUDA(cudaMemcpy(&peak_host, d_final_peak, sizeof(Peak), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&cent_host, d_centroid, sizeof(Centroid), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_peek_end, stream));

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

        // ピーク・重心結果をホストに戻して表示
        double peak_x = peak_host.idx % img.width;
        double peak_y = peak_host.idx / img.width;
        double t_x = (cent_host.m00 > 0.0f) ? (cent_host.m10 / cent_host.m00) : peak_x;
        double t_y = (cent_host.m00 > 0.0f) ? (cent_host.m01 / cent_host.m00) : peak_y;
        // d_pfm_f は scale_and_shift で 1/(W*H) を掛け済みなので、さらにサイズで割らない
        double response = peak_host.val;
        double center_x = static_cast<double>(img.width)  / 2.0;
        double center_y = static_cast<double>(img.height) / 2.0;
        double shift_x = t_x - center_x;
        double shift_y = t_y - center_y;
        if (shift_x > center_x) shift_x -= img.width;
        if (shift_y > center_y) shift_y -= img.height;
        std::printf("phaseCorrelate peak=(%.0f,%.0f) subpix=(%.4f,%.4f) shift=(%.4f,%.4f) response=%.6f\n",
                    peak_x, peak_y, t_x, t_y, shift_x, shift_y, response);
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
    
    cudaHostUnregister(img.data.data());
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
    cudaFree(d_block_peaks);
    cudaFree(d_tmp_peaks);
    cudaFree(d_final_peak);
    cudaFree(d_centroid);
    return 0;
}
