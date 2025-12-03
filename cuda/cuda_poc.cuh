// myproc.h
struct CudaHandle; // 前方宣言のみ

CudaHandle* create_cuda_handle(int width, int height);
void destroy_cuda_handle(CudaHandle* h);

// 画像処理など、上位が呼ぶAPI
bool run_phase_correlation(CudaHandle* h,
                           const uint8_t* img_host, // H2D転送元
                           const float*  fft2_host, // 外部FFT入力
                           /* 結果など */);
