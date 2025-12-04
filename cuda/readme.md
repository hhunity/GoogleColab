
GPU メモリアロケターが最適化され、初期の遅さが劇的に減る。
cudaMallocAsync(&ptr, size, stream);
cudaFreeAsync(ptr, stream)


ループ前にダミー
void* dummy;
cudaMalloc(&dummy, 500*1024*1024);
cudaFree(dummy);

FFT の workspace を固定サイズで明示的に確保する
cufftSetAutoAllocation(plan, 0);
cufftMakePlan2d(..., &worksize);
cudaMalloc(...worksize...);
cufftSetWorkArea(plan, workspace_ptr);

cufftSetAutoAllocation(fft_plan, 0);

size_t worksize;
cufftMakePlan2d(fft_plan, fft_h, fft_w, CUFFT_R2C, &worksize);

// 外で固定確保
cudaMalloc(&fft_workspace, worksize);
cufftSetWorkArea(fft_plan, fft_workspace);

