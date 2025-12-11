#include <cuda_runtime.h>
#include <thread>
#include <queue>
#include <future>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <chrono>

// =====================================================
// GPU のダミーカーネル（例：単純に配列を2倍にする）
// =====================================================
__global__ void myKernel(float* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2.0f;
}

// =====================================================
// ジョブデータ構造
// =====================================================
struct GpuJob {
    float* d_data;
    int N;

    cudaEvent_t doneEvent;     // CUDA の完了通知イベント
    std::promise<void> promise; // 完了通知（future で受け取る）
};

// =====================================================
// CUDA 専用スレッドクラス
// =====================================================
class CudaWorker {
public:
    CudaWorker() {
        cudaStreamCreate(&stream_);
        worker_ = std::thread(&CudaWorker::threadFunc, this);
    }

    ~CudaWorker() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        worker_.join();
        cudaStreamDestroy(stream_);
    }

    std::future<void> submitJob(float* d_data, int N) {
        GpuJob job;
        job.d_data = d_data;
        job.N = N;

        cudaEventCreateWithFlags(&job.doneEvent, cudaEventDisableTiming);

        auto fut = job.promise.get_future();

        {
            std::lock_guard<std::mutex> lk(mtx_);
            queue_.push(std::move(job));
        }
        cv_.notify_one();

        return fut;
    }

private:
    void threadFunc() {
        while (true) {
            GpuJob job;

            // 仕事を取り出す
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
                if (stop_) return;

                job = std::move(queue_.front());
                queue_.pop();
            }

            // ------ GPU ジョブ実行 ------
            int block = 256;
            int grid = (job.N + block - 1) / block;

            myKernel<<<grid, block, 0, stream_>>>(job.d_data, job.N);
            cudaEventRecord(job.doneEvent, stream_);

            // ------ 非同期で完了を待つ（busy-waitしない） ------
            while (cudaEventQuery(job.doneEvent) == cudaErrorNotReady) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }

            // 完了したので future に通知
            job.promise.set_value();

            cudaEventDestroy(job.doneEvent);
        }
    }

    std::thread worker_;
    cudaStream_t stream_;
    std::queue<GpuJob> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// =====================================================
// メイン側
// =====================================================
int main()
{
    const int N = 1024;
    float* d_data;
    cudaMalloc(&d_data, sizeof(float) * N);

    // ダミー初期化（全て1.0に）
    float h_data[N];
    std::fill(h_data, h_data + N, 1.0f);
    cudaMemcpy(d_data, h_data, sizeof(float) * N, cudaMemcpyHostToDevice);

    CudaWorker worker;

    // ------ GPU ジョブを依頼（Queue に積む） ------
    auto fut = worker.submitJob(d_data, N);

    // ------ 完了まで待つ（ここは CPU idle で軽い） ------
    fut.get();  // 完了したら復帰

    cudaMemcpy(h_data, d_data, sizeof(float) * N, cudaMemcpyDeviceToHost);

    std::cout << "h_data[0] = " << h_data[0] << std::endl; // 2.0 になっているはず

    cudaFree(d_data);
    return 0;
}