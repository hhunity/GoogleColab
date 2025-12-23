## GPUのメモリについて

### メモリの種類

1. Registers（レジスタ）
    - スレッド専用
	- 最速（1サイクル）
	- 格納できるサイズは非常に小さい
	- スレッド間共有できない
1. Shared Memory（ブロック内の高速メモリ）
    - 1ブロック内のスレッド全員で共有
	- レジスタに次いで高速（数サイクル）
	- 手動で明示的に使う
	- サイズは 48KB〜100KB 程度
	- __syncthreads() が必須（同期が必要だから）
1. L1 Cache
    - ロード/ストアで使われる
	- Kepler 以降 shared memory と物理的には統合
	- 「自動管理」される
1. L2 Cache
    - 全ての SM（Streaming Multiprocessor）で共有
1. Global Memory（VRAM）
    - 一般的な cudaMalloc で確保したメモリ
	- とても遅い（数百サイクル）
	- 全スレッド、全ブロックがアクセス可能
1. Constant Memory（特殊キャッシュ）
    - 固定で64Kほど
    - 読み込み専用で、カーネル内で書き込みできない
    - 全スレッドで共有される定数キャッシュ。全スレッドが同じアドレスを読む場合に最適
    - 宣言<br>
      `__constant__ float filter[256];`
    - ホストから書き込む<br>
      `cudaMemcpyToSymbol(filter, hostFilter, sizeof(float)*256);`
    - カーネルで読む<br>
      `float v = filter[i];  // 高速な constant cache から取得`
1. Texture memory
    - 読み込み専用で、カーネル内で書き込みできない
    - グローバルメモリの一部で、キャッシュだけ専用。128Kほど
    - 乱れたアクセスに強い
    - 自動補完 bilinear / trilinear filtering が可能
    - 特化キャッシュ 2D / 3D 空間アクセスに最適化 (近傍アクセスに強い 2D ローカリティ最適化)
    - 使用例<br>
        `cudaTextureObject_t texObj;
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = d_data;
        resDesc.res.linear.sizeInBytes = width * height * sizeof(float);
        resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
        cudaTextureDesc texDesc = {};
        texDesc.readMode = cudaReadModeElementType;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
        //カーネル側
        __global__ void kernel(cudaTextureObject_t tex) {
        float v = tex1Dfetch<float>(tex, idx);}`
1. Host Memory（CPU RAM）
	- PCIe 経由（非常に遅い）
	- pinned memory（cudaHostAlloc）で高速化可

## ユーザ操作の有無

|メモリ|直接操作可能|備考|
|:--|:--|:--|
|Register|×（間接的に割当されるだけ）|多すぎると spill して遅くなる|
|Shared memory|○（明示的にプログラマが定義）|高速、ブロック単位共有|
|Local memory|×（register 溢れ時に自動使用）|実体は global memory|
|Global memory|○（ユーザ操作）|とても遅い|
|Constant memory|△（操作は可能だが read-only）|高速キャッシュあり|
|Texture memory|○（API で専用のアクセスパス）|L1/L2 の外に独自 cache|

```
// 目的：各ブロックが担当する src の範囲で最大値（ピーク）とその index を求め、block_out[blockIdx.x] に出力する。
// 手法：
//   1) 各スレッドが src[idx] を 1 要素読む（範囲外は極小値で無効化）
//   2) warp 内で shuffle を使って最大値リダクション（値 v と index id をペアで運ぶ）
//   3) 各 warp の lane0 が結果を shared に書く（warp ごとの代表値）
//   4) warp0 が shared 上の warp 代表をさらにリダクションしてブロック最大を作り、block_out に書く
__global__ void block_peak(const float* src, int n, Peak* block_out) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 自分が担当する要素を読む。
    // idx が範囲外なら、最大値競争から確実に負けるように -1e30f を入れる（“無効スレッド”扱い）。
    float v = (idx < n) ? src[idx] : -1e30f;

    // v がどの要素由来かを追跡するラベル（グローバル index）
    // ※ idx>=n のときも id=idx のままだが、v=-1e30f なので通常は勝たない想定。
    int id = idx;

    // shuffle の参加マスク（ここでは warp 全員参加：0xffffffff）
    unsigned mask = 0xffffffff;

    // -------------------------------
    // (A) warp 内リダクション（最大値）
    // -------------------------------
    // __shfl_down_sync(mask, v, offset):
    //   同一 warp の lane+offset の値を自分の lane に持ってくる。
    // offset=16,8,4,2,1 と半分ずつ畳み込むと 32 要素の最大が 5 ステップで決まる。
    // 最大値に更新したとき、対応する index も iid に更新して「値と位置」をセットで運ぶ。
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float vv = __shfl_down_sync(mask, v, offset);
        int iid  = __shfl_down_sync(mask, id, offset);
        if (vv > v) {
            v  = vv;
            id = iid;
        }
    }
    // ここまでで、「各 warp の lane0」には、その warp 内最大 v とその index id が揃う
    // （他 lane は途中状態のまま）

    // -------------------------------
    // (B) warp 代表を shared に集める
    // -------------------------------
    // 1ブロック最大 1024 スレッド = 32 warp を想定し、shared 配列を 32 要素確保している。
    __shared__ float warp_val[32];
    __shared__ int   warp_idx[32];

    int lane = tid & (warpSize - 1);   // tid % 32
    int warp = tid / warpSize;         // warp 番号（0..warp_count-1）

    // 各 warp の lane0 が代表値を書き込む
    if (lane == 0) {
        warp_val[warp] = v;
        warp_idx[warp] = id;
    }

    // shared 書き込み完了待ち
    __syncthreads();

    // -------------------------------
    // (C) warp0 が warp 代表をリダクションしてブロック最大にする
    // -------------------------------
    if (warp == 0) {
        // ブロック内の warp 数（blockDim が 32 の倍でない場合にも対応）
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;

        // warp0 の各 lane が shared の warp_val / warp_idx を 1 つ担当して読む。
        // lane >= warp_count は無効なので極小値を与えて脱落させる。
        float val = (lane < warp_count) ? warp_val[lane] : -1e30f;
        int   idx2 = (lane < warp_count) ? warp_idx[lane] : 0;

        // warp0 内で再び shuffle リダクションして、warp 代表の中の最大を求める（＝ブロック最大）。
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float vv = __shfl_down_sync(mask, val, offset);
            int iid  = __shfl_down_sync(mask, idx2, offset);
            if (vv > val) {
                val  = vv;
                idx2 = iid;
            }
        }

        // lane0 がブロック結果を出力：block_out[blockIdx.x] に「最大値とその index」を格納
        if (lane == 0) {
            block_out[blockIdx.x].val = val;
            block_out[blockIdx.x].idx = idx2;
        }
    }
}


// block_peak の出力（ブロックごとのピーク配列）を最終的に 1 つのピークに reduce し、さらに centroid 用のモーメントを計算する。
// 目的：
//   1) block_peaks[0..block_count-1] から全体最大（最終ピーク）を求める
//   2) corr 上のピーク近傍（ここでは 5x5）からモーメント m00,m10,m01 を計算し out_centroid に出す
// 手法：
//   - 各スレッドが block_peaks をストライド走査してローカル最大を作る
//   - warp 内 shuffle + shared + warp0 reduce でブロック最大（＝全体最大）を out_peak に書く
//   - tid==0 が out_peak の位置から 5x5 を走査して m00,m10,m01 を計算する（25サンプルなので 1 スレッドで処理）
__global__ void final_peak_and_centroid(const Peak* block_peaks, int block_count,
                                        const float* corr, int width, int height,
                                        Peak* out_peak, Centroid* out_centroid) {
    int tid = threadIdx.x;

    // 各スレッドのローカル最大（初期値は極小）
    float best_val = -1e30f;
    int   best_idx = 0;

    // -------------------------------
    // (A) block_peaks をストライド走査してローカル最大を作る
    // -------------------------------
    // i = tid, tid+blockDim, tid+2*blockDim,... のように飛び飛びで担当
    // block_count が小さい場合でも blockDim で分散できる
    for (int i = tid; i < block_count; i += blockDim.x) {
        float v = block_peaks[i].val;
        if (v > best_val) {
            best_val = v;
            best_idx = block_peaks[i].idx; // 元の corr のグローバル index（想定）
        }
    }

    unsigned mask = 0xffffffff;

    // -------------------------------
    // (B) warp 内リダクション（ローカル最大同士を統合）
    // -------------------------------
    // 各 warp で最大 best_val を求め、対応する best_idx も更新してペアで運ぶ
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float v = __shfl_down_sync(mask, best_val, offset);
        int   id = __shfl_down_sync(mask, best_idx, offset);
        if (v > best_val) {
            best_val = v;
            best_idx = id;
        }
    }

    // warp 代表を shared に集める（最大 32 warp 前提）
    __shared__ float warp_val[32];
    __shared__ int   warp_idx[32];
    int lane = tid & (warpSize - 1);
    int warp = tid / warpSize;

    if (lane == 0) {
        // 各 warp の lane0 が代表値を書き込む
        warp_val[warp] = best_val;
        warp_idx[warp] = best_idx;
    }

    __syncthreads();

    // -------------------------------
    // (C) warp0 が warp 代表の最大を求め、out_peak に書く
    // -------------------------------
    if (warp == 0) {
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;

        // warp0 が shared から読み、無効 lane は極小値で脱落
        float v = (lane < warp_count) ? warp_val[lane] : -1e30f;
        int   id = (lane < warp_count) ? warp_idx[lane] : 0;

        // warp0 内で shuffle リダクションして最終最大を得る
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float vv = __shfl_down_sync(mask, v, offset);
            int   iid = __shfl_down_sync(mask, id, offset);
            if (vv > v) {
                v  = vv;
                id = iid;
            }
        }

        // lane0 が out_peak に最終ピークを書き込む（値と index）
        if (lane == 0) {
            out_peak->val = v;
            out_peak->idx = id;
        }
    }

    // out_peak に書き込みが完了するまで待つ（後段の tid==0 が idx を読む前の同期）
    __syncthreads();

    // -------------------------------
    // (D) tid==0 がピーク近傍のモーメント（centroid用）を計算する
    // -------------------------------
    // corr は width*height の 2D を 1D に並べた配列という想定。
    // peak_idx から (px,py) を復元し、±2 の 5x5 で m00,m10,m01 を計算する。
    //   m00 = Σ v
    //   m10 = Σ v * x
    //   m01 = Σ v * y
    // これらから centroid を出すなら、一般に cx=m10/m00, cy=m01/m00（m00==0 には注意）
    if (tid == 0) {
        int peak_idx = out_peak->idx;

        int px = peak_idx % width;
        int py = peak_idx / width;

        float m00 = 0.0f, m10 = 0.0f, m01 = 0.0f;

        // 5x5 の近傍（境界ははみ出しチェック）
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
```
### 改善案

3) 実務上の注意点（重要）

3.1 out_peak をグローバルメモリに書いてから同ブロック内で読む点

warp==0 && lane==0 が out_peak->idx を グローバルメモリに書き、後段で tid==0 が読みます。

この「同一カーネル内・同一ブロック内」の可視性は多くの場合 __syncthreads() で期待通り動きますが、設計としては 共有メモリに peak_idx を置く方が明確です（グローバルの読み書きを挟まない）。

改善イメージ：
	•	__shared__ int s_peak_idx;
	•	lane0 が s_peak_idx = id;
	•	__syncthreads();
	•	tid0 が s_peak_idx を使う

3.2 “タイ（同値）”の扱い

比較が if (vv > v) のみなので、同値の場合は「先に持っていた方が勝つ」になります。
必要なら「同値なら index が小さい方」などのルールを入れます。

3.3 共有配列サイズ warp_val[32]

これは blockDim.x <= 1024 前提です（CUDA の上限なので通常OK）。もし将来変えるなら注意。
