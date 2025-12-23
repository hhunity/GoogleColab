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
