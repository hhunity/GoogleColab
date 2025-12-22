# siamese_shift_512_commented.py
#
# 目的:
#   - master画像とimg画像の「平行移動 (dx, dy)」を推定する
#   - Siamese(重み共有)のCNNで特徴を抽出し、特徴同士の相関(FFT)でズレを求める
#
# 重要な考え方:
#   1) 画像(画素)そのものではなく、CNNが作る特徴マップ同士で「相関」を取る
#      -> 照明変動/ノイズ/ボケ等に対して、画素相関より頑健になりやすい
#   2) 相関マップのピーク位置が (dx,dy) の推定値
#   3) 学習は「ランダムにズラしたペア」を合成し、そのズレ(dx,dy)を当てさせる回帰問題
#
# インストール:
#   pip install torch torchvision pillow
#
# 学習:
#   python siamese_shift_512_commented.py --data ./images --epochs 20 --batch 8 --max_shift 64
#
# 生成されるもの:
#   - siamese_shift_512.pth (学習済み重み)
#
# 注意:
#   - ここでは合成データで「img = shift(master, dx, dy)」として作っています。
#     実運用での dx,dy の符号は、あなたの warp / phaseCorrelate の定義に合わせてください。
#   - soft-argmax なのでピークが複数ある周期模様だと平均化してズレることがあります。
#     （その場合は argmax + 近傍サブピクセル補間、または前回近傍ピーク優先が実務的に強いです。）

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T


# ============================================================
# 1) 便利関数群
# ============================================================

def set_seed(seed: int = 1234):
    """乱数の固定（再現性のため）"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_images(root: str) -> List[str]:
    """フォルダ配下の画像パスを列挙"""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            if os.path.splitext(f.lower())[1] in exts:
                paths.append(os.path.join(dp, f))
    paths.sort()
    return paths

def fftshift2d(x: torch.Tensor) -> torch.Tensor:
    """
    FFTで得た相関結果は「(0,0)が左上」にピークが来る表現になるので、
    見慣れた相関(中心が0シフト)にするために fftshift 相当のrollを行う。

    x: (B,H,W) の実数テンソル
    """
    B, H, W = x.shape
    return torch.roll(torch.roll(x, shifts=H // 2, dims=1), shifts=W // 2, dims=2)

def make_coords(H: int, W: int, device) -> torch.Tensor:
    """
    soft-argmaxで「期待値座標」を取るための座標テーブルを作る。

    出力:
      coords: (H*W, 2) で (x,y)
      x,y は中心が0になるようにシフト済み:
        x in [-(W/2), ..., W/2-1]
        y in [-(H/2), ..., H/2-1]
    """
    ys = torch.arange(H, device=device) - (H // 2)
    xs = torch.arange(W, device=device) - (W // 2)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # (x,y) の順で格納（慣習的にdx,dyの順に合わせやすい）
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).float()

def soft_argmax_2d(score: torch.Tensor, temperature: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    相関マップscoreから、softmax重みの期待値としてピーク位置を求める。
    “微分可能”なので学習が楽になるが、
    周期模様などでピークが複数あると平均化されるリスクがある。

    score: (B,H,W)
    temperature:
      - 小さいほど分布が尖る（argmaxに近づく）
      - 小さすぎると勾配が不安定になり得る

    戻り値:
      dx_f, dy_f: feature座標系のズレ（float）
    """
    B, H, W = score.shape
    flat = score.reshape(B, -1)               # (B,H*W)
    w = F.softmax(flat / temperature, dim=1)  # (B,H*W)

    coords = make_coords(H, W, score.device)  # (H*W,2)
    exp = w @ coords                          # (B,2) = Σ w_i * coord_i
    dx_f = exp[:, 0]
    dy_f = exp[:, 1]
    return dx_f, dy_f

def peak_confidence(corr: torch.Tensor) -> torch.Tensor:
    """
    相関マップのピークがどれだけ「際立っているか」を簡易に数値化。
    実運用で “今回の推定は怪しい” を判定する指標に使える。

    ここでは (peak - mean)/std のz-score風。
    corr: (B,H,W)
    戻り: (B,)
    """
    B = corr.shape[0]
    flat = corr.reshape(B, -1)
    peak = flat.max(dim=1).values
    mean = flat.mean(dim=1)
    std = flat.std(dim=1).clamp_min(1e-6)
    return (peak - mean) / std

def warp_by_dxdy(x: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """
    grid_sample を使って平行移動した画像を作る。
    学習データ合成（ラベル自動生成）のために使用。

    x : (B,1,H,W)
    dx,dy : (B,) ピクセル単位。dx>0 で右へ、dy>0 で下へシフト（出力画像が動く方向）

    注意:
      grid_sample は「出力の各画素が、入力のどこを参照するか」を与えるので、
      出力を右へ +dx 動かしたい場合、参照座標は左へ（-dx）ずらす。
    """
    B, C, H, W = x.shape
    device = x.device

    # 正規化座標 [-1,1] 上の基準グリッドを作る
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1)            # (H,W,2) in (x,y)
    base = base.unsqueeze(0).repeat(B, 1, 1, 1)     # (B,H,W,2)

    # dx,dy(ピクセル) を正規化座標のシフト量に変換
    # 1ピクセルの正規化量は 2/(W-1), 2/(H-1)
    sx = (2.0 * dx / (W - 1)).view(B, 1, 1)
    sy = (2.0 * dy / (H - 1)).view(B, 1, 1)

    grid = base.clone()
    grid[..., 0] = grid[..., 0] - sx   # x方向: 出力を右へ動かす => 入力参照は左へ
    grid[..., 1] = grid[..., 1] - sy   # y方向も同様

    # padding_mode="zeros" は、画像外参照を0埋めにする
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


# ============================================================
# 2) データセット（合成ペア生成）
# ============================================================

@dataclass
class AugCfg:
    """
    合成データ生成の設定
    max_shift_px: 生成する平行移動の最大値
    noise_std: ガウシアンノイズ
    brightness/contrast: 簡易照明変動
    blur_p: 簡易ブラー（avgpoolで代用）
    """
    max_shift_px: int = 64
    noise_std: float = 0.02
    brightness: float = 0.15
    contrast: float = 0.15
    blur_p: float = 0.2

class ShiftPairDataset(Dataset):
    """
    1枚の元画像から:
      master = 元画像(512x512に整形)
      img    = masterをランダム(dx,dy)で平行移動 + 劣化
    を作り、label=(dx,dy)を返す。

    つまり学習上は:
      ネットワークに (master, img) を入れて (dx,dy) を回帰させる。
    """
    def __init__(self, image_paths: List[str], size: int = 512, cfg: AugCfg = AugCfg()):
        self.paths = image_paths
        self.size = size
        self.cfg = cfg

        # 512x512・グレースケール・float(0..1) へ
        self.to_tensor = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def _photo_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """明るさ/コントラストを軽く揺らす（実画像への一般化を狙う）"""
        b = 1.0 + random.uniform(-self.cfg.brightness, self.cfg.brightness)
        c = 1.0 + random.uniform(-self.cfg.contrast, self.cfg.contrast)
        x = torch.clamp((x - 0.5) * c + 0.5, 0.0, 1.0)
        x = torch.clamp(x * b, 0.0, 1.0)
        return x

    def __getitem__(self, idx: int):
        # 元画像読み込み
        pil = Image.open(self.paths[idx]).convert("RGB")

        # master作成
        master = self.to_tensor(pil)  # (1,512,512)
        master = self._photo_jitter(master)

        # ランダム平行移動量（これが教師ラベル）
        dx = random.uniform(-self.cfg.max_shift_px, self.cfg.max_shift_px)
        dy = random.uniform(-self.cfg.max_shift_px, self.cfg.max_shift_px)

        # img = shift(master, dx, dy)
        img = warp_by_dxdy(
            master.unsqueeze(0),
            torch.tensor([dx], dtype=torch.float32),
            torch.tensor([dy], dtype=torch.float32),
        ).squeeze(0)

        # 追加劣化（ブラー/ノイズ）
        if random.random() < self.cfg.blur_p:
            # 本格的にはガウシアンやモーションブラーを入れる方が現場適合しやすいが、
            # ここでは例としてavgpoolを使用
            img = F.avg_pool2d(img.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)

        if self.cfg.noise_std > 0:
            img = torch.clamp(img + torch.randn_like(img) * self.cfg.noise_std, 0.0, 1.0)

        label = torch.tensor([dx, dy], dtype=torch.float32)
        return master, img, label


# ============================================================
# 3) Siamese モデル（共有Encoder + FFT相関）
# ============================================================

class Encoder(nn.Module):
    """
    512x512 -> 64x64 へダウンサンプルしつつ特徴抽出する小型CNN。
    stride=2 を3回かけるので全体stride=8。

    出力: (B,C,64,64)
    """
    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 512 -> 256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 48, 3, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, out_ch, 3, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f = self.net(x)
        # 相関は内積(積和)なので、チャネル方向に正規化すると安定しやすい
        # （特徴のスケールが暴れにくい）
        return F.normalize(f, dim=1, eps=1e-6)


class SiameseShiftNet(nn.Module):
    """
    Siameseの核:
      - master と img を同一Encoder(重み共有)に通す
      - 特徴マップ同士の相関をFFTで計算して相関マップを得る
      - 相関マップのピーク位置が(dx,dy)
    """
    def __init__(self, feat_ch: int = 64, stride: int = 8, temperature: float = 0.05):
        super().__init__()
        self.enc = Encoder(out_ch=feat_ch)
        self.stride = stride
        self.temperature = temperature

    def _corr_map(self, Fm: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        特徴マップ同士の相関をFFTで計算する。

        Fm,F: (B,C,Hf,Wf) ここでは Hf=Wf=64
        出力 corr: (B,Hf,Wf)

        周波数領域で:
          corr = ifft2( sum_c FFT(Fm_c) * conj(FFT(F_c)) )
        （チャネル方向に足し合わせて1枚の相関マップにする）
        """
        # rfft2 を使うと実数入力のFFTを効率計算できる
        Fm_fft = torch.fft.rfft2(Fm, norm="backward")  # (B,C,Hf,Wf/2+1)
        F_fft  = torch.fft.rfft2(F,  norm="backward")

        # クロス相関: A * conj(B)
        R = (Fm_fft * torch.conj(F_fft)).sum(dim=1)    # (B,Hf,Wf/2+1)

        # 逆FFTで空間領域の相関へ戻す
        corr = torch.fft.irfft2(R, s=(Fm.shape[-2], Fm.shape[-1]), norm="backward").real  # (B,Hf,Wf)

        # 0シフトが左上に来るので中心に移す（見た目と座標計算が楽）
        corr = fftshift2d(corr)
        return corr

    def forward(self, master: torch.Tensor, img: torch.Tensor):
        """
        入力:
          master,img: (B,1,512,512)

        出力:
          dx,dy: (B,) ピクセル単位の推定値
          conf : (B,) ピークの鋭さ（簡易信頼度）
          corr : (B,64,64) 相関マップ（デバッグ用）
        """
        # Siamese: 重み共有のEncoderに両方通す
        Fm = self.enc(master)  # (B,C,64,64)
        F  = self.enc(img)     # (B,C,64,64)

        # 特徴相関 -> 相関マップ
        corr = self._corr_map(Fm, F)  # (B,64,64)

        # 相関マップのピーク位置をsoft-argmaxで連続値として推定（feature座標系）
        dx_f, dy_f = soft_argmax_2d(corr, temperature=self.temperature)

        # feature座標 -> 元画像ピクセル座標へ
        # stride=8 なので featureの1ピクセルは元画像の8ピクセルに相当
        dx = dx_f * self.stride
        dy = dy_f * self.stride

        # 信頼度（ピークがどれだけ尖っているか）
        conf = peak_confidence(corr)
        return dx, dy, conf, corr


# ============================================================
# 4) 学習ループ
# ============================================================

def train_one_epoch(model, loader, opt, device) -> float:
    """1エポック学習して平均lossを返す"""
    model.train()
    total = 0.0

    for master, img, label in loader:
        master = master.to(device)      # (B,1,512,512)
        img = img.to(device)            # (B,1,512,512)
        label = label.to(device)        # (B,2) = [dx,dy]

        pred_dx, pred_dy, _, _ = model(master, img)
        pred = torch.stack([pred_dx, pred_dy], dim=1)  # (B,2)

        # SmoothL1(Huber)は外れ値に多少強い回帰loss
        loss = F.smooth_l1_loss(pred, label)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += loss.item() * master.size(0)

    return total / len(loader.dataset)

@torch.no_grad()
def evaluate_mae(model, loader, device) -> Tuple[float, float]:
    """検証データでMAE(dx), MAE(dy) を返す"""
    model.eval()
    sx = 0.0
    sy = 0.0
    n = 0

    for master, img, label in loader:
        master = master.to(device)
        img = img.to(device)
        label = label.to(device)

        pred_dx, pred_dy, _, _ = model(master, img)
        sx += (pred_dx - label[:, 0]).abs().sum().item()
        sy += (pred_dy - label[:, 1]).abs().sum().item()
        n += master.size(0)

    return sx / n, sy / n


# ============================================================
# 5) エントリポイント
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="学習用画像フォルダ")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_shift", type=int, default=64)
    ap.add_argument("--feat_ch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # データ列挙と分割
    paths = list_images(args.data)
    if len(paths) < 10:
        raise RuntimeError("画像が少なすぎます（最低でも10枚程度推奨）。")

    random.shuffle(paths)
    split = int(0.9 * len(paths))
    train_paths = paths[:split]
    val_paths = paths[split:]

    # Dataset / DataLoader
    cfg = AugCfg(max_shift_px=args.max_shift)
    train_ds = ShiftPairDataset(train_paths, size=512, cfg=cfg)
    val_ds = ShiftPairDataset(val_paths, size=512, cfg=cfg)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Model / Optimizer
    model = SiameseShiftNet(feat_ch=args.feat_ch, stride=8, temperature=args.temp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train
    for e in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_ld, opt, device)
        mx, my = evaluate_mae(model, val_ld, device)
        print(f"Epoch {e:02d} | loss={loss:.5f} | MAE dx={mx:.2f}px dy={my:.2f}px")

    # Save weights
    out = "siamese_shift_512.pth"
    torch.save(model.state_dict(), out)
    print("Saved:", out)


if __name__ == "__main__":
    main()