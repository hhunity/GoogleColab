import math
import cv2
import numpy as np


def _to_gray_float32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def estimate_rotation_scale_logpolar_poc(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    *,
    center=None,
    lp_size=(512, 512),
    use_hanning=True,
    eps=1e-7,
):
    """
    Log-Polar + Phase Correlation で回転[deg]とスケール比を推定する。
    想定：img_mov は img_ref を回転+拡大縮小したもの（平行移動は影響が小さくなるように振幅で処理）
    戻り値：
      rotation_deg: mov を ref に合わせるための回転角度[deg]（符号は OpenCV の座標系に依存）
      scale: mov を ref に合わせるための拡大率（>1 なら mov を縮小が必要なケースが多い）
      response: phaseCorrelate の信頼度っぽい値（高いほどピークが明瞭）
      shift_lp: Log-Polar 空間での (dx, dy)
    """

    ref = _to_gray_float32(img_ref)
    mov = _to_gray_float32(img_mov)

    if ref.shape != mov.shape:
        raise ValueError(f"img_ref and img_mov must have same shape. got {ref.shape} vs {mov.shape}")

    h, w = ref.shape[:2]
    if center is None:
        center = (w * 0.5, h * 0.5)

    # 1) 低周波優位やDC成分の影響を下げる：平均値除去 + 窓関数
    ref0 = ref - ref.mean()
    mov0 = mov - mov.mean()

    if use_hanning:
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)  # (width, height)
        ref0 = ref0 * win
        mov0 = mov0 * win

    # 2) FFT振幅画像（平行移動の影響を受けにくい）
    #    ※ log(1+|F|) にしてダイナミックレンジを圧縮
    F_ref = np.fft.fft2(ref0)
    F_mov = np.fft.fft2(mov0)

    mag_ref = np.abs(F_ref).astype(np.float32)
    mag_mov = np.abs(F_mov).astype(np.float32)

    mag_ref = np.log(mag_ref + 1.0)
    mag_mov = np.log(mag_mov + 1.0)

    # 3) スペクトル中心を画像中心へ
    mag_ref = np.fft.fftshift(mag_ref)
    mag_mov = np.fft.fftshift(mag_mov)

    # 4) Log-Polar 変換
    #    OpenCV の warpPolar では M が「半径の log スケール」を決める重要パラメータ。
    #    M を大きくするとスケール分解能が上がる一方、画像の使い方が変わる。
    #    ここでは半径 r_max を基準に、M = out_w / log(r_max) とする例。
    out_w, out_h = lp_size[0], lp_size[1]
    r_max = min(center[0], center[1], w - center[0], h - center[1])
    r_max = max(r_max, 1.0)
    M = out_w / math.log(r_max)

    flags = cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS

    lp_ref = cv2.warpPolar(mag_ref, (out_w, out_h), center, r_max, flags)
    lp_mov = cv2.warpPolar(mag_mov, (out_w, out_h), center, r_max, flags)

    # phaseCorrelate は float32 期待
    lp_ref = lp_ref.astype(np.float32)
    lp_mov = lp_mov.astype(np.float32)

    # 5) Log-Polar 空間で POC
    #    x方向シフト = log(スケール) に相当
    #    y方向シフト = 回転角に相当
    shift, response = cv2.phaseCorrelate(lp_ref, lp_mov)  # shift=(dx, dy)
    dx, dy = shift

    # 6) (dx, dy) -> scale, rotation
    #    角度：dy が lp の高さ(out_h)に対するシフト量。
    #    スケール：dx が log(r) 軸。M の定義より scale = exp(dx / M)
    #    注意：符号はデータ・実装・座標系で反転することがあるため、必要なら - を付け替える。
    rotation_deg = - (dy * 360.0 / out_h)
    scale = math.exp(dx / M)

    return rotation_deg, scale, response, (dx, dy)


def compensate_rotation_scale(img: np.ndarray, rotation_deg: float, scale: float, center=None):
    """
    推定した回転・スケールで画像を補正して返す（mov を ref に寄せる想定）。
    """
    if img.ndim == 3:
        h, w = img.shape[:2]
    else:
        h, w = img.shape
    if center is None:
        center = (w * 0.5, h * 0.5)

    # mov を ref に合わせるなら、回転は rotation_deg を適用し、スケールは 1/scale を適用することが多い
    # ただし符号や invert はケースで変わるので、結果を見て調整してください。
    M_aff = cv2.getRotationMatrix2D(center, rotation_deg, 1.0 / scale)
    return cv2.warpAffine(img, M_aff, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def estimate_translation_poc(img_ref: np.ndarray, img_mov: np.ndarray, use_hanning=True):
    """
    回転・スケール補正後の平行移動を通常POCで推定。
    """
    ref = _to_gray_float32(img_ref)
    mov = _to_gray_float32(img_mov)

    ref0 = ref - ref.mean()
    mov0 = mov - mov.mean()

    if use_hanning:
        h, w = ref0.shape
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        ref0 *= win
        mov0 *= win

    shift, response = cv2.phaseCorrelate(ref0, mov0)
    return shift, response


if __name__ == "__main__":
    # 例：2枚読み込み（同サイズ推奨）
    ref = cv2.imread("ref.png", cv2.IMREAD_GRAYSCALE)
    mov = cv2.imread("mov.png", cv2.IMREAD_GRAYSCALE)

    rot_deg, sc, resp, shift_lp = estimate_rotation_scale_logpolar_poc(
        ref, mov,
        lp_size=(512, 512),
        use_hanning=True
    )

    print("rotation_deg:", rot_deg)
    print("scale:", sc)
    print("response:", resp)
    print("logpolar shift (dx, dy):", shift_lp)

    mov_rs = compensate_rotation_scale(mov, rot_deg, sc)

    (dx, dy), resp_t = estimate_translation_poc(ref, mov_rs)
    print("translation (dx, dy):", (dx, dy), "resp:", resp_t)

import cv2
import numpy as np
import math

def rot_scale_poc_simple(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # FFT振幅
    A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
    B = cv2.dft(b, flags=cv2.DFT_COMPLEX_OUTPUT)

    magA = cv2.magnitude(A[:,:,0], A[:,:,1])
    magB = cv2.magnitude(B[:,:,0], B[:,:,1])

    magA = np.log(magA + 1)
    magB = np.log(magB + 1)

    # 中心
    h, w = a.shape
    center = (w//2, h//2)

    # log-polar
    lpA = cv2.logPolar(
        magA, center, 40,
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )
    lpB = cv2.logPolar(
        magB, center, 40,
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )

    # POC
    (dx, dy), response = cv2.phaseCorrelate(lpA, lpB)

    angle = 360.0 * dy / lpA.shape[0]
    scale = math.exp(dx / 40.0)

    return angle, scale, response