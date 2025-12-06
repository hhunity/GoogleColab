import cv2
import numpy as np
import pfm
import matplotlib.pyplot as plt
from PIL import Image

shift_x = 10
shift_y = 128

image_path = 'imori.pgm'
img_1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img1_f32 = img_1.astype(np.float32)

fft_COMPLEX_OUT = cv2.dft(img1_f32, flags=cv2.DFT_COMPLEX_OUTPUT)  # shape=(H,W,2), fft[:,:,0]=Re, fft[:,:,1]=Im
pfm.save_pfm(fft_COMPLEX_OUT,"img_DEF_COMPLEX_OUTPUT.pfm")

#順方向のDFT_REAL_OUTPUTは２チャンネルを１チャンネルにパックするらしい
fft_REAL_OUT = cv2.dft(img1_f32, flags=cv2.DFT_REAL_OUTPUT)  # shape=(H,W,2), fft[:,:,0]=Re, fft[:,:,1]=Im
pfm.save_pfm(fft_REAL_OUT,"img_DEF_DFT_REAL_OUTPUT.pfm")

fft_COMPLEX_OUT = cv2.idft(fft_COMPLEX_OUT, flags=cv2.DFT_REAL_OUTPUT)  # 形状 (H,W)
fft_REAL_OUT = cv2.idft(fft_REAL_OUT, flags=cv2.DFT_REAL_OUTPUT)  # 形状 (H,W)

#なので戻せる
plt.subplot(1, 2, 1); plt.imshow(fft_COMPLEX_OUT, cmap="gray"); plt.axis("off"); plt.title("COMPLEX_OUT")
plt.subplot(1, 2, 2); plt.imshow(fft_REAL_OUT, cmap="gray"); plt.axis("off"); plt.title("REAL_OUT")
plt.tight_layout()
plt.show()



