import cv2
import numpy as np
import pfm
import matplotlib.pyplot as plt
from PIL import Image

shift_x = 10
shift_y = 50

image_path = 'imori.pgm'
img_1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_2 = np.roll(np.roll(img_1, shift_y, axis=0), shift_x, axis=1)

img1_f32 = img_1.astype(np.float32)                     # CV_32FC1 相当
img2_f32 = img_2.astype(np.float32)                     # CV_32FC1 相当

cv2.imwrite("img_1.pfm",img1_f32)
cv2.imwrite("img_2.pfm",img2_f32,)

shift,response = cv2.phaseCorrelate(img1_f32,img2_f32)

print("shift:", shift, "response:", response)


shift,response = cv2.phaseCorrelate(img1_f32,img2_f32)

print("shift:", shift, "response:", response)

img1_f32_fft = cv2.dft(img1_f32, flags=cv2.DFT_COMPLEX_OUTPUT)  # shape=(H,W,2), fft[:,:,0]=Re, fft[:,:,1]=Im
pfm.save_pfm(img1_f32_fft,"cv_fft1.pfm")

img2_f32_fft = cv2.dft(img2_f32, flags=cv2.DFT_COMPLEX_OUTPUT)  # shape=(H,W,2), fft[:,:,0]=Re, fft[:,:,1]=Im
pfm.save_pfm(img2_f32_fft,"cv_fft2.pfm")

img_1_pfm = cv2.imread("img_1.pfm", cv2.IMREAD_GRAYSCALE)
img_2_pfm = cv2.imread("img_2.pfm", cv2.IMREAD_GRAYSCALE)

plt.subplot(1, 2, 1); plt.imshow(img_1_pfm, cmap="gray"); plt.axis("off"); plt.title("COMPLEX_OUT")
plt.subplot(1, 2, 2); plt.imshow(img_2_pfm, cmap="gray"); plt.axis("off"); plt.title("REAL_OUT")
plt.tight_layout()
plt.show()
