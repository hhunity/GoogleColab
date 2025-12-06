import cv2
import numpy as np
import pfm

def create_dft(img_gray):
    # Get image dimensions
    h, w = img_gray.shape[:2]

    # Calculate the image's center coordinates
    center = (0, 0)
    
    # Define the angle for clockwise rotation (-1 degree)
    angle = -10.0

    # Use cv2.getRotationMatrix2D to create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation to img_gray using cv2.warpAffine
    # The output image will have the same size as the input
    rotated_img = cv2.warpAffine(img_gray, rotation_matrix, (w, h))

    # Apply Sobel filter in X direction to the rotated image
    sobel_x = cv2.Sobel(rotated_img, cv2.CV_32F, 1, 0, ksize=3)

    # Apply Sobel filter in Y direction to the rotated image
    sobel_y = cv2.Sobel(rotated_img, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate the magnitude of the gradients
    magnitude_sobel = cv2.magnitude(sobel_x, sobel_y)

    write_pfm("out.pgm_sobel_opencv.pfm", magnitude_sobel)

    print(f"Successfully applied 1-degree clockwise rotation, Sobel filters, and calculated gradient magnitude for {image_path}.")
    print(f"Shape of magnitude_sobel: {magnitude_sobel.shape}, Dtype: {magnitude_sobel.dtype}")

    # Get dimensions of magnitude_sobel
    height, width = magnitude_sobel.shape

    # 2Dハン窓をOpenCVで生成（DFT入力サイズに合わせる）
    hanning_2d = cv2.createHanningWindow((width, height), cv2.CV_32F)

    # OpenCVは(幅, 高さ)の順なので、転置不要でそのまま要素ごとの積
    windowed_magnitude_sobel = magnitude_sobel.astype(np.float32) * hanning_2d

    # 1. OpenCVのDFT（複素2ch）を実行
    f_transform = cv2.dft(windowed_magnitude_sobel, flags=cv2.DFT_COMPLEX_OUTPUT)

    return f_transform

# Load the image in grayscale
image_path = 'imori.pgm'
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

f_transform       = create_dft(img_gray)

pfm.save_pfm(f_transform,'out.pgm_fft_opencv.pfm')

if 1:
    # 任意のオフセットでDFT結果をシフトしたコピーを作る（垂直=shift_y, 水平=shift_x）
    shift_y = 10  # 上下方向のシフト量（正で下方向、負で上方向）
    shift_x = 20  # 左右方向のシフト量（正で右方向、負で左方向）
    img_gray_shifted = np.roll(np.roll(img_gray, shift_y, axis=0), shift_x, axis=1)

    # ここ読み込んでくる
    f_transform_shift = create_dft(img_gray_shifted)
    pfm.save_pfm(f_transform_shift,'out.pgm_fft_opencv2_0.pfm')

    # F1, F2 は cv.dft(..., flags=cv.DFT_COMPLEX_OUTPUT) 済みの2ch複素
    # 1) 相互スペクトル P = F1 * conj(F2)
    P = cv2.mulSpectrums(f_transform, f_transform_shift, 0, conjB=True)
    pfm.save_pfm(P,'out.pgm_fft_opencv2_P.pfm')

    # 2) 振幅 Pm = |P|
    eps = 1e-6
    Pm = cv2.magnitude(P[:, :, 0], P[:, :, 1])
    Pm = cv2.max(Pm, eps) 
    pfm.save_pfm(Pm,'out.pgm_fft_opencv2_Pm.pfm')
    
    # 3) クロスパワースペクトル C = P / |P|
    #    振幅を2chにしてから割る
    Pm2 = cv2.merge([Pm, Pm])
    C = cv2.divide(P, Pm2)
    pfm.save_pfm(C,'out.pgm_fft_opencv2_C.pfm')

    # 4) 逆DFTで相関面（ピーク位置を見る）
    corr = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # 5) エネルギーを中央にシフト
    corr_shift = np.fft.fftshift(corr, axes=(0, 1))
else:
    # corr_shift = np.fft.fftshift(f_transform, axes=(0, 1))
    corr_shift = f_transform

pfm.save_pfm(corr_shift,'out.pgm_fft_opencv2.pfm')

print("FFT and log magnitude spectrum calculated successfully after applying Hanning window.")
