import cv2
import numpy as np

def write_pfm(filename, image_data):
    """
    Saves a NumPy array as a PFM file.

    Args:
        filename (str): The path to the output PFM file.
        image_data (np.ndarray): The image data to save (float32).
    """
    with open(filename, 'wb') as file:
        # Determine dimensions and number of channels
        if image_data.ndim == 2:
            height, width = image_data.shape
            channels = 1
            header = b'Pf\n'
        elif image_data.ndim == 3:
            height, width, channels = image_data.shape
            if channels == 1:
                header = b'Pf\n'
            elif channels == 3:
                header = b'PF\n'
            else:
                raise ValueError("Unsupported number of channels for PFM: must be 1 or 3.")
        else:
            raise ValueError("Unsupported image dimensions: must be 2D (grayscale) or 3D (color).")

        # Write PFM header
        file.write(header)
        file.write(f"{width} {height}\n".encode('ascii'))
        # Use -1.0 for little-endian byte order, which is common in Python/NumPy
        file.write(b'-1.0\n')

        # Ensure data is float32 and little-endian, then write it
        # PFM typically stores data upside-down, so flip it.
        # If image_data is already 3D and channels are last (e.g., HWC), no need to reshape for flatten.
        # If it's a single channel 3D array (H, W, 1), it's treated as grayscale.
        if channels == 1 and image_data.ndim == 3:
            image_data = image_data.squeeze() # Convert (H, W, 1) to (H, W)

        # Data should be in float32 format
        image_data = image_data.astype(np.float32)
        
        # PFM is usually stored bottom-up, so flipud for consistency with common readers
        image_data = np.flipud(image_data)

        file.write(image_data.tobytes())
    print(f"Successfully wrote PFM file to {filename}")

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

def save_pfm(corr_shift,filename):
    # corr_shift が複素2chか実数2Dかで処理を分ける
    if corr_shift.ndim == 3:
        # 2ch複素
        # f_transform_shifted = np.fft.fftshift(corr_shift, axes=(0, 1))
        # magnitude_spectrum = cv2.magnitude(f_transform_shifted[:, :, 0], f_transform_shifted[:, :, 1])
        # log_magnitude_spectrum = cv2.log(magnitude_spectrum + epsilon)

        # 実部・虚部を3chに分離してPFM保存用の配列を作る
        complex_fft_output_3ch = np.zeros((corr_shift.shape[0], corr_shift.shape[1], 3), dtype=np.float32)
        complex_fft_output_3ch[:, :, 0] = corr_shift[:, :, 0]  # real
        complex_fft_output_3ch[:, :, 1] = corr_shift[:, :, 1]  # imag
        complex_fft_output_3ch[:, :, 2] = 0.0  # ダミー
        write_pfm(filename, complex_fft_output_3ch)
        print(f"Complex FFT result saved as 3-channel PFM to {filename}")
    else:
        # 実数2D（例: 相関面のidft出力）
        # f_transform_shifted = np.fft.fftshift(corr_shift, axes=(0, 1))
        # magnitude_spectrum = np.abs(f_transform_shifted)
        # log_magnitude_spectrum = np.log(magnitude_spectrum + epsilon)
        # 実数としてPFMに保存
        real_pfm_path = 'out.pgm_fft_opencv_real.pfm'
        write_pfm(filename, corr_shift.astype(np.float32))
        print(f"Real-valued result saved as PFM to {filename}")

# Load the image in grayscale
image_path = 'imori.pgm'
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

f_transform       = create_dft(img_gray)

save_pfm(f_transform,'out.pgm_fft_opencv.pfm')

if 1:
    # 任意のオフセットでDFT結果をシフトしたコピーを作る（垂直=shift_y, 水平=shift_x）
    shift_y = 10  # 上下方向のシフト量（正で下方向、負で上方向）
    shift_x = 20  # 左右方向のシフト量（正で右方向、負で左方向）
    img_gray_shifted = np.roll(np.roll(img_gray, shift_y, axis=0), shift_x, axis=1)

    # ここ読み込んでくる
    f_transform_shift = create_dft(img_gray_shifted)

    # F1, F2 は cv.dft(..., flags=cv.DFT_COMPLEX_OUTPUT) 済みの2ch複素
    # 1) 相互スペクトル P = F1 * conj(F2)
    P = cv2.mulSpectrums(f_transform, f_transform_shift, 0, conjB=True)
    save_pfm(P,'out.pgm_fft_opencv2_P.pfm')

    # 2) 振幅 Pm = |P|
    eps = 1e-6
    Pm = cv2.magnitude(P[:, :, 0], P[:, :, 1])
    Pm = cv2.max(Pm, eps) 
    save_pfm(Pm,'out.pgm_fft_opencv2_Pm.pfm')

    # 3) クロスパワースペクトル C = P / |P|
    #    振幅を2chにしてから割る
    Pm2 = cv2.merge([Pm, Pm])
    C = cv2.divide(P, Pm2)
    save_pfm(C,'out.pgm_fft_opencv2_C.pfm')

    # 4) 逆DFTで相関面（ピーク位置を見る）
    corr = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # 5) エネルギーを中央にシフト
    corr_shift = np.fft.fftshift(corr, axes=(0, 1))
else:
    # corr_shift = np.fft.fftshift(f_transform, axes=(0, 1))
    corr_shift = f_transform

save_pfm(f_transform,'out.pgm_fft_opencv2.pfm')

eps = 1e-6
if corr_shift.ndim == 2:
    magnitude_spectrum = np.abs(corr_shift)
else:
    magnitude_spectrum = cv2.magnitude(corr_shift[:, :, 0], corr_shift[:, :, 1])
magnitude_spectrum = np.nan_to_num(magnitude_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
log_magnitude_spectrum = np.log(magnitude_spectrum + eps)

print("FFT and log magnitude spectrum calculated successfully after applying Hanning window.")
print(f"Shape of magnitude_spectrum: {magnitude_spectrum.shape}, Dtype: {magnitude_spectrum.dtype}")
print(f"Shape of log_magnitude_spectrum: {log_magnitude_spectrum.shape}, Dtype: {log_magnitude_spectrum.dtype}")
