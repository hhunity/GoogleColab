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