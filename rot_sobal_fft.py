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

# Load the image in grayscale
image_path = 'imori.pgm'
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Get image dimensions
    h, w = img_gray.shape[:2]

    # Calculate the image's center coordinates
    center = (0, 0)
    
    # Define the angle for clockwise rotation (-1 degree)
    angle = -1.0

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

# Generate 1D Hanning windows for height and width
hanning_h = np.hanning(height)
hanning_w = np.hanning(width)

# Create a 2D Hanning window using outer product
hanning_2d = np.outer(hanning_h, hanning_w)

# Multiply magnitude_sobel by the 2D Hanning window
windowed_magnitude_sobel = magnitude_sobel * hanning_2d

# 1. Perform 2D FFT on the windowed_magnitude_sobel array
f_transform = np.fft.fft2(windowed_magnitude_sobel)

# 2. Apply fftshift to center the zero-frequency component
f_transform_shifted = np.fft.fftshift(f_transform)

# 3. Calculate the magnitude spectrum
magnitude_spectrum = np.abs(f_transform_shifted)

# 4. Apply a logarithmic scale for better visualization
epsilon = 1e-6 # Small constant to avoid log(0)
log_magnitude_spectrum = np.log(magnitude_spectrum + epsilon)

print("FFT and log magnitude spectrum calculated successfully after applying Hanning window.")
print(f"Shape of magnitude_spectrum: {magnitude_spectrum.shape}, Dtype: {magnitude_spectrum.dtype}")
print(f"Shape of log_magnitude_spectrum: {log_magnitude_spectrum.shape}, Dtype: {log_magnitude_spectrum.dtype}")

# f_transform は複素数配列なので、実部と虚部を分離して3チャンネルのPFM形式に変換
# 3チャンネル目はダミーとしてゼロを格納
complex_fft_output_3ch = np.zeros((f_transform.shape[0], f_transform.shape[1], 3), dtype=np.float32)
complex_fft_output_3ch[:, :, 0] = f_transform.real.astype(np.float32)
complex_fft_output_3ch[:, :, 1] = f_transform.imag.astype(np.float32)
# 3チャンネル目はゼロ（ダミー）
complex_fft_output_3ch[:, :, 2] = np.zeros_like(f_transform.real, dtype=np.float32)

# 保存パス
complex_fft_pfm_path = 'out.pgm_fft_complex.pfm'

# PFMファイルとして保存
write_pfm(complex_fft_pfm_path, complex_fft_output_3ch)

print(f"Complex FFT result saved as 3-channel PFM to {complex_fft_pfm_path}")