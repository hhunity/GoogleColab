from PIL import Image
import numpy as np
import struct
import re
import matplotlib.pyplot as plt

def read_pfm_to_numpy(filename):
    with open(filename, 'rb') as file:
        header = file.readline().decode('utf-8').strip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s+(\d+)$', file.readline().decode('utf-8').strip())
        if not dim_match:
            raise Exception('Malformed PFM header: width/height not found.')
        width, height = map(int, dim_match.groups())

        scale = float(file.readline().decode('utf-8').strip())
        # PFM header specifies byte order by the sign of the scale factor
        # Positive scale means big-endian, negative means little-endian
        byteorder = '>' if scale > 0 else '<' # > for big-endian, < for little-endian
        scale = abs(scale)

        data = np.fromfile(file, byteorder + 'f')
        shape = (height, width, 3) if color else (height, width)
        pfm_data = np.reshape(data, shape)
        pfm_data = np.flipud(pfm_data) # PFM is typically stored bottom-up

        return pfm_data, color

def process_pfm_for_magnitude(raw_pfm_data, is_complex_fft_output):
    epsilon = 1e-6

    if is_complex_fft_output:
        # Assume 3-channel PFM where channels 0 and 1 are real and imaginary parts
        real_fft = raw_pfm_data[:, :, 0]
        imag_fft = raw_pfm_data[:, :, 1]
        magnitude_spectrum = np.sqrt(real_fft**2 + imag_fft**2)
        magnitude_spectrum = np.fft.fftshift(magnitude_spectrum)
    else:
        # Assume raw_pfm_data is already the magnitude or a single channel image
        magnitude_spectrum = raw_pfm_data
    
    # Apply logarithmic normalization
    log_magnitude_spectrum = np.log(magnitude_spectrum + epsilon)

    # Return the log-normalized float array
    return log_magnitude_spectrum

# 画像ファイルのパス
imori_path = 'imori.pgm'
pfm_path1 = 'out.pgm_fft_opencv.pfm'
pfm_path2 = 'out.pgm_fft_opencv2.pfm'

try:
    # PFMファイルを読み込み、処理してlog_magnitude_spectrumを生成
    raw_pfm_data1, is_color1 = read_pfm_to_numpy(pfm_path1)
    log_mag_spectrum1 = process_pfm_for_magnitude(raw_pfm_data1, is_color1)

    raw_pfm_data2, is_color2 = read_pfm_to_numpy(pfm_path2)
    log_mag_spectrum2 = process_pfm_for_magnitude(raw_pfm_data2, is_color2)

    # 絶対差を計算
    # abs_diff = np.abs(log_mag_spectrum1 - log_mag_spectrum2)
    abs_diff = np.abs(raw_pfm_data1 - raw_pfm_data2)

    # 差を0-255に正規化
    min_diff = abs_diff.min()
    max_diff = abs_diff.max()

    if max_diff - min_diff > 0:
        normalized_diff = ((abs_diff - min_diff) / (max_diff - min_diff) * 255).astype(np.uint8)
    else:
        normalized_diff = np.zeros_like(abs_diff, dtype=np.uint8)

    # ヒートマップとして表示
    plt.figure(figsize=(8, 6))
    plt.imshow(normalized_diff, cmap='hot', origin='lower')
    plt.colorbar(label='Absolute Difference (Normalized 0-255)')
    plt.title('Absolute Difference between Log-Normalized FFT Magnitude Spectra')
    plt.axis('off')
    plt.show()

    print(f"'Absolute difference heatmap between '{pfm_path1}' and '{pfm_path2}' displayed.")
    print(f"'min_diff:{min_diff} max_diff:{max_diff}")

except FileNotFoundError as e:
    print(f"エラー: {e.filename} が見つかりませんでした。ファイルが存在するか、パスが正しいか確認してください。")
except Exception as e:
    print(f"処理中にエラーが発生しました: {e}")