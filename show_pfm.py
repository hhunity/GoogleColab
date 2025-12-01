from PIL import Image
import numpy as np
import struct
import re

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

def normalize(img):
    min_log_val = img.min()
    max_log_val = img.max()

    if max_log_val - min_log_val > 0:
        normalized_magnitude_spectrum = ((img - min_log_val) / (max_log_val - min_log_val) * 255).astype(np.uint8)
    else:
        # All pixel values are the same (e.g., all 0 or all constant)
        normalized_magnitude_spectrum = np.zeros_like(img, dtype=np.uint8)

    return normalized_magnitude_spectrum

def pfm_open(pfm_path):
    img_out,color = read_pfm_to_numpy(pfm_path)
    log_magnitude_spectrum_normalize = normalize(img_out)

    return Image.fromarray(log_magnitude_spectrum_normalize, mode='L')

def pfm_open_logmag(pfm_path,blog=False):
    img_out,color = read_pfm_to_numpy(pfm_path)

    if blog:
      log_magnitude_spectrum = process_pfm_for_magnitude(img_out,color)
    else:
      log_magnitude_spectrum = img_out

    log_magnitude_spectrum_normalize = normalize(log_magnitude_spectrum)

    return Image.fromarray(log_magnitude_spectrum_normalize, mode='L')

# 画像ファイルのパス
imori_path = 'imori.pgm'
pfm_path1 = 'out.pgm_fft_opencv.pfm'
pfm_path2 = 'out.pgm_fft_opencv2_Pm.pfm'
pfm_path3 = 'out.pgm_fft_opencv2_C.pfm'
pfm_path4 = 'out.pgm_fft_opencv2.pfm'

try:
    # imori.pgmをオープン
    img_imori = Image.open(imori_path)
    # グレースケール画像であることを確認
    if img_imori.mode != 'L':
        img_imori = img_imori.convert('L')
    
    # PFMファイルを読み込み
    img_out_1 = pfm_open_logmag(pfm_path1,True)
    img_out_2 = pfm_open_logmag(pfm_path2,True)
    img_out_3 = pfm_open_logmag(pfm_path3,True)
    img_out_4 = pfm_open_logmag(pfm_path4)

    # 2つの画像を並べて表示するために、新しい画像を作成
    total_width = img_imori.width + img_out_1.width + img_out_2.width+ img_out_3.width + img_out_4.width
    max_height = max(img_imori.height, img_out_1.height,img_out_2.height,img_out_3.height,img_out_4.height)

    # 新しい画像を作成 (モード 'L'はグレースケール)
    combined_img = Image.new('L', (total_width, max_height), color=255) # 背景を白に設定

    # 各画像を新しい画像に貼り付け
    combined_img.paste(img_imori, (0, 0))
    combined_img.paste(img_out_1, (img_imori.width, 0))
    combined_img.paste(img_out_2, (img_imori.width+img_out_1.width, 0))
    combined_img.paste(img_out_3, (img_imori.width+img_out_1.width+img_out_2.width, 0))
    combined_img.paste(img_out_4, (img_imori.width+img_out_1.width+img_out_2.width+img_out_3.width, 0))

    # 結合した画像を表示
    display(combined_img)

    print(f"'{imori_path}' と '{img_out_1}' '{img_out_2}' '{img_out_3}' '{img_out_4}'を並べて表示しました。")


except FileNotFoundError as e:
    print(f"エラー: {e.filename} が見つかりませんでした。ファイルが存在するか、パスが正しいか確認してください。")
except Exception as e:
    print(f"処理中にエラーが発生しました: {e}")