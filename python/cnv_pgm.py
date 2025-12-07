import sys
from PIL import Image
import cv2

if 'google.colab' in sys.modules:
    from google.colab import drive
    print("Google Colab環境です")
    bcolab = True
else:
    print("Colab環境ではありません")
    bcolab = False

if bcolab:
    # Google Driveをマウント
    drive.mount('/content/drive')
    # imori.jpgのパスを設定（例: MyDriveの直下にある場合）
    # ご自身のファイルのパスに合わせて適宜変更してください
    image_path = '/content/drive/MyDrive/sample/imori.jpg'
else:
    image_path = 'img/imori.jpg'

output_path = 'imori.pgm'

try:
    # 画像をオープン
    img = Image.open(image_path)

    # グレースケールに変換
    grayscale_img = img.convert('L')

    # 256x256ピクセルにリサイズ
    w = 2048
    h = 2048
    resized_img = grayscale_img.resize((w, h))

    # PGM形式で保存
    filename = f'imori_{w}_{h}'
    resized_img.save(filename+'.pgm')
    img = cv2.imread(filename+'.pgm')
    cv2.imwrite(filename+'.pfm',img)

    print(f"'{image_path}' をグレースケールに変換し、256x256ピクセルにリサイズ後、'{output_path}' として保存しました。")

except FileNotFoundError:
    print(f"エラー: '{image_path}' が見つかりませんでした。Google Driveにファイルが存在するか、パスが正しいか確認してください。")
except Exception as e:
    print(f"処理中にエラーが発生しました: {e}")