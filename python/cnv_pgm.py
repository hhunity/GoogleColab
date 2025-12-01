import sys
from PIL import Image

if 'google.colab' in sys.modules:
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
    resized_img = grayscale_img.resize((256, 256))

    # PGM形式で保存
    resized_img.save(output_path)

    print(f"'{image_path}' をグレースケールに変換し、256x256ピクセルにリサイズ後、'{output_path}' として保存しました。")

except FileNotFoundError:
    print(f"エラー: '{image_path}' が見つかりませんでした。Google Driveにファイルが存在するか、パスが正しいか確認してください。")
except Exception as e:
    print(f"処理中にエラーが発生しました: {e}")