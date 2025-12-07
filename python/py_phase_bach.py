import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

shift_x = 10
shift_y = 50
split_x = 2
split_y = 2

image_path = 'imori.pgm'
img_1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_2 = np.roll(np.roll(img_1, shift_y, axis=0), shift_x, axis=1)

img1_f32 = img_1.astype(np.float32)
img2_f32 = img_2.astype(np.float32)
cv2.imwrite(f"img_1_{}_{img_1}.pfm",img1_f32)
cv2.imwrite(f"img_2_{}_{}.pfm",img2_f32)

# 全体の位相相関
shift, response = cv2.phaseCorrelate(img1_f32, img2_f32)
print(f"full shift={shift}, response={response}")

# タイルごとの位相相関
h, w = img1_f32.shape
tile_w = w // split_x
tile_h = h // split_y
for ty in range(split_y):
    for tx in range(split_x):
        y0 = ty * tile_h
        x0 = tx * tile_w
        tile1 = img1_f32[y0:y0 + tile_h, x0:x0 + tile_w]
        tile2 = img2_f32[y0:y0 + tile_h, x0:x0 + tile_w]
        s, r = cv2.phaseCorrelate(tile1, tile2)
        cv2.imwrite(f"img_1_{x0}_{y0}.pfm",tile1)
        cv2.imwrite(f"img_2_{x0}_{y0}.pfm",tile2)
        print(f"tile ({tx},{ty}) shift={s}, response={r}")
