import cv2
import numpy as np
import math

def rot_scale_poc_simple(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # FFT振幅
    A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
    B = cv2.dft(b, flags=cv2.DFT_COMPLEX_OUTPUT)

    magA = cv2.magnitude(A[:,:,0], A[:,:,1])
    magB = cv2.magnitude(B[:,:,0], B[:,:,1])

    magA = np.log(magA + 1)
    magB = np.log(magB + 1)

    # 中心
    h, w = a.shape
    center = (w//2, h//2)

    # log-polar
    lpA = cv2.logPolar(
        magA, center, 40,
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )
    lpB = cv2.logPolar(
        magB, center, 40,
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )

    # POC
    (dx, dy), response = cv2.phaseCorrelate(lpA, lpB)

    angle = 360.0 * dy / lpA.shape[0]
    scale = math.exp(dx / 40.0)

    return angle, scale, response