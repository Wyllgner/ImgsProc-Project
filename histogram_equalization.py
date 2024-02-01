import cv2
import numpy as np

def histogram_equalization(f):
    bit_depth = f.dtype.itemsize * 8

    L = np.power(2, bit_depth)

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    hist_img = cv2.calcHist([f], [0], None, [256], [0, 256])

    acsum = np.cumsum(hist_img)

    eqh = np.zeros(L, dtype=hist_img.dtype)

    result: np.ndarray = np.zeros((row, col), dtype=f.dtype)

    for i in range(L):
        eqh[i] = np.round(((L - 1) / (row * col)) * acsum[i])

    for i in range(row):
        for j in range(col):
            result[i, j] = eqh[f[i, j]]

    return result