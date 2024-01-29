import cv2
import numpy as np

def median_filter(f):
    row, col, _ = f.shape
    x = f.dtype
    s = np.zeros_like(f)
    for i in range(1, row-1):
        for j in range(1, col-1):
            neighbors = [f[i-1, j-1], f[i-1, j], f[i, j-1],
                          f[i+1, j], f[i, j+1], f[i+1, j+1]]
            median_value = np.median(neighbors)
            s[i, j] = median_value

    return s.astype(x)

