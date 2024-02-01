import numpy as np
import cv2

def contrast_stretching(f, smin, smax):
    rmin = np.min(f)
    rmax = np.max(f)

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            output = ((smax - smin) / (rmax - rmin)) * (f[i][j] - rmin) + smin

    return output.astype(type_img)