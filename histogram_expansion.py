import numpy as np
import cv2

def histogram_expansion(f):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    min_val = np.min(f)
    max_val = np.max(f)
    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            output[i][j] = np.round(((f[i][j] - min_val) / (max_val - min_val)) * 255)

    return output