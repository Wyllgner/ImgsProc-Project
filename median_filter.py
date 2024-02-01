import cv2
import numpy as np

def median_filter(f):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)
    for i in range(1, row-1):
        for j in range(1, col-1):
            neighbors = [f[i-1, j-1], f[i-1, j], f[i, j-1],
                          f[i+1, j], f[i, j+1], f[i+1, j+1]]
            median_value = np.median(neighbors)
            output[i, j] = median_value

    return output.astype(type_img)

