import numpy as np
import cv2

def uniform_cross_dissolve(f, g, factor):

    if f.shape != g.shape:
        print("Images must have the same size")
        exit()

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    row, col = f.shape
    type_img = f.dtype

    f = f.astype(np.float32)
    g = g.astype(np.float32)

    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            output[i][j] = (1 - factor) * f[i][j] + factor * g[i][j]

    return output.astype(type_img)