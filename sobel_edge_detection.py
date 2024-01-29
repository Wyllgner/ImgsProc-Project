import numpy as np
import cv2

def sobel_edge_detection(f, t):

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    x = f.dtype
    f = f.astype(np.float32)
    s = np.zeros_like(f)

    Mx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    My = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            submatrix = f[i - 1:i + 2, j - 1:j + 2]

            Gx = 0
            Gy = 0

            for k in range(3):
                for h in range(3):
                    Gx += Mx[k][h] * submatrix[k, h]
                    Gy += My[k][h] * submatrix[k, h]

            s[i, j] = np.sqrt(Gx ** 2 + Gy ** 2)

    result = np.maximum(s, t)

    for i in range(row):
        for j in range(col):
            if result[i, j] == t:
                result[i, j] = 0

    return result.astype(x)
