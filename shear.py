import numpy as np
import cv2

def shear(f, sy, sx):

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            new_i = int(i + (j * sx))
            new_j = int((i * sy) + j)
            if 0 <= new_i < row and 0 <= new_j < col:
                output[new_i][new_j] = f[i][j]

    return output.astype(type_img)