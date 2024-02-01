import numpy as np
import cv2

# precisa ser um valor alto
def power_transformation(f, c, y):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            output[i][j] = c * f[i][j] ** y

    return output.astype(type_img)