import numpy as np
import cv2

# precisa ser um valor alto
def logarithmic_transformation(f, c):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            output[i][j] = c * np.log1p(f[i][j])

    return output.astype(type_img)