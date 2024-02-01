import numpy as np
import cv2

# testar mais
def thresholding(f, k):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = f.copy()
    output = output.astype(np.float32)

    for i in range(row):
        for j in range(col):
            if (output[i][j] > k).any():
                output[i][j] = 255
            else:
                output[i][j] = 0

    return output.astype(type_img)