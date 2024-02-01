import numpy as np
import cv2

def convolution(f, kernel, offset):

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape
    krow, kcol = kernel.shape
    type_img = f.dtype
    f = f.astype(np.float32)

    output: np.ndarray = np.zeros((row - krow + 1, col - kcol + 1), dtype=f.dtype)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(f[i:i + krow, j:j + kcol] * kernel) + offset

    return output.astype(type_img)