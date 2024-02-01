import numpy as np
import cv2

def contrast_control(f, c, v):
    if c < 0:
        print("The intensity control must be a non-negative parameter")
        exit()

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if v == 4:
                neighbors = [f[i + 1][j], f[i - 1][j], f[i][j + 1], f[i][j - 1]]
            elif v == 2:
                neighbors = [f[i + 1][j + 1], f[i + 1][j - 1], f[i - 1][j + 1], f[i - 1][j - 1]]
            else:
                neighbors = [f[i + 1][j + 1], f[i + 1][j], f[i + 1][j - 1], f[i - 1][j],
                             f[i - 1][j + 1], f[i][j + 1], f[i - 1][j - 1], f[i][j - 1]]

            mean = np.mean(neighbors)
            deviation = np.std(neighbors)

            if deviation != 0:
                output[i][j] = mean + (c / deviation) * (f[i][j] - mean)
            else:
                output[i][j] = f[i][j]

    return output.astype(type_img)