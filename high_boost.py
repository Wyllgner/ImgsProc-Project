import numpy as np
import cv2

def smooth_image(f, kernel_size):
    row, col = f.shape
    smoothed_img = np.zeros_like(f)
    k = kernel_size // 2

    for i in range(k, row - k):
        for j in range(k, col - k):
            smoothed_img[i, j] = np.mean(f[i - k:i + k + 1, j - k:j + k + 1])

    return smoothed_img

def high_boost(f, k):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    img_type = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)
    mask = np.zeros_like(f)

    smoothed_img = smooth_image(f, 3)

    for i in range(row):
        for j in range(col):
            mask[i][j] = f[i][j] - smoothed_img[i][j]

            output[i][j] = f[i][j] + k * mask[i][j]

    return output.astype(img_type)