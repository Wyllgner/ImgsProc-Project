import numpy as np
import cv2
def zoom_in_image(f, zoom_factor):

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    new_row = int(row * zoom_factor)
    new_col = int(col * zoom_factor)

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros((new_row, new_col), dtype=f.dtype)

    for i in range(new_row):
        for j in range(new_col):

            original_row = int(i / zoom_factor)
            original_col = int(j / zoom_factor)

            original_row = min(original_row, row - 1)
            original_col = min(original_col, col - 1)

            output[i, j] = f[original_row, original_col]

    return output.astype(type_img)