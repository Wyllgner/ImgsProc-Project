import numpy as np

def zoom_in_image(f, zoom_factor):

    row, col, ch = f.shape

    new_row = int(row * zoom_factor)
    new_col = int(col * zoom_factor)

    x = f.dtype
    f = f.astype(np.float32)
    s = np.zeros((new_row, new_col, ch), dtype=f.dtype)

    for i in range(new_row):
        for j in range(new_col):

            original_row = int(i / zoom_factor)
            original_col = int(j / zoom_factor)

            original_row = min(original_row, row - 1)
            original_col = min(original_col, col - 1)

            s[i, j] = f[original_row, original_col]

    return s.astype(x)