import numpy as np

# precisa ser um valor alto
def power_transformation(f, c, y):
    row, col, _ = f.shape

    x = f.dtype
    s = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            s[i][j] = c * f[i][j] ** y

    return s.astype(x)