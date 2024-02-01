import numpy as np

def negative(f):
    row, col, _ = f.shape

    x = f.dtype
    s = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            s[i][j] = (255 - f[i][j])

    return s.astype(x)