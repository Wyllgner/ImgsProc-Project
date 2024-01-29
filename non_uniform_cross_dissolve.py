import numpy as np

def non_uniform_cross_dissolve(f, g, factor):
    if f.shape != g.shape or f.shape != factor.shape:
        print("Images and matrix t must have the same size")
        exit()

    row, col, _ = f.shape
    x = f.dtype

    f = f.astype(np.float32)
    g = g.astype(np.float32)

    s = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            s[i][j] = (1 - factor[i][j]) * f[i][j] + factor[i][j] * g[i][j]

    return s.astype(x)