import numpy as np

def contrast_stretching(f, smin, smax):
    rmin = np.min(f)
    rmax = np.max(f)

    row, col, _ = f.shape
    x = f.dtype
    s = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            s = ((smax - smin) / (rmax - rmin)) * (f[i][j] - rmin) + smin

    return s.astype(x)