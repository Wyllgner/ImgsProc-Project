import numpy as np

# precisa ser um valor alto
def logarithmic_transformation(f, c):
    row, col, _ = f.shape

    x = f.dtype
    f = f.astype(np.float32)
    s = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            s[i][j] = c * np.log1p(f[i][j])

    return s.astype(x)