import numpy as np

def histogram_expansion(f):
    row, col, _ = f.shape

    min_val = np.min(f)
    max_val = np.max(f)
    s = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            s[i][j] = np.round(((f[i][j] - min_val) / (max_val - min_val)) * 255)

    return s