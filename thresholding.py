import numpy as np

# testar mais
def thresholding(f, k):
    row, col, _ = f.shape

    x = f.dtype
    s = f.copy()
    s = s.astype(np.float32)

    for i in range(row):
        for j in range(col):
            if (s[i][j] > k).any():
                s[i][j] = 255
            else:
                s[i][j] = 0

    return s.astype(x)