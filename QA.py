import cv2
import numpy as np
from PIL import Image


def uniform_cross_dissolve(f, g, t):
    if f.shape != g.shape:
        print("Images must have the same size")
        exit()

    row, col, _ = f.shape
    x = f.dtype

    f = f.astype(np.float32)
    g = g.astype(np.float32)

    h = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            h[i][j] = (1 - t) * f[i][j] + t * g[i][j]

    return h.astype(x)


def non_uniform_cross_dissolve(f, g, t):
    if f.shape != g.shape or f.shape != t.shape:
        print("Images and matrix t must have the same size")
        exit()

    row, col, _ = f.shape
    x = f.dtype

    f = f.astype(np.float32)
    g = g.astype(np.float32)

    s = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            s[i][j] = (1 - t[i][j]) * f[i][j] + t[i][j] * g[i][j]

    return s.astype(x)


def negative(f):
    row, col, _ = f.shape

    x = f.dtype
    s = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            s[i][j] = (255 - f[i][j])

    return s.astype(x)


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


def thresholding(f, k):
    x = f.dtype
    s = f.copy()
    s = s.astype(np.float32)

    s[s > k] = 255
    s[s <= k] = 0

    return s.astype(x)


