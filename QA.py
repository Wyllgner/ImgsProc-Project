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

# precisa ser um valor alto
def power_transformation(f,c,y):

    row, col, _ = f.shape

    x = f.dtype
    s = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            s[i][j] = c * f[i][j] ** y

    return s.astype(x)

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

def histogram_expansion(f):

    row, col, _ = f.shape

    min_val = np.min(f)
    max_val = np.max(f)
    s = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            s[i][j] = np.round(((f[i][j] - min_val) / (max_val - min_val)) * 255)

    return s

#  equalizacao(f):

def contrast_control(f, c, v):
    if c < 0:
        print("The intensity control must be a non-negative parameter")
        exit()

    row, col, _ = f.shape

    x = f.dtype
    f = f.astype(np.float32)
    s = np.zeros_like(f)

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if v == 4:
                neighbors = [f[i + 1][j], f[i - 1][j], f[i][j + 1], f[i][j - 1]]
            elif v == 2:
                neighbors = [f[i + 1][j + 1], f[i + 1][j - 1], f[i - 1][j + 1], f[i - 1][j - 1]]
            else:
                neighbors = [f[i + 1][j + 1], f[i + 1][j], f[i + 1][j - 1], f[i - 1][j],
                             f[i - 1][j + 1], f[i][j + 1], f[i - 1][j - 1], f[i][j - 1]]

            mean = np.mean(neighbors)
            deviation = np.std(neighbors)

            if deviation != 0:
                s[i][j] = mean + (c / deviation) * (f[i][j] - mean)
            else:
                s[i][j] = f[i][j]

    return s.astype(x)
