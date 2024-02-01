import cv2
import numpy as np


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
def power_transformation(f, c, y):
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


def rotation(f, angle):
    angle = np.deg2rad(angle)  # Converte o angulo de graus para radianos, obg andrey

    row, col, _ = f.shape

    ic, jc = row // 2, col // 2

    x = f.dtype
    f = f.astype(np.float32)
    s = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            il = round(((i - ic) * np.cos(angle)) - ((j - jc) * np.sin(angle)) + ic)
            jl = round(((i - ic) * np.sin(angle)) + ((j - jc) * np.cos(angle)) + jc)

            if row - 1 >= il >= 0 and col - 1 >= jl >= 0:
                s[il][jl] = f[i][j]

    # interpolacao pra tirar os ruidos
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if (s[i][j] == [0, 0, 0]).all():
                s[i][j] = (s[i - 1][j - 1] + s[i - 1][j + 1] + s[i + 1][j - 1] + s[i + 1][j + 1]) / 4

    return s.astype(x)


def rebound(f, direction):
    row, col, _ = f.shape

    x = f.dtype
    f = f.astype(np.float32)
    s = np.zeros_like(f)

    for i in range(row):
        for j in range(col):

            # gira pra esquerda
            if direction == 1:
                s[i][j] = f[j][i]

            # gira pra direita
            elif direction == 2:
                s[i][j] = f[row - 1 - j][i]
            else:
                print("Invalid direction")
                exit()

    return s.astype(x)


# t -> entre 0 e 255
def sobel_edge_detection(f, t):

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    x = f.dtype
    f = f.astype(np.float32)
    s = np.zeros_like(f)

    Mx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    My = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            submatrix = f[i - 1:i + 2, j - 1:j + 2]

            Gx = 0
            Gy = 0

            for k in range(3):
                for h in range(3):
                    Gx += Mx[k][h] * submatrix[k, h]
                    Gy += My[k][h] * submatrix[k, h]

            s[i, j] = np.sqrt(Gx ** 2 + Gy ** 2)

    result = np.maximum(s, t)

    for i in range(row):
        for j in range(col):
            if result[i, j] == t:
                result[i, j] = 0

    return result.astype(x)

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

def zoom_out_image(f, zoom_factor):

    row, col, ch = f.shape

    new_row = int(row / zoom_factor)
    new_col = int(col / zoom_factor)

    x = f.dtype
    f = f.astype(np.float32)
    s = np.zeros((new_row, new_col, ch), dtype=f.dtype)

    for i in range(new_row):
        for j in range(new_col):

            original_row = int(i * zoom_factor)
            original_col = int(j * zoom_factor)

            original_row = min(original_row, row - 1)
            original_col = min(original_col, col - 1)

            s[i, j] = f[original_row, original_col]

    return s.astype(x)



