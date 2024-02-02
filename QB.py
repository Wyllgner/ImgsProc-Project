import numpy as np
import cv2


def average_filter(f):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            neighbours = [f[i - 1, j - 1], f[i - 1, j], f[i, j - 1], f[i + 1, j], f[i, j + 1], f[i + 1, j + 1]]

            average_value = np.average(neighbours)
            output[i, j] = average_value

    return output.astype(type_img)


def contrast_control(f, c, v):
    if c < 0:
        print("The intensity control must be a non-negative parameter")
        exit()

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

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
                output[i][j] = mean + (c / deviation) * (f[i][j] - mean)
            else:
                output[i][j] = f[i][j]

    return output.astype(type_img)


def contrast_stretching(f, smin, smax):
    rmin = np.min(f)
    rmax = np.max(f)

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            output = ((smax - smin) / (rmax - rmin)) * (f[i][j] - rmin) + smin

    return output.astype(type_img)


def convolution(f, kernel, offset):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape
    krow, kcol = kernel.shape
    type_img = f.dtype
    f = f.astype(np.float32)

    output: np.ndarray = np.zeros((row - krow + 1, col - kcol + 1), dtype=f.dtype)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(f[i:i + krow, j:j + kcol] * kernel) + offset

    return output.astype(type_img)


def smooth_image(f, kernel_size):
    row, col = f.shape
    smoothed_img = np.zeros_like(f)
    k = kernel_size // 2

    for i in range(k, row - k):
        for j in range(k, col - k):
            smoothed_img[i, j] = np.mean(f[i - k:i + k + 1, j - k:j + k + 1])

    return smoothed_img


def high_boost(f, k):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    img_type = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)
    mask = np.zeros_like(f)

    smoothed_img = smooth_image(f, 3)

    for i in range(row):
        for j in range(col):
            mask[i][j] = f[i][j] - smoothed_img[i][j]

            output[i][j] = f[i][j] + k * mask[i][j]

    return output.astype(img_type)


def histogram_equalization(f):
    bit_depth = f.dtype.itemsize * 8

    L = np.power(2, bit_depth)

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    hist_img = cv2.calcHist([f], [0], None, [256], [0, 256])

    acsum = np.cumsum(hist_img)

    eqh = np.zeros(L, dtype=hist_img.dtype)

    result: np.ndarray = np.zeros((row, col), dtype=f.dtype)

    for i in range(L):
        eqh[i] = np.round(((L - 1) / (row * col)) * acsum[i])

    for i in range(row):
        for j in range(col):
            result[i, j] = eqh[f[i, j]]

    return result


def histogram_expansion(f):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    min_val = np.min(f)
    max_val = np.max(f)
    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            output[i][j] = np.round(((f[i][j] - min_val) / (max_val - min_val)) * 255)

    return output


def logarithmic_transformation(f, c):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            output[i][j] = c * np.log1p(f[i][j])

    return output.astype(type_img)


def median_filter(f):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            neighbors = [f[i - 1, j - 1], f[i - 1, j], f[i, j - 1],
                         f[i + 1, j], f[i, j + 1], f[i + 1, j + 1]]
            median_value = np.median(neighbors)
            output[i, j] = median_value

    return output.astype(type_img)


def negative(f):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            output[i][j] = (255 - f[i][j])

    return output.astype(type_img)


def non_uniform_cross_dissolve(f, g, factor):
    if f.shape != g.shape or f.shape != factor.shape:
        print("Images and matrix t must have the same size")
        exit()

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    row, col = f.shape
    type_img = f.dtype

    f = f.astype(np.float32)
    g = g.astype(np.float32)

    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            output[i][j] = (1 - factor[i][j]) * f[i][j] + factor[i][j] * g[i][j]

    return output.astype(type_img)


def power_transformation(f, c, y):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)
    f = f.astype(np.float32)

    for i in range(row):
        for j in range(col):
            output[i][j] = c * f[i][j] ** y

    return output.astype(type_img)


def rebound(f, direction):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):

            # gira pra esquerda
            if direction == 1:
                output[i][j] = f[j][i]

            # gira pra direita
            elif direction == 2:
                output[i][j] = f[row - 1 - j][i]
            else:
                print("Invalid direction")
                exit()

    return output.astype(type_img)


def rotation(f, angle):
    angle = np.deg2rad(angle)  # Converte o angulo de graus para radianos, obg andrey

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    ic, jc = row // 2, col // 2

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            il = round(((i - ic) * np.cos(angle)) - ((j - jc) * np.sin(angle)) + ic)
            jl = round(((i - ic) * np.sin(angle)) + ((j - jc) * np.cos(angle)) + jc)

            if row - 1 >= il >= 0 and col - 1 >= jl >= 0:
                output[il][jl] = f[i][j]

    # interpolacao pra tirar os ruidos
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if (output[i][j] == [0, 0, 0]).all():
                output[i][j] = (output[i - 1][j - 1] + output[i - 1][j + 1] + output[i + 1][j - 1] + output[i + 1][
                    j + 1]) / 4

    return output.astype(type_img)


def shear(f, sy, sx):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            new_i = int(i + (j * sx))
            new_j = int((i * sy) + j)
            if 0 <= new_i < row and 0 <= new_j < col:
                output[new_i][new_j] = f[i][j]

    return output.astype(type_img)


def sobel_edge_detection(f, t):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros_like(f)

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

            output[i, j] = np.sqrt(Gx ** 2 + Gy ** 2)

    result = np.maximum(output, t)

    for i in range(row):
        for j in range(col):
            if result[i, j] == t:
                result[i, j] = 0

    return result.astype(type_img)


def thresholding(f, k):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = f.copy()
    output = output.astype(np.float32)

    for i in range(row):
        for j in range(col):
            if (output[i][j] > k).any():
                output[i][j] = 255
            else:
                output[i][j] = 0

    return output.astype(type_img)


def uniform_cross_dissolve(f, g, factor):
    if f.shape != g.shape:
        print("Images must have the same size")
        exit()

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    row, col = f.shape
    type_img = f.dtype

    f = f.astype(np.float32)
    g = g.astype(np.float32)

    output = np.zeros_like(f)

    for i in range(row):
        for j in range(col):
            output[i][j] = (1 - factor) * f[i][j] + factor * g[i][j]

    return output.astype(type_img)


def zoom_in_image(f, zoom_factor):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    new_row = int(row * zoom_factor)
    new_col = int(col * zoom_factor)

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros((new_row, new_col), dtype=f.dtype)

    for i in range(new_row):
        for j in range(new_col):
            original_row = int(i / zoom_factor)
            original_col = int(j / zoom_factor)

            original_row = min(original_row, row - 1)
            original_col = min(original_col, col - 1)

            output[i, j] = f[original_row, original_col]

    return output.astype(type_img)


def zoom_out_image(f, zoom_factor):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    new_row = int(row / zoom_factor)
    new_col = int(col / zoom_factor)

    type_img = f.dtype
    f = f.astype(np.float32)
    output = np.zeros((new_row, new_col), dtype=f.dtype)

    for i in range(new_row):
        for j in range(new_col):
            original_row = int(i * zoom_factor)
            original_col = int(j * zoom_factor)

            original_row = min(original_row, row - 1)
            original_col = min(original_col, col - 1)

            output[i, j] = f[original_row, original_col]

    return output.astype(type_img)


def laplacian_sharpening(f, t, k):
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape
    type_img = f.dtype

    output = np.zeros((row, col), dtype=f.dtype)

    g = np.zeros((row, col), dtype=f.dtype)

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            g[i - 1, j - 1] = f[i + 1, j] + f[i - 1, j] + f[i, j + 1] + f[i, j - 1] - 4 * f[i, j]

    for i in range(row):
        for j in range(col):
            if g[i, j] <= t:
                g[i, j] = 0

    for i in range(row):
        for j in range(col):
            output[i, j] = f[i, j] + g[i, j]

    return output.astype(type_img)


img2 = cv2.imread()

imagem_resultante = contrast_control(img2, 0,255)
cv2.imshow('Imagem Resultante', imagem_resultante)
cv2.imshow('Imagem original', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()




