import numpy as np

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
