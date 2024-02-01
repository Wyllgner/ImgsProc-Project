import numpy as np

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