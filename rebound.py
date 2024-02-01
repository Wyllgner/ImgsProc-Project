import numpy as np
import cv2

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