import numpy as np
import cv2

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

img1 = cv2.imread(r"C:\Users\wyllg\Desktop\Unir\4_Periodo\Processamento_de_Imagens\imagens\lena_color.tif")
img2 = cv2.imread(r"C:\Users\wyllg\Desktop\Unir\4_Periodo\Processamento_de_Imagens\imagens\lena_gray_256.tif")


imagem_resultante = negative(img2)

cv2.imshow('Imagem Resultante', imagem_resultante)
cv2.imshow('Imagem original', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()