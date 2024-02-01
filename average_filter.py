import numpy as np
import cv2

def average_filter(f):

    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    row, col = f.shape

    type_img = f.dtype
    output = np.zeros_like(f)

    for i in range(1,row-1):
        for j in range(1, col-1):
            neighbours = [f[i-1,j-1], f[i-1,j], f[i,j-1], f[i+1,j], f[i,j+1], f[i+1,j+1]]
            
            average_value = np.average(neighbours)
            output[i,j] = average_value

    return output.astype(type_img)


img1 = cv2.imread(r"C:\Users\wyllg\Desktop\Unir\4_Periodo\Processamento_de_Imagens\imagens\lena_color.tif")
img2 = cv2.imread(r"C:\Users\wyllg\Desktop\Unir\4_Periodo\Processamento_de_Imagens\imagens\lena_gray_256.tif")


imagem_resultante = average_filter(img2)

cv2.imshow('Imagem Resultante', imagem_resultante)
cv2.imshow('Imagem original', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()