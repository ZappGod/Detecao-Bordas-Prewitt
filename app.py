import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminho da imagem e filtro cinza pra imagem
img = "./img/placa.jpg"
imagemCinza = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

# Criar máscaras Prewitt
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])

# Aplicar convolução
prewitt_x_result= cv2.filter2D(imagemCinza, -1, prewitt_x)
prewitt_y_result = cv2.filter2D(imagemCinza, -1, prewitt_y)

# Aplicando as 2 bordas
prewitt = cv2.addWeighted(prewitt_x_result, 0.5, prewitt_y_result, 0.5, 0)

# Mostrando o resultado
plt.subplot(2, 2, 1), plt.imshow(imagemCinza, cmap='gray'), plt.title('Original'), plt.axis('off')
plt.subplot(2, 2, 2), plt.imshow(prewitt_x_result, cmap='gray'), plt.title('Prewitt X'), plt.axis('off')
plt.subplot(2, 2, 3), plt.imshow(prewitt_y_result, cmap='gray'), plt.title('Prewitt Y'), plt.axis('off')
plt.subplot(2, 2, 4), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt combinado'), plt.axis('off')
plt.tight_layout()
plt.show()

# Salvando em imagens
cv2.imwrite("resultadoX.jpg",prewitt_x_result)
cv2.imwrite("resultadoY.jpg",prewitt_y_result)
cv2.imwrite("resultadoX_Y.jpg",prewitt)