from flask import Flask, request, send_file
from PIL import Image
import os
from werkzeug.utils import secure_filename
import io
import cv2
import numpy as np

app = Flask(__name__)
app.logger.setLevel('DEBUG')  # Adicionado para definir o nível de log como DEBUG

UPLOAD_FOLDER = 'uploads'  # Certifique-se de que este diretório exista
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        app.logger.debug("Entrou na rota /process_image")  # Adicionado log
        # Obter a imagem da solicitação POST
        image_data = request.files['image'].read()
        filename = secure_filename('uploaded_image.jpg')
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(image_path, 'wb') as f:
            f.write(image_data)

        # Aplicar o filtro de negativo
        processed_image = negative(image_path)
        
        # Aqui, estou apenas salvando a imagem para ilustração
        processed_image_path = 'processed_image.jpg'
        Image.fromarray(processed_image).save(processed_image_path)

        # Enviar a imagem processada de volta
        app.logger.debug("Enviando imagem processada de volta")  # Adicionado log
        return send_file(processed_image_path, mimetype='image/jpeg', as_attachment=True)

    except Exception as e:
        app.logger.error(f"Erro durante o processamento da imagem: {e}")
        return str(e)

def negative(image_path):
    # Lógica para aplicar o filtro de negativo
    image = cv2.imread(image_path)

    # Verificar se a leitura da imagem foi bem-sucedida
    if image is None:
        raise Exception("Falha ao ler a imagem")

    # Obter as dimensões da imagem
    row, col, _ = image.shape

    # Criar uma imagem com os mesmos valores de pixel, mas com tipo de dados uint8
    negative_image = np.zeros_like(image, dtype=np.uint8)

    # Aplicar o filtro de negativo
    negative_image[:, :, 0] = 255 - image[:, :, 0]  # Canal Azul
    negative_image[:, :, 1] = 255 - image[:, :, 1]  # Canal Verde
    negative_image[:, :, 2] = 255 - image[:, :, 2]  # Canal Vermelho

    return negative_image

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
