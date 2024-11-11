import cv2
import numpy as np
import psutil
import time
import os
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.layers import Dropout

# Define a camada personalizada
l2 = tf.keras.regularizers.L2

@tf.keras.utils.register_keras_serializable()
class CustomMaxPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='valid', **kwargs):
        super(CustomMaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.dropout = Dropout(0.5)
        self.regularizer = l2(0.01)

    def build(self, input_shape):
        super(CustomMaxPooling, self).build(input_shape)

    def call(self, inputs):
        x = tf.nn.max_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper())
        return self.dropout(x)

    def get_config(self):
        config = super(CustomMaxPooling, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

# Carrega o modelo de reconhecimento facial
model_path = 'C:\\Users\\pedro\\OneDrive\\Área de Trabalho\\face_Security\\prime_model.keras'
model = load_model(model_path, custom_objects={'CustomMaxPooling': CustomMaxPooling})

# Inicializa o detector de rostos MTCNN
detector = MTCNN()

# Função para detectar e cortar o rosto
def detect_and_crop_face(img):
    faces = detector.detect_faces(img)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        return face
    return None

# Função de reconhecimento facial
def facial_recognition():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Erro ao capturar a imagem")
        return False
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = detect_and_crop_face(rgb_img)
    if face is not None:
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        prediction = model.predict(face, verbose=0)[0][0]  
        return prediction <= 0.1  
    return False

# Função para monitorar o Discord e verificar o rosto
def monitor_discord():
    while True:
        discord_running = any("Discord" in p.name() for p in psutil.process_iter())
        if discord_running:
            print("Discord detectado. Executando reconhecimento facial...")
            if not facial_recognition():
                print("Acesso negado. Fechando o Discord.")
                for process in psutil.process_iter():
                    if process.name() == "Discord.exe":
                        process.terminate()
                        time.sleep(1)
                time.sleep(5)
            else:
                print("Acesso autorizado. Discord liberado.")
                while discord_running:
                    discord_running = any("Discord" in p.name() for p in psutil.process_iter())
                    time.sleep(10)
        else:
            time.sleep(5)


monitor_discord()



