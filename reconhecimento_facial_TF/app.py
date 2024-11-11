import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
import cv2
import numpy as np
from mtcnn import MTCNN

# Caminho para o dataset
DATASET_PATH = 'C:\\Users\\pedro\\OneDrive\\Área de Trabalho\\face_Security\\dataset - Copia'

# Defina a largura e altura da imagem
IMG_WIDTH, IMG_HEIGHT = 128, 128  # Dimensões originais


BATCH_SIZE = 32
EPOCHS = 200




# Inicializa o detector MTCNN
detector = MTCNN()

def detect_and_crop_face(img):
    # Converte a imagem para RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detecta rostos
    faces = detector.detect_faces(rgb_img)
    # Verifica se algum rosto foi detectado
    if len(faces) == 0:
        return img  # Se não detectar, retorna a imagem original
    # Recorta a região da face (primeiro rosto detectado)
    for face in faces:
        x, y, w, h = face['box']
        cropped_face = img[y:y+h, x:x+w]
        return cv2.resize(cropped_face, (128, 128))  # Redimensiona a face cortada
    return img  









# Função de pré-processamento para ImageDataGenerator
def preprocess_function(img):
    # Converte a imagem para BGR para manter a consistência com cv2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Detecta e corta o rosto
    img = detect_and_crop_face(img)
    # Converte de volta para RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Gerador de dados com a função de pré-processamento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=preprocess_function  # Adiciona a função de pré-processamento
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=preprocess_function  # Adiciona a função de pré-processamento
)

# Gerar dados de treino e validação
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = valid_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)



# Camada de MaxPooling personalizada
@tf.keras.utils.register_keras_serializable()  
class CustomMaxPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=(1, 1), strides=(4, 4), padding='valid', **kwargs):
        super(CustomMaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.dropout = Dropout(0.5)
        self.regularizer = tf.keras.regularizers.l2(0.01)

    def build(self, input_shape):
        super(CustomMaxPooling, self).build(input_shape)
    

    def call(self, inputs):
        # Aplica a operação de max pooling
        x = tf.nn.max_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper())
        return self.dropout(x)
        #return x
    
    
    def get_config(self):
        config = super(CustomMaxPooling, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

# Definindo a arquitetura da CNN
model2 = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    CustomMaxPooling(pool_size=(1, 1), strides=(4, 4), padding='valid'),
    BatchNormalization(),
    
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    CustomMaxPooling(pool_size=(1, 1), strides=(4, 4), padding='valid'),
    
    BatchNormalization(),
    
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    CustomMaxPooling(pool_size=(1, 1), strides=(4, 4), padding='valid'),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model2.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumo da arquitetura
model2.summary()

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Treinando o modelo
history = model2.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Salvando o modelo treinado
model2.save('model2_keras2.keras')


import matplotlib.pyplot as plt

# Visualizar a acurácia e a perda ao longo do treinamento
plt.figure(figsize=(12, 4))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.show()