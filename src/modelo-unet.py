import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # deve ser definido antes de importar o TF
import tensorflow as tf
# print(tf.__version__)

import numpy as np

ARQUIVO_DE_DADOS = os.path.abspath('./dataset/processed/normalized-numpy.npz')

data = np.load(ARQUIVO_DE_DADOS)

masks = np.expand_dims(data['masks'],axis=-1)
images = data['images']

print(images.shape)
print(masks.shape)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_unet_model(input_shape, num_classes):
    """
    Constrói uma arquitetura de modelo U-Net completa.

    Argumentos:
        input_shape (tuple): A forma da imagem de entrada (altura, largura, canais).
        num_classes (int): O número de classes para a segmentação de saída.

    Retorna:
        model (keras.Model): O modelo Keras da U-Net.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Caminho de Contração (Encoder) ---

    # Bloco 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Bloco 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bloco 3
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bloco 4
    c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # --- Bottleneck ---
    
    b = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    b = layers.Dropout(0.3)(b)
    b = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(b)

    # --- Caminho de Expansão (Decoder) ---

    # Bloco 6
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b)
    u6 = layers.concatenate([u6, c4]) # Skip Connection
    c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    # Bloco 7
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3]) # Skip Connection
    c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    # Bloco 8
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2]) # Skip Connection
    c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    # Bloco 9
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1]) # Skip Connection
    c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # --- Camada de Saída ---
    
    # A ativação 'softmax' é usada para segmentação multi-classe.
    # Se fosse binária, seria 1 filtro com ativação 'sigmoid'.
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax',padding='same')(c9)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

from tensorflow.keras.metrics import MeanIoU

# --- Parâmetros ---
# Defina o tamanho da sua imagem e o número de classes
# Exemplo para imagens 128x128 com 3 canais (RGB)
INPUT_SHAPE = (256, 256, 3) 
NUM_CLASSES = 4 # Ex: Fundo, Carro, Pedestre, Estrada

# --- Construir o Modelo ---
# Chame a função para criar a U-Net completa
modelo = build_unet_model(INPUT_SHAPE, NUM_CLASSES)

# --- Compilar o Modelo ---
# Use 'sparse_categorical_crossentropy' se suas máscaras forem de inteiros
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Visualizar a Arquitetura ---
# Isso imprimirá uma tabela com todas as camadas, shapes e parâmetros
modelo.summary()

# --- Carregar Dados e Treinar ---
# O resto do seu código para carregar o .npz e chamar model.fit() vem aqui
# ...
# historico = modelo.fit(X_train, y_train, ...)

model = modelo.fit(images,masks,batch_size=4,epochs=20)

model.save()