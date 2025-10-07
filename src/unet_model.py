import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # deve ser definido antes de importar o TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from pre_processamento import Processor, ImagePreProcessor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class Unet:
    
    def __init__(self, DATASET_PATH : str, INPUT_SHAPE : tuple, NUM_CLASSES : int,MODEL = None):
        self.DATASET_PATH = os.path.abspath(DATASET_PATH)
        self.INPUT_SHAPE = INPUT_SHAPE
        self.NUM_CLASSES = NUM_CLASSES
        self.TEST_IMAGES = None
        self.TEST_LABELS = None
        
        if (MODEL==None):
            self.MODEL = self.build_unet_model()
        else:
            self.MODEL = MODEL
    
    def build_unet_model(self) -> keras.Model:
        """
        Constrói uma arquitetura de modelo U-Net completa.

        Argumentos:
            input_shape (tuple): A forma da imagem de entrada (altura, largura, canais).
            num_classes (int): O número de classes para a segmentação de saída.

        Retorna:
            model (keras.Model): O modelo Keras da U-Net.
        """
        inputs = layers.Input(shape=self.INPUT_SHAPE)

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
        outputs = layers.Conv2D(self.NUM_CLASSES, (1, 1), activation='softmax',padding='same')(c9)

        model = keras.Model(inputs=[inputs], outputs=[outputs])
        
        return model

    def train_model(self,EPOCHS : int= 10,BATCH_SIZE : int = 2,OPTIMIZER : str = 'adam', LOSS = keras.losses.SparseCategoricalCrossentropy(from_logits=False), METRICS : list[str]= ['accuracy'], MODEL_NAME = 'unet_model'):
        data = np.load(self.DATASET_PATH)
        masks = data['masks']
        images = data['images']
        
        # DIVIDE IMAGENS EM TREINO/TESTE/VALIDAÇÃO:
        self.TRAIN_VAL_IMAGES, self.TEST_IMAGES, self.TRAIN_VAL_LABELS, self.TEST_LABELS = train_test_split(images,masks,test_size=0.2,random_state=42)
        self.TRAIN_IMAGES, self.VAL_IMAGES, self.TRAIN_LABELS, self.VAL_LABELS = train_test_split(self.TRAIN_VAL_IMAGES,self.TRAIN_VAL_LABELS,test_size=0.2,random_state=42)
        
        self.MODEL.compile(
            optimizer=OPTIMIZER,
            loss=LOSS,
            metrics=METRICS
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_unet.keras', monitor='val_loss', save_best_only=True)
        historico = self.MODEL.fit(self.TRAIN_IMAGES,self.TRAIN_LABELS,validation_data=(self.VAL_IMAGES,self.VAL_LABELS),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[early_stop, checkpoint])
        # historico = self.MODEL.fit(self.TRAIN_IMAGES,self.TRAIN_LABELS,batch_size=BATCH_SIZE,epochs=EPOCHS)
        self.MODEL.save(f"{MODEL_NAME}_{EPOCHS}_epochs_{OPTIMIZER}_{LOSS}.keras")
        
        return historico
    
    def predict(self,image_path : str | list[str]): # image_path : str | list[str]
        COLOR_MAP_RGB = {
        (0, 0, 0): 0,     # unknown    
        (128, 128, 0): 1,  # edificação - amarelo
        (128, 0, 0): 2,    # industrial - vermelho
        (0, 128, 0): 3,     # vegetada - verde
        }
        processor = ImagePreProcessor(self.INPUT_SHAPE[0])
        image = cv.imread(os.path.abspath(image_path))
        processed_image = processor.process_image(image) # retorna o lote normalizado
        # image = processor.resize_with_padding(image).astype(np.float32)
        # PREPARA IMAGENS PARA O MODELO
        processed_image = np.expand_dims(processed_image, axis=0) 
        print(f"processed_image.shape ----> {processed_image.shape}")
        pred_mask = self.MODEL.predict(processed_image)
        print(f"pred_mask.shape ----> {pred_mask.shape}")
        pred_mask = np.argmax(pred_mask, axis=-1)  # shape: (1, 256, 256)
        print(f"pred_mask.shape pós argmax ----> {pred_mask.shape}")
        pred_mask = pred_mask[0]  # remove a dimensão de batch
        print(f"pred_mask.shape pós remover dimensão de batch----> {pred_mask.shape}")
        segmented_image = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        
        for class_idx, color in enumerate(COLOR_MAP_RGB):
            segmented_image[pred_mask == class_idx] = color
            
        print(f"  - Classes únicas encontradas(segmented_image): {np.unique(segmented_image)}") 

        print(f"segmented_image.shape ----> {segmented_image.shape}")

        return segmented_image
    
    def evaluate(self):
        results = self.MODEL.evaluate(self.TEST_IMAGES,self.TEST_LABELS)
        return results