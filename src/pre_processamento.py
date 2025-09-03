import cv2 as cv
import os
import numpy as np
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImagePreProcessor:
    """
    Processa um array de imagem BGR e retorna um array RGB normalizado.
    """
    def __init__(self, target_size):
        logging.info(f"Processador inicializado com target size: {target_size}'.")
        self.target_size = target_size

    def process_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None:
            logging.warning("Não foi possível ler a imagem: array é None")
            return None
        lote_resized_bgr = self.resize_with_padding(image_bgr)
        lote_resized_rgb = cv.cvtColor(lote_resized_bgr, cv.COLOR_BGR2RGB)
        lote_normalized = lote_resized_rgb.astype(np.float32) / 255.0
        return lote_normalized

    def resize_with_padding(self,img):
        """Redimensiona mantendo 'aspect ratio' e adicionando padding """
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)
        resized = cv.resize(img, (int(w*scale), int(h*scale)))
        h_pad = self.target_size - resized.shape[0]
        w_pad = self.target_size - resized.shape[1]
        top = h_pad // 2
        bottom = h_pad - top
        left = w_pad // 2
        right = w_pad - left
        padded = cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0,0,0])
        return padded

class MaskPreProcessor:
    def __init__(self, target_size: int, color_map: dict):
        self.target_size = target_size
        if not isinstance(color_map,dict):
            raise TypeError("color_map deve ser um dicionário.")
        self.color_map = color_map
        logging.info("Processador de máscara inicializado.")
    
    def process_mask(self, mask_bgr: np.ndarray) -> np.ndarray:
        """Converte uma máscara BGR em uma máscara de classes (inteiros)."""
        resized_mask = self.resize_with_padding(mask_bgr)
        
        mask_rgb = cv.cvtColor(resized_mask, cv.COLOR_BGR2RGB)
        
        # Cria um array vazio para a máscara de classes
        class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

        # Mapeia as cores para os índices de classe
        for color, class_id in self.color_map.items():
            # Encontra todos os pixels com a cor específica
            matches = np.all(mask_rgb == color, axis=-1)
            # logging.info(f"Matches: {matches}")
            class_mask[matches] = class_id
            # logging.info(f"Final pós Matches: {class_mask}")

            
        return class_mask # Retorna a máscara com shape (H, W, 1)
    
    def resize_with_padding(self,img):
        """Redimensiona mantendo 'aspect ratio' e adicionando padding """
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)
        resized = cv.resize(img, (int(w*scale), int(h*scale)))
        h_pad = self.target_size - resized.shape[0]
        w_pad = self.target_size - resized.shape[1]
        top = h_pad // 2
        bottom = h_pad - top
        left = w_pad // 2
        right = w_pad - left
        padded = cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0,0,0])
        return padded

def run_preprocessing_pipeline(input_dir: str, output_dir: str, target_size: int, mask_color_map : dict):
    """
    Orquestra o pipeline: lê imagens de um diretório, as processa e salva os resultados.
    """
    if not os.path.isdir(input_dir):
        logging.error(f"O diretório de entrada não foi encontrado: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Instancia os pré processadores
    image_preprocessor = ImagePreProcessor(target_size=target_size)
    mask_preprocessor = MaskPreProcessor(target_size=target_size, color_map= mask_color_map)
    
    # 2. Loop de I/O
    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    logging.info(f"Encontradas {len(filenames)} imagens para processar.")

    # print(filenames)
    array_de_imagens = []
    array_de_labels = []
    
    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        image_output_path = os.path.join(output_dir, 'images',os.path.splitext(filename)[0] + '.npy')
        label_output_path = os.path.join(output_dir, 'labels',os.path.splitext(filename)[0] + '_label_.npy')
        print(image_output_path)
        print(label_output_path)
        # 2a. Leitura do arquivo
        original_image = cv.imread(input_path)
        my_mask_bgr = cv.imread(os.path.abspath('./dataset/raw/masks/label.png'))
        
        if (original_image is None) or (my_mask_bgr is None):
            logging.warning(f"Não foi possível realiza a leitura da imagem ou label.")
            continue

        # 2b. Chamada da lógica de processamento pura
        processed_image = image_preprocessor.process_image(original_image)
        processed_class_mask = mask_preprocessor.process_mask(my_mask_bgr)

        # 2c. Escrita do resultado
        if processed_image is not None:
            # print(type(processed_image))
            array_de_imagens.append(processed_image)
            array_de_labels.append(processed_class_mask)
            array_de_imagens = np.array(array_de_imagens)
            array_de_labels = np.array(array_de_labels)
            # print(type(array_de_imagens))
            np.save(os.path.join(image_output_path), processed_image)
            np.save(os.path.join(label_output_path),processed_class_mask)
            # print(processed_image.shape, processed_image.size)
            logging.info(f"Imagem processada e salva em: {output_dir}")

# USAGE:
# run_preprocessing_pipeline('caminho/para/pngs', 'caminho/para/npys', 256)