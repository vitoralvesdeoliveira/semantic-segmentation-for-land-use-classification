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

    def process_image(self, image_bgr: np.ndarray, show_image_before_resizing) -> np.ndarray:
        if image_bgr is None:
            logging.warning("Não foi possível ler a imagem: array é None")
            return None
        lote_resized_bgr = self.resize_with_padding(image_bgr)
        lote_resized_rgb = cv.cvtColor(lote_resized_bgr, cv.COLOR_BGR2RGB)
        lote_normalized = lote_resized_rgb.astype(np.float32) / 255.0
        print(type(lote_normalized),lote_normalized.shape,image_bgr)
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

def run_preprocessing_pipeline(input_dir: str, output_dir: str, target_size: int):
    """
    Orquestra o pipeline: lê imagens de um diretório, as processa e salva os resultados.
    """
    if not os.path.isdir(input_dir):
        logging.error(f"O diretório de entrada não foi encontrado: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Instancia o processador
    preprocessor = ImagePreProcessor(target_size=target_size)
    
    # 2. Loop de I/O
    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    logging.info(f"Encontradas {len(filenames)} imagens para processar.")

    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npy')

        # 2a. Leitura do arquivo
        original_image = cv.imread(input_path)
        if original_image is None:
            logging.warning(f"Não foi possível ler a imagem: {input_path}")
            continue

        # 2b. Chamada da lógica de processamento pura
        processed_image = preprocessor.process_image(original_image)
        
        # 2c. Escrita do resultado
        if processed_image is not None:
            np.save(output_path, processed_image)
            logging.info(f"Imagem processada e salva em: {output_path}")

# USAGE:
# run_preprocessing_pipeline('caminho/para/pngs', 'caminho/para/npys', 256)