import cv2 as cv
import numpy as np
import os
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImagePreProcessor:
    """
    Classe para carregar e pré-processar imagens para a rede neural.
    """
    def __init__(self,image_input_dir,output_dir, target_size):
        if not os.path.isdir(image_input_dir):
            raise FileNotFoundError(f"O diretório de entrada não foi encontrado: {image_input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Processador inicializado. Imagens serão salvas em '{output_dir}'.")
        self.image_input_dir =image_input_dir
        self.output_dir = output_dir
        self.target_size = target_size

    def process_dir(self, show_image_before_resizing=False):
        nomes_arquivos = [image for image in os.listdir(self.image_input_dir) if image.lower().endswith('.png')]
        for image_filename in nomes_arquivos:
            self.process_image(image_filename, show_image_before_resizing) #retorna array numpy e salva imagens pre-processadas

    def process_image(self, filename, show_image_before_resizing) -> np.ndarray:
        input_path = os.path.join(self.image_input_dir, filename)
        output_path = os.path.join(self.output_dir,filename)
        lote = cv.imread(input_path)
        if lote is None:
            logging.warning(f"Não foi possível ler a imagem: {input_path}")
            return None
        lote_resized_bgr = self.resize_with_padding(lote)
        if(show_image_before_resizing):
            self._show_resized_image(lote_resized_bgr,3000)
        lote_resized_rgb = cv.cvtColor(lote_resized_bgr, cv.COLOR_BGR2RGB)
        cv.imwrite(output_path,lote_resized_bgr) # Espera imagem em BGR
        return lote_resized_rgb # nao usado 
        # lote_processed = lote_resized.astype(np.float32) / 255.0
        # ARRAY NUMPY lote_processed deve estar em um metodo separado para converter img para numpy? 

    #pode ser metodo de classe?
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

    def _show_resized_image(self,image,interval):
        if image is not None:
            cv.imshow("Imagem Redimensionada",image)
            cv.waitKey(interval)
            cv.destroyAllWindows()
    