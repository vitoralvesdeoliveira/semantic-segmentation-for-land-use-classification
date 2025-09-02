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
            imagem_normalizada = self.process_image(image_filename, show_image_before_resizing) #retorna array numpy e salva imagens pre-processadas
            output_path = os.path.join(self.output_dir,image_filename)
            np.save(output_path,imagem_normalizada)
            # salvar a mascara
            # salvar usando np.savez 
            # np.savez_compressed(output_path, image=imagem_array, mask=mascara_array) # Salva múltiplos arrays NumPy em um único arquivo.


    def process_image(self, filename, show_image_before_resizing) -> np.ndarray:
        input_path = os.path.join(self.image_input_dir, filename)
        lote = cv.imread(input_path)
        if lote is None:
            logging.warning(f"Não foi possível ler a imagem: {input_path}")
            return None
        lote_resized_bgr = self.resize_with_padding(lote)
        if(show_image_before_resizing):
            self._show_resized_image(lote_resized_bgr,1000)
        lote_resized_rgb = cv.cvtColor(lote_resized_bgr, cv.COLOR_BGR2RGB)
        # cv.imwrite(output_path,lote_resized_bgr) # Espera imagem em BGR #salvando imagem aqui
        lote_normalized = lote_resized_rgb.astype(np.float32) / 255.0
        print(type(lote_normalized),lote_normalized.shape,filename)
        return lote_normalized 

    def process_mask(self, filename, show_image_before_resizing) -> np.ndarray:
        filename_temp = 'label.png'
        input_path = os.path.join(self.image_input_dir,'masks', filename_temp)
        mascara = cv.imread(input_path)
        if mascara is None:
            logging.warning(f"Não foi possível ler a mascara da imagem: {input_path}")
            return None
        mascara_resized_bgr = self.resize_with_padding(mascara)
        if(show_image_before_resizing):
            self._show_resized_image(mascara_resized_bgr,1000)
        mascara_resized_rgb = cv.cvtColor(mascara_resized_bgr, cv.COLOR_BGR2RGB)
        # ate aqui tudo igual, da pra colocar o processamento da mascara junto com a imagem
        # a mascara vai ter shape (256,256,1) e tenho que rotular com 0-(numero de classes -1)
        # mapear com o dict em view_masks_numpy e retornar uma tupla na funcao process_image com (lote_normalized , mask_normalized) 
        COLOR_MAP = {
        (0, 0, 0): 0,         # unknown
        (0, 255, 0): 1,       # pastagem
        (255, 0, 0): 2,       # agricultura
        (0, 0, 255): 3,       # água
        (128, 128, 128): 4,   # edificação
        (128, 0, 0): 5,       # indústria
        (0, 128, 0): 6        # floresta
        }
        # print(type(lote_normalized),lote_normalized.shape,filename)
        return mask_normalized 

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
