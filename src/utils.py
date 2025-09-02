import cv2 as cv

def _show_resized_image(self,image,interval):
        if image is not None:
            cv.imshow("Imagem Redimensionada",image)
            cv.waitKey(interval)
            cv.destroyAllWindows()
            
def process_dir(self, show_image_before_resizing=False):
    nomes_arquivos = [image for image in os.listdir(self.image_input_dir) if image.lower().endswith('.png')]
    for image_filename in nomes_arquivos:
        imagem_normalizada = self.process_image(image_filename, show_image_before_resizing) #retorna array numpy e salva imagens pre-processadas
        output_path = os.path.join(self.output_dir,image_filename)
        np.save(output_path,imagem_normalizada)
        # salvar a mascara
        # salvar usando np.savez 
        # np.savez_compressed(output_path, image=imagem_array, mask=mascara_array) # Salva múltiplos arrays NumPy em um único arquivo.

def process_mask(self, filename, show_image_before_resizing) -> np.ndarray:
        filename_temp = 'label.png' #HARDCODED
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
        COLOR_MAP = { #HARDCODED
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
    
# def salvar_array_em_txt(array: np.ndarray, caminho_arquivo: str):
#             """
#             Salva um array RGB 3D (altura, largura, 3 canais) em um arquivo .txt,
#             com 4 casas decimais por valor.
#             """
#             with open(caminho_arquivo, 'w') as f:
#                 for linha in array:
#                     for pixel in linha:
#                         f.write(' '.join(f"{valor:.4f}" for valor in pixel))  # RGB
#                         f.write('\n')

# path = os.path.abspath(os.path.join(cwd,'src', 'label.png'))
# mask = cv.imread(path)#.astype(np.float32) / 255.0
# mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
# mask = mask.astype(np.float32) / 255.0
# salvar_array_em_txt(mask,'mask.txt')