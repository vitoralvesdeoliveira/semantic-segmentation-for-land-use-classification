from recorte_individual_dos_lotes import ProcessadorLotes
from pre_processamento import ImagePreProcessor
import os
import json
import cv2 as cv
import numpy as np

with open('../config.json','r',encoding='utf-8') as config_file:
      config = json.load(config_file)

cwd = os.getcwd()

RASTER_INPUT = os.path.abspath(os.path.join(cwd,'dataset', 'raw', 'Quadras-AOI.tif'))
print(RASTER_INPUT)
VECTOR_INPUT = os.path.abspath(os.path.join(cwd,'dataset', 'raw', 'Lotes-AOI.shp'))
OUTPUT_DIR = os.path.abspath(os.path.join(cwd,'dataset', 'processed'))

extrator = ProcessadorLotes(raster_path=RASTER_INPUT,vector_path=VECTOR_INPUT,output_dir=OUTPUT_DIR)

extrator.extrair_lotes(fids_a_processar=[0,1]) # caso nao seja passado o argumento, processa toda a lista

processor = ImagePreProcessor(image_input_dir=f'{OUTPUT_DIR}/png',output_dir=f'{OUTPUT_DIR}/preprocessed',target_size=256)

processor.process_dir(show_image_before_resizing=True)

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