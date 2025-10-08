from recorte_individual_dos_lotes import ProcessadorLotes
from pre_processamento import run_preprocessing_pipeline
from unet_model import Unet
import os

cwd = os.getcwd()

RAW_DATA =  os.path.abspath(os.path.join(cwd,'dataset', 'raw'))
PROCESSED_DATA =  os.path.abspath(os.path.join(cwd,'dataset', 'processed'))

RASTER_INPUT = os.path.abspath(os.path.join(RAW_DATA, 'Quadras-AOI.tif'))
VECTOR_INPUT = os.path.abspath(os.path.join(RAW_DATA, 'Lotes-AOI.shp'))

COLOR_MAP_RGB = {
    (0, 0, 0): 0,     # unknown    - preto
    (128, 128, 0): 1,  # edificação - amarelo
    (128, 0, 0): 2,    # industrial - vermelho
    (0, 128, 0): 3,     # vegetada - verde
    (128, 0, 128) : 4, # roxo - agropecuária
    (0,0,128):5, # agua - azul
}

DATASET_PATH = os.path.abspath(os.path.join(PROCESSED_DATA,'dataset-images-labels.npz'))

extrator = ProcessadorLotes(raster_path=RASTER_INPUT,vector_path=VECTOR_INPUT,output_dir=os.path.join(PROCESSED_DATA,"lotes-png"))

fids_to_process = os.listdir(os.path.abspath(os.path.join(PROCESSED_DATA,'labels-json')))
fids_to_process = [f.replace('.json', '') for f in fids_to_process]
fids_to_process = [f.replace('corte_', '') for f in fids_to_process]
fids_to_process = [int(f) for f in fids_to_process]

print(fids_to_process)

extrator.extrair_lotes(fids_a_processar=fids_to_process) # caso nao seja passado o argumento, processa toda a lista

run_preprocessing_pipeline(
    PROCESSED_DATA,
    DATASET_PATH,
    256,
    COLOR_MAP_RGB)

modelo = Unet(DATASET_PATH,(256,256,3),6,COLOR_MAP_RGB)

modelo.train_model()

modelo.evaluate()