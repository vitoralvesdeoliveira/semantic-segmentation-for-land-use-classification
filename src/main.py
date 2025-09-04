from recorte_individual_dos_lotes import ProcessadorLotes
from pre_processamento import run_preprocessing_pipeline
import os
# import json

# with open('./config.json','r',encoding='utf-8') as config_file:
#       config = json.load(config_file)
# print(config['paths']['output']) # Uma boa alternativa é usar .env

cwd = os.getcwd()

RAW_DATA =  os.path.abspath(os.path.join(cwd,'dataset', 'raw'))
PROCESSED_DATA =  os.path.abspath(os.path.join(cwd,'dataset', 'processed'))

RASTER_INPUT = os.path.abspath(os.path.join(RAW_DATA, 'Quadras-AOI.tif'))
VECTOR_INPUT = os.path.abspath(os.path.join(RAW_DATA, 'Lotes-AOI.shp'))

COLOR_MAP = {
        (0, 0, 0): 0,         # unknown
        (0, 255, 0): 1,       # pastagem
        (255, 0, 0): 2,       # agricultura
        (0, 0, 255): 3,       # água
        (128, 128, 128): 4,   # edificação
        (128, 0, 0): 5,       # indústria
        (0, 128, 0): 6        # floresta
}

extrator = ProcessadorLotes(raster_path=RASTER_INPUT,vector_path=VECTOR_INPUT,output_dir=os.path.join(PROCESSED_DATA,"lotes-png"))

extrator.extrair_lotes(fids_a_processar=[0,1,2,3,4,5,6,7,8,9,10,22,24,25,35,36,109]) # caso nao seja passado o argumento, processa toda a lista

run_preprocessing_pipeline(
    PROCESSED_DATA,
    os.path.abspath(os.path.join(PROCESSED_DATA,'normalized-numpy')),
    256,
    COLOR_MAP)