from recorte_individual_dos_lotes import ProcessadorLotes
from pre_processamento import ImagePreProcessor,run_preprocessing_pipeline
import os
# import json

# with open('../config.json','r',encoding='utf-8') as config_file:
#       config = json.load(config_file)

cwd = os.getcwd()

RASTER_INPUT = os.path.abspath(os.path.join(cwd,'dataset', 'raw', 'Quadras-AOI.tif'))
print(RASTER_INPUT)
VECTOR_INPUT = os.path.abspath(os.path.join(cwd,'dataset', 'raw', 'Lotes-AOI.shp'))
OUTPUT_DIR = os.path.abspath(os.path.join(cwd,'dataset', 'raw','images'))
# mudar para processed dir e raw dir 

extrator = ProcessadorLotes(raster_path=RASTER_INPUT,vector_path=VECTOR_INPUT,output_dir=OUTPUT_DIR)

extrator.extrair_lotes(fids_a_processar=[24]) # caso nao seja passado o argumento, processa toda a lista

run_preprocessing_pipeline(os.path.join(OUTPUT_DIR,"png"), os.path.abspath(os.path.join(cwd,'dataset', 'processed','images')), 256)



