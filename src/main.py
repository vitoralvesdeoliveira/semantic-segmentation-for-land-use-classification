from recorte_individual_dos_lotes import ProcessadorLotes
from pre_processamento import ImagePreProcessor
import os

cwd = os.getcwd()

RASTER_INPUT = os.path.abspath(os.path.join(cwd,'src','dataset', 'raw', 'Quadras-AOI.tif'))
VECTOR_INPUT = os.path.abspath(os.path.join(cwd,'src','dataset', 'raw', 'Lotes-AOI.shp'))
OUTPUT_DIR = os.path.abspath(os.path.join(cwd,'src','dataset', 'processed'))

extrator = ProcessadorLotes(raster_path=RASTER_INPUT,vector_path=VECTOR_INPUT,output_dir=OUTPUT_DIR)

extrator.extrair_lotes(fids_a_processar=[24,50]) # caso nao seja passado o argumento, processa toda a lista

processor = ImagePreProcessor(image_input_dir=f'{OUTPUT_DIR}/png',output_dir=f'{OUTPUT_DIR}/preprocessed',target_size=256)

processor.process_dir(show_image_before_resizing=False)
