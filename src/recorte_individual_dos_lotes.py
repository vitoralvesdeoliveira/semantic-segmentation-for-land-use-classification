import os
import logging
from osgeo import gdal, ogr

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessadorLotes:
    """
    Classe para extrair recortes (lotes) de um arquivo raster com base em feições de um arquivo vetorial.

    Atributos:
        raster_path (str): Caminho para o arquivo raster de entrada.
        vector_path (str): Caminho para o arquivo vetorial de entrada (shapefile).
        output_dir (str): Diretório base onde as imagens processadas serão salvas.
    """

    def __init__(self, raster_path: str, vector_path: str, output_dir: str):
        """
        Inicializa o processador de lotes.

        Args:
            raster_path (str): Caminho para o arquivo raster de entrada (ex: .tif).
            vector_path (str): Caminho para o arquivo vetorial de entrada (ex: .shp).
            output_dir (str): Diretório base para salvar os resultados.
        """
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Arquivo raster não encontrado em: {raster_path}")
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Arquivo vetorial não encontrado em: {vector_path}")

        self.raster_path = raster_path
        self.vector_path = vector_path
        self.output_dir = output_dir
        self.output_png_dir = os.path.join(self.output_dir,'png')
        os.makedirs(self.output_png_dir, exist_ok=True)
        
        logging.info("Processador de Lotes inicializado com sucesso.")

    def _processar_feicao(self, feature, raster_ds, x_res, y_res):
        """Processa uma única feição do arquivo vetorial."""
        fid = feature.GetFID()
        output_png = os.path.join(self.output_png_dir, f'lote_{fid}.png')

        try:
            # --- 1. Warp (corte) ---
            warped_ds = gdal.Warp(
                '',  # Saída em memória
                raster_ds,
                cutlineDSName=self.vector_path,
                cutlineWhere=f"FID={fid}",
                cropToCutline=True,
                dstNodata=0,
                xRes=x_res,
                yRes=y_res,
                resampleAlg='near',
                format='VRT')
            
            if warped_ds is None:
                logging.error(f"Falha ao criar o dataset em memória para a feição {fid}.")
                return

            # --- 2. Translate (conversão para PNG) ---
            gdal.Translate(
                output_png,
                warped_ds,
                format='PNG',
                outputType=gdal.GDT_Byte
            )
            logging.info(f"Feição {fid} salva em: {output_png}.")

        except Exception as e:
            logging.error(f"Falha ao processar a feição {fid}. Erro: {e}")
        finally:
            warped_ds = None
            
    def extrair_lotes(self, fids_a_processar: list = None) -> None:
        """
        Executa o processo de extração para todas as feições ou para uma lista específica.

        Args:
            fids_a_processar (list, optional): Uma lista de FIDs (Feature IDs) para processar.
                                              Se for None, todas as feições serão processadas.
        """
        raster_ds = gdal.Open(self.raster_path)
        vector_ds = ogr.Open(self.vector_path)

        if not raster_ds or not vector_ds:
            logging.error("Não foi possível abrir os arquivos de dados. Verifique os caminhos.")
            return

        try:
            #Resolução Original
            layer = vector_ds.GetLayer()
            gt = raster_ds.GetGeoTransform()
            x_res = gt[1]
            y_res = abs(gt[5])

            logging.info(f"Iniciando extração de {len(fids_a_processar)} lote(s) selecionados entre {layer.GetFeatureCount()} lotes.")
            
            if fids_a_processar:
                for fid in fids_a_processar:
                    feature = layer.GetFeature(fid)
                    if feature:
                        self._processar_feicao(feature, raster_ds, x_res, y_res)
                    else:
                        logging.warning(f"FID {fid} não encontrado no arquivo vetorial.")
            else:
                for feature in layer:
                    self._processar_feicao(feature, raster_ds, x_res, y_res)

        finally:
            raster_ds = None
            vector_ds = None
            logging.info("Processo de extração concluído!")

    
