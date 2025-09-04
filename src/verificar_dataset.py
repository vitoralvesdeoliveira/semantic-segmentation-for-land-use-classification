import numpy as np
import os

def verificar_dataset():    
    """ Essa função verifica o arquivo .npz que engloba todo o dataset (imagens e máscaras). Retorna informações sobre formato dos dados e conteúdo. """
    ARQUIVO_DE_DADOS = os.path.abspath('./dataset/processed/segmentation_data.npz') # /normalized-numpy.npz

    if not os.path.exists(ARQUIVO_DE_DADOS):
        print(f"Erro: Arquivo não encontrado em '{ARQUIVO_DE_DADOS}'")
    else:
        # 1. Carregar os dados
        data = np.load(ARQUIVO_DE_DADOS)

        # 2. Listar os arrays salvos no arquivo
        print(f"Arrays encontrados no arquivo: {data.files}")

        # 3. Extrair os arrays para variáveis
        imagens = data['images']
        mascaras = data['masks']

        # 4. Checar as informações fundamentais
        print("\n--- ANÁLISE DOS ARRAYS ---")
        
        # Informações do array de IMAGENS
        print("\n[IMAGENS]")
        print(f"  - Shape total: {imagens.shape}")
        print(f"  - Tipo de dado (dtype): {imagens.dtype}")
        print(f"  - Valor Mínimo: {np.min(imagens):.4f}") # Deve ser próximo de 0.0
        print(f"  - Valor Máximo: {np.max(imagens):.4f}") # Deve ser próximo de 1.0

        # Informações do array de MÁSCARAS
        print("\n[MÁSCARAS]")
        print(f"  - Shape total: {mascaras.shape}")
        print(f"  - Tipo de dado (dtype): {mascaras.dtype}")
        print(f"  - Classes únicas encontradas: {np.unique(mascaras)}") # Mostra todos os IDs de classe presentes
        
