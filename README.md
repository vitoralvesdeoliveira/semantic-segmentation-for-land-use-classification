# Segmentação Semântica para Classificação de Uso do Solo

Este projeto é um pipeline de preparação de dados projetado para processar dados geoespaciais (imagens de satélite/aéreas e shapefiles) e gerar um conjunto de dados de imagens padronizadas, prontas para o treinamento de modelos de deep learning para classificação de uso do solo.

## Funcionalidades Principais

- **Extração Georreferenciada**: Recorta imagens de lotes individuais a partir de um grande arquivo raster, usando os polígonos de um arquivo vetorial (shapefile) como moldes.
- **Pré-processamento de Imagem**: Padroniza as imagens extraídas para um tamanho fixo (ex: 256x256 pixels), mantendo a proporção original através da adição de preenchimento (`padding`).
- **Automação**: Scripts que automatizam o fluxo de trabalho desde os dados brutos até o dataset pronto para o treinamento.

## Tecnologias Utilizadas

- **Python 3.x**
- **GDAL/OGR**: Para manipulação de dados geoespaciais (raster e vetorial).
- **OpenCV-Python**: Para tarefas de processamento de imagem.
- **NumPy**: Para operações numéricas eficientes.

---

## Como Configurar e Executar o Projeto

Siga os passos abaixo para configurar o ambiente e executar o pipeline de processamento.

### 1. Pré-requisitos

- **Python 3.8+** instalado.
- **Git** para clonar o repositório.
- **GDAL (biblioteca do sistema)**: A instalação do pacote `gdal` para Python pode ser complexa, pois depende de uma instalação prévia da biblioteca GDAL no seu sistema operacional.
  - **No Windows**: A maneira mais fácil é instalar via `pip` pacotes pré-compilados (wheels).
  - **No Linux (Ubuntu/Debian)**: Instale as dependências com `sudo apt-get install libgdal-dev g++`.
  - **No macOS**: Use `brew install gdal`.

### 2. Instalação

**a. Clone o repositório:**

```bash
git clone [https://github.com/vitoralvesdeoliveira/semantic-segmentation-for-land-use-classification.git](https://github.com/vitoralvesdeoliveira/semantic-segmentation-for-land-use-classification.git)
cd semantic-segmentation-for-land-use-classification
```

**b. Crie e ative um ambiente virtual (altamente recomendado):**

```bash
# Cria o ambiente
python -m venv venv

# Ativa o ambiente
# No Windows:
.\venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

**c. Instale as dependências a partir do `requirements.txt`:**

```bash
pip install -r requirements.txt
```

### 3. Estrutura de Diretórios

Para que o script funcione corretamente, seus dados brutos devem seguir a estrutura de pastas abaixo dentro do diretório `src/`:

```
.
├── src/
│   ├── dataset/
│   │   ├── raw/
│   │   │   ├── Quadras-AOI.tif  # Sua imagem raster principal
│   │   │   └── Lotes-AOI.shp    # Seu shapefile (incluindo .dbf, .prj, .shx etc.)
│   │   └── processed/         # Diretório de saída, será criado automaticamente
│   │
│   └── main.py                # Seu script principal
│
├── requirements.txt
└── README.md
```

### 4. Executando o Pipeline

Com o ambiente virtual ativado e os dados na estrutura correta, execute o script principal:

```bash
python src/main.py
```

O script irá:

1.  **Extrair** os lotes definidos na lista `fids_a_processar` (no exemplo, lotes 24 e 50).
2.  Salvar as imagens recortadas em `src/dataset/processed/png/`.
3.  **Pré-processar** essas imagens (redimensionar com padding para 256x256).
4.  Salvar o resultado final em `src/dataset/processed/preprocessed/`.

#### Para processar todos os lotes

Para processar todos os lotes do seu shapefile em vez de uma lista específica, modifique a seguinte linha no arquivo `src/main.py`:

De:

```python
extrator.extrair_lotes(fids_a_processar=[24,50])
```

Para:

```python
extrator.extrair_lotes() # Sem argumentos, processa todos os lotes
```

## Resultado Esperado

Após a execução, o diretório `src/dataset/processed` conterá:

- `png/`: Imagens dos lotes em seus tamanhos e proporções originais.
- `preprocessed/`: As mesmas imagens, mas padronizadas em 256x256 pixels com preenchimento, prontas para serem usadas em um modelo de IA.
