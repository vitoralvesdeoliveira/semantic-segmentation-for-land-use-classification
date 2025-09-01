import os 
import cv2 as cv
import numpy as np
import json
from pathlib import Path


def resize_with_padding(target_size, img, interpolation=cv.INTER_LINEAR):
    """Redimensiona mantendo aspecto e adicionando padding"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    resized = cv.resize(img, (int(w*scale), int(h*scale)), interpolation=interpolation)
    h_pad = target_size - resized.shape[0]
    w_pad = target_size - resized.shape[1]
    top = h_pad // 2
    bottom = h_pad - top
    left = w_pad // 2
    right = w_pad - left
    padded = cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0,0,0])
    return padded


def salvar_array_em_txt(array: np.ndarray, caminho_arquivo_saida: str):
            """
            Salva um array RGB 3D (altura, largura, 3 canais) em um arquivo .txt,
            com 4 casas decimais por valor.
            """
            with open(caminho_arquivo_saida, 'w') as f:
                for linha in array:
                    for pixel in linha:
                        f.write(' '.join(f"{valor:.4f}" for valor in pixel))  # RGB
                        f.write('\n')

def salvar_array_em_txt_2d(array: np.ndarray, caminho_arquivo_saida: str):
    with open(caminho_arquivo_saida, 'w') as f:
        for linha in array:
            f.write(' '.join(f"{valor:.4f}" for valor in linha))
            f.write('\n')

cwd = os.getcwd()

path = Path(os.path.abspath(os.path.join(cwd, 'dataset','processed','masks')))

COLOR_MAP = {
    (0, 0, 0): 0,         # unknown
    (0, 255, 0): 1,       # pastagem
    (255, 0, 0): 2,       # agricultura
    (0, 0, 255): 3,       # água
    (128, 128, 128): 4,   # edificação
    (128, 0, 0): 5,       # indústria
    (0, 128, 0): 6        # floresta
}

for idx, subfolder in enumerate(path.iterdir()):
      if subfolder.is_dir():
        label_path = subfolder / "label.png"
        img_path = subfolder / "img.png"
        if idx == 0:
            if img_path.exists():
                img_bgr = cv.imread(img_path)
                salvar_array_em_txt(img_bgr,'img-bgr.txt')
                img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
                salvar_array_em_txt(img_rgb,'img-rgb.txt')
                print(img_bgr.dtype)
                img_rgb_normalized = img_rgb.astype(np.float32) / 255.0
                salvar_array_em_txt(img_rgb_normalized,'img-normalized.txt')
                img_preprocessed = resize_with_padding(256,img_rgb_normalized)
                # np.save(f"dataset/images/{subfolder.name}.npy", img_preprocessed)
                # np.save(f"dataset/masks/{subfolder.name}.npy", mask_class)
            else:
                print(f"Nenhum 'img.png' encontrado em: {subfolder}")
            if label_path.exists():
                mask_bgr = cv.imread(label_path)
                mask_rgb = cv.cvtColor(mask_bgr,cv.COLOR_BGR2RGB)
                mask_preprocessed = resize_with_padding(256,mask_rgb,interpolation=cv.INTER_NEAREST)
                altura, largura = mask_preprocessed.shape[:2]
                mask_class = np.zeros((altura, largura), dtype=np.uint8)
                for rgb, class_id in COLOR_MAP.items():
                     match = np.all(mask_preprocessed == rgb, axis=-1)
                     mask_class[match] = class_id
                # salvar_array_em_txt_2d(mask_class,'maskk.txt')
                salvar_array_em_txt_2d(mask_preprocessed,'maskk.txt')
            else:
                print(f"Nenhum 'label.png' encontrado em: {subfolder}")
            