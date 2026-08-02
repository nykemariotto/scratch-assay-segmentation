# -*- coding: utf-8 -*-
"""
unet_data.py — dataset da U-Net sobre EXATAMENTE a mesma partição da grade.

CONTRATO DE JUSTIÇA (é o que torna a comparação válida; se algum item quebrar,
a comparação vira propaganda):
  1. mesma partição — lê o `data.yaml` da grade, não uma cópia
  2. mesma resolução e mesmo letterbox (640, padding preto = 0)
  3. mesma augmentation ATIVA na grade: HSV 0.015/0.7/0.4, translate 0.1,
     scale 0.5, fliplr 0.5 — e nada além disso
  4. mesmas máscaras: rasterizadas dos MESMOS polígonos YOLO-seg que supervisionam
     o YOLO11, não de uma anotação paralela

ASSIMETRIA DECLARADA: a grade usa mosaic (1.0, desligado nas últimas 10 épocas).
Mosaic é augmentation de detecção — cola 4 imagens e recorta — e não faz parte de
nenhum pipeline padrão de U-Net. Aplicá-lo aqui seria inventar; omiti-lo é a
escolha honesta, e tem de ser declarada no Methods. É a única diferença de
augmentation entre os dois braços.

FORMATO DOS RÓTULOS. YOLO-seg: uma linha por polígono,
`classe x1 y1 x2 y2 …` normalizado pela largura/altura DA PRÓPRIA IMAGEM. Como o
dataset tem imagens em 2452×2056 e em 640×640, a desnormalização usa o tamanho
real de cada arquivo — usar um tamanho fixo produziria máscaras deslocadas
justamente no subconjunto center-crop (D7).

Arquivo de rótulo ausente ou vazio = negativo (imagem sem ferida anotada). São
150 no dataset e entram com máscara toda zero, não são descartados.
"""
import glob
import os
import random

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# augmentation espelhando os defaults ativos do Ultralytics nesta grade
HSV_H, HSV_S, HSV_V = 0.015, 0.7, 0.4
TRANSLATE, SCALE, FLIPLR = 0.1, 0.5, 0.5

# SEMENTE DA AUGMENTATION — constante, NAO o seed do treino.
#
# Isto espelha o que o Ultralytics faz e o que o D13 documentou: la, as sementes
# dos workers derivam de um gerador fixo (6148914691236517205 + RANK), entao a
# ordem dos dados e a augmentation NAO variam com o seed de treino; so a
# inicializacao dos pesos varia.
#
# Se aqui a augmentation variasse com o seed, o braco U-Net teria uma fonte de
# variancia a mais que o braco YOLO, e o desvio-padrao entre seeds dos dois nao
# seria comparavel — justamente o numero que a Table 2 poe lado a lado.
SEED_AUG = 20260728



def carrega_splits(data_yaml):
    # `path` relativo resolve contra o diretorio DO PROPRIO YAML, nao contra o
    # CWD: um script rodado de outra pasta encontraria o dataset errado ou
    # nenhum, e a falha silenciosa (split vazio) e pior que a barulhenta.
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    aqui = os.path.dirname(os.path.abspath(data_yaml))
    raiz = d.get("path") or aqui
    if not os.path.isabs(raiz):
        raiz = os.path.join(aqui, raiz)
    out = {}
    for k in ("train", "val", "test"):
        if k not in d:
            continue
        p = d[k] if os.path.isabs(d[k]) else os.path.join(raiz, d[k])
        out[k] = os.path.normpath(p)
    return out


def rotulo_de(img_path):
    """dataset/images/<split>/x.png -> dataset/labels/<split>/x.txt"""
    d, nome = os.path.split(img_path)
    d = d.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    d = d.replace("/images/", "/labels/")
    return os.path.join(d, os.path.splitext(nome)[0] + ".txt")


def rasteriza(label_path, w, h):
    """polígonos YOLO-seg -> máscara binária uint8 {0,1} no tamanho da imagem."""
    m = np.zeros((h, w), np.uint8)
    if not os.path.isfile(label_path):
        return m
    for linha in open(label_path, encoding="utf-8"):
        v = linha.split()
        if len(v) < 7:                      # classe + ao menos 3 vértices
            continue
        c = np.asarray(v[1:], dtype=np.float64)
        if c.size % 2:
            c = c[:-1]
        pts = c.reshape(-1, 2) * np.asarray([w, h])
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m


def letterbox(img, mask, alvo=640, fill=0):
    """redimensiona preservando aspecto e preenche até alvo x alvo.

    Devolve tambem os parametros, para que a predicao possa ser desfeita de volta
    ao espaco da imagem original — sem isso a area em pixels sai errada, que e
    exatamente a grandeza que o benchmark compara.
    """
    h, w = img.shape[:2]
    r = min(alvo / h, alvo / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    top, left = (alvo - nh) // 2, (alvo - nw) // 2
    out_i = np.full((alvo, alvo, img.shape[2]), fill, np.uint8)
    out_m = np.zeros((alvo, alvo), np.uint8)
    out_i[top:top + nh, left:left + nw] = img
    out_m[top:top + nh, left:left + nw] = mask
    return out_i, out_m, {"r": r, "top": top, "left": left, "nh": nh, "nw": nw,
                          "orig_h": h, "orig_w": w}


def desfaz_letterbox(mask640, lb):
    """máscara 640x640 -> máscara no tamanho original da imagem."""
    rec = mask640[lb["top"]:lb["top"] + lb["nh"], lb["left"]:lb["left"] + lb["nw"]]
    return cv2.resize(rec, (lb["orig_w"], lb["orig_h"]), interpolation=cv2.INTER_NEAREST)


def _hsv(img, rng):
    g = rng.uniform(-1, 1, 3) * np.asarray([HSV_H, HSV_S, HSV_V]) + 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    x = np.arange(256, dtype=np.int16)
    lut = np.stack([((x * g[0]) % 180).astype(np.uint8),
                    np.clip(x * g[1], 0, 255).astype(np.uint8),
                    np.clip(x * g[2], 0, 255).astype(np.uint8)])
    hsv = np.stack([lut[i][hsv[..., i]] for i in range(3)], axis=-1)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _afim(img, mask, rng, alvo, fill):
    s = rng.uniform(1 - SCALE, 1 + SCALE)
    tx = rng.uniform(-TRANSLATE, TRANSLATE) * alvo
    ty = rng.uniform(-TRANSLATE, TRANSLATE) * alvo
    c = alvo / 2
    M = np.asarray([[s, 0, c - s * c + tx], [0, s, c - s * c + ty]], np.float32)
    img = cv2.warpAffine(img, M, (alvo, alvo), flags=cv2.INTER_LINEAR,
                         borderValue=(fill, fill, fill))
    mask = cv2.warpAffine(mask, M, (alvo, alvo), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, mask


class WoundDataset(Dataset):
    """`epoca` PRECISA ser atualizada a cada época pelo laço de treino.

    DEFEITO CORRIGIDO (revisao 2026-07-28). A versao anterior semeava o RNG com
    `(seed, indice)` apenas — sem epoca. Consequencia: cada imagem recebia UMA
    transformacao fixa e a repetia nas 100 epocas. Isso nao e augmentation
    estocastica, e um dataset transformado uma unica vez — e quebrava o contrato
    de justica, porque o YOLO sorteia transformacao nova a cada epoca.

    O smoke test da epoca certificava o defeito: afirmava "mesma amostra, mesma
    saida" como se fosse a propriedade desejada. Determinismo tem de ser sobre
    (semente, epoca, indice), nao sobre (semente, indice).
    """

    def __init__(self, pasta, alvo=640, treino=False, fill=0, seed=0):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
        self.arquivos = sorted(f for e in exts for f in glob.glob(os.path.join(pasta, e)))
        if not self.arquivos:
            raise RuntimeError(f"nenhuma imagem em {pasta}")
        self.alvo, self.treino, self.fill = alvo, treino, fill
        self.seed = seed
        self.epoca = 0

    def set_epoca(self, e):
        """Chamar ANTES de criar o iterador da época.

        Com num_workers>0 e persistent_workers=False o dataset e re-picklado a
        cada epoca, entao o atributo atualizado viaja para os workers.
        """
        self.epoca = int(e)

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, i):
        f = self.arquivos[i]
        img = cv2.imread(f, cv2.IMREAD_COLOR)     # força 3 canais (L e RGB no dataset)
        if img is None:
            raise RuntimeError(f"cv2 nao leu {f}")
        h, w = img.shape[:2]
        m = rasteriza(rotulo_de(f), w, h)
        img, m, lb = letterbox(img, m, self.alvo, self.fill)

        if self.treino:
            # (SEED_AUG, epoca, indice): varia entre epocas, e NAO varia com o
            # seed de treino — igual ao Ultralytics (ver D13 e SEED_AUG acima).
            rng = np.random.default_rng((SEED_AUG, self.epoca, i))
            img = _hsv(img, rng)
            img, m = _afim(img, m, rng, self.alvo, self.fill)
            if rng.random() < FLIPLR:
                img, m = img[:, ::-1].copy(), m[:, ::-1].copy()

        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255)
        y = torch.from_numpy(m.copy()).float().unsqueeze(0)
        return x, y, os.path.basename(f), lb
