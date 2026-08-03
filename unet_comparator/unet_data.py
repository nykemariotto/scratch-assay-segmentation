# -*- coding: utf-8 -*-
"""
unet_data.py — the U-Net dataset over EXACTLY the same partition as the grid.

FAIRNESS CONTRACT (this is what makes the comparison valid; if any item breaks, the
comparison becomes advocacy):
  1. same partition — it reads the grid's `data.yaml`, not a copy
  2. same resolution and same letterbox (640, black padding = 0)
  3. the same augmentation ACTIVE in the grid: HSV 0.015/0.7/0.4, translate 0.1,
     scale 0.5, fliplr 0.5 — and nothing beyond that
  4. the same masks: rasterised from the SAME YOLO-seg polygons that supervise
     YOLO11, not from a parallel annotation

DECLARED ASYMMETRY: the grid uses mosaic (1.0, switched off for the last 10 epochs).
Mosaic is a detection augmentation — it tiles 4 images and crops — and is not part
of any standard U-Net pipeline. Applying it here would be inventing a method;
omitting it is the honest choice, and it has to be declared in the Methods. It is
the only augmentation difference between the two arms.

LABEL FORMAT. YOLO-seg: one line per polygon, `class x1 y1 x2 y2 …` normalised by
the width/height OF THE IMAGE ITSELF. Since the dataset holds images at 2452×2056
and at 640×640, the denormalisation uses each file's real size — using a fixed size
would produce displaced masks precisely on the centre-crop subset.

A missing or empty label file means a negative (an image with no annotated wound).
There are 150 in the dataset; they enter with an all-zero mask and are not dropped.
"""
import glob
import os
import random

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# augmentation mirroring the Ultralytics defaults active in this grid
HSV_H, HSV_S, HSV_V = 0.015, 0.7, 0.4
TRANSLATE, SCALE, FLIPLR = 0.1, 0.5, 0.5

# AUGMENTATION SEED — a constant, NOT the training seed.
#
# This mirrors what Ultralytics does: there, the worker seeds derive from a fixed
# generator (6148914691236517205 + RANK), so data order and augmentation do NOT vary
# with the training seed; only weight initialisation varies.
#
# If augmentation varied with the seed here, the U-Net arm would carry one more
# source of variance than the YOLO arm, and the seed-to-seed standard deviation of
# the two would not be comparable — precisely the number Table 2 puts side by side.
SEED_AUG = 20260728



def carrega_splits(data_yaml):
    # a relative `path` resolves against the YAML'S OWN directory, not against the
    # CWD: a script run from another folder would find the wrong dataset or none, and
    # the silent failure (an empty split) is worse than the noisy one.
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
    """YOLO-seg polygons -> a binary uint8 {0,1} mask at the image size."""
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
    """resizes preserving aspect ratio and pads up to target x target.

    It also returns the parameters, so that the prediction can be undone back into
    the original image space — without that the area in pixels comes out wrong,
    which is precisely the quantity the benchmark compares.
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
    """640x640 mask -> mask at the original image size."""
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
    """`epoca` MUST be updated every epoch by the training loop.

    DEFECT FIXED (review of 2026-07-28). The previous version seeded the RNG with
    `(seed, index)` only — no epoch. Consequence: each image received ONE fixed
    transform and repeated it across all 100 epochs. That is not stochastic
    augmentation, it is a dataset transformed exactly once — and it broke the
    fairness contract, because YOLO draws a new transform every epoch.

    The smoke test of the time certified the defect: it asserted "same sample, same
    output" as though that were the desired property. Determinism has to hold over
    (seed, epoch, index), not over (seed, index).
    """

    def __init__(self, pasta, alvo=640, treino=False, fill=0, seed=0):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
        self.arquivos = sorted(f for e in exts for f in glob.glob(os.path.join(pasta, e)))
        if not self.arquivos:
            raise RuntimeError(f"no image in {pasta}")
        self.alvo, self.treino, self.fill = alvo, treino, fill
        self.seed = seed
        self.epoca = 0

    def set_epoca(self, e):
        """Call BEFORE creating the epoch's iterator.

        With num_workers>0 and persistent_workers=False the dataset is re-pickled
        every epoch, so the updated attribute travels to the workers.
        """
        self.epoca = int(e)

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, i):
        f = self.arquivos[i]
        img = cv2.imread(f, cv2.IMREAD_COLOR)     # forces 3 channels (L and RGB both occur)
        if img is None:
            raise RuntimeError(f"cv2 could not read {f}")
        h, w = img.shape[:2]
        m = rasteriza(rotulo_de(f), w, h)
        img, m, lb = letterbox(img, m, self.alvo, self.fill)

        if self.treino:
            # (SEED_AUG, epoch, index): varies across epochs, and does NOT vary with
            # the training seed — same as Ultralytics (see SEED_AUG above).
            rng = np.random.default_rng((SEED_AUG, self.epoca, i))
            img = _hsv(img, rng)
            img, m = _afim(img, m, rng, self.alvo, self.fill)
            if rng.random() < FLIPLR:
                img, m = img[:, ::-1].copy(), m[:, ::-1].copy()

        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255)
        y = torch.from_numpy(m.copy()).float().unsqueeze(0)
        return x, y, os.path.basename(f), lb
