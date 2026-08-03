# -*- coding: utf-8 -*-
"""
stage2/verify_padding.py — verifies AT PIXEL LEVEL that the padding patch works
on both paths (training with mosaic, validation with letterbox). It does not
trust the patch: it measures.

For each mode it builds the real Ultralytics dataloader and measures the
histogram of the border pixels of the batches.
"""
import sys
import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "black"

import padding_patch
val = padding_patch.apply(MODE)
print(f"mode={MODE}  fill value={val}\n")

from ultralytics.data.build import build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
import ultralytics.data.augment as A

cfg = get_cfg(DEFAULT_CFG)
cfg.imgsz = 640
cfg.mosaic = 1.0
cfg.task = "segment"

# ---------- TRAINING path (augment=True -> Mosaic + RandomPerspective) ----------
ds_tr = build_yolo_dataset(cfg, "dataset/images/train", 4, {"names": {0: "wound"}, "channels": 3},
                           mode="train", rect=False)
ds_tr.use_segments = True
sample = ds_tr[0]
img = sample["img"]
arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
if arr.ndim == 3 and arr.shape[0] in (1, 3):
    arr = np.transpose(arr, (1, 2, 0))
print(f"TRAIN   shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}")
vals, cnts = np.unique(arr, return_counts=True)
top = sorted(zip(cnts, vals), reverse=True)[:5]
print(f"  most frequent values: {[(int(v), int(c)) for c, v in top]}")

# ---------- VAL path (augment=False -> LetterBox) ----------
ds_va = build_yolo_dataset(cfg, "dataset/images/val", 4, {"names": {0: "wound"}, "channels": 3},
                           mode="val", rect=False)
s2 = ds_va[0]
a2 = s2["img"]
a2 = a2.numpy() if hasattr(a2, "numpy") else np.asarray(a2)
if a2.ndim == 3 and a2.shape[0] in (1, 3):
    a2 = np.transpose(a2, (1, 2, 0))
print(f"\nVAL     shape={a2.shape} dtype={a2.dtype}")
# the padding bars sit at the top/bottom (2452x2056 images -> AR 1.19)
top_rows = a2[:12].reshape(-1, a2.shape[-1])
bot_rows = a2[-12:].reshape(-1, a2.shape[-1])
print(f"  mean of the 12 TOP rows    : {top_rows.mean(axis=0).round(1)}")
print(f"  mean of the 12 BOTTOM rows : {bot_rows.mean(axis=0).round(1)}")
uniq_t = np.unique(top_rows)
print(f"  unique values at the top (max 8): {uniq_t[:8]}")

esperado = val
ok_val = abs(float(top_rows.mean()) - esperado) < 3
print(f"\n  expected={esperado}  -> VAL padding {'OK' if ok_val else 'DIVERGENT'}")
