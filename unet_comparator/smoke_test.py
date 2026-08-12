# -*- coding: utf-8 -*-
"""
smoke_test.py — validates the comparator without training and without touching the GPU.

It forces CUDA_VISIBLE_DEVICES before importing torch: the YOLO grid may be using
the card, and an accidental `torch.cuda.init()` here would compete for memory in
the middle of a two-hour training run.

It checks, in this order:
  1. the three splits match the grid's partition (932 / 197 / 234)
  2. polygon rasterisation produces plausible masks, and the negatives (no label
     file) become an empty mask rather than an error
  3. the letterbox is reversible — undo(letterbox(m)) ≈ m. This is the check that
     matters: if it fails, the measured area comes out wrong, and area is the
     quantity the benchmark compares
  4. augmentation preserves the image↔mask correspondence
  5. the model accepts the input and returns the right shape
  6. one complete training pass on CPU (1 batch), to catch a shape or dtype error
     before scheduling 5 × 100 epochs
"""
import os
import sys

# "-1", not "": on Windows the empty string is ignored and the card stays visible.
# Found because the assert further down refused to run — which is why it exists.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # BEFORE importing torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch

from unet_data import (WoundDataset, carrega_splits, desfaz_letterbox, letterbox,
                       rasteriza, rotulo_de)
from unet_model import UNet, BCEDiceLoss

assert not torch.cuda.is_available(), "GPU visible — the guard failed, aborting"
print("GPU successfully hidden · running on CPU\n")

falhas = []


def check(cond, rot, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {rot}{('  ' + extra) if extra else ''}")
    if not cond:
        falhas.append(rot)


# ---------------------------------------------------------------- 1. splits
print("1. partition")
sp = carrega_splits("data.yaml")
ESPERADO = {"train": 932, "val": 197, "test": 234}
ds = {}
for k, n in ESPERADO.items():
    ds[k] = WoundDataset(sp[k], 640, treino=(k == "train"), fill=0, seed=42)
    check(len(ds[k]) == n, f"{k}: {len(ds[k])} images", f"(expected {n})")

# --------------------------------------------------------- 2. rasterisation
print("\n2. polygon rasterisation")
import cv2
vazias = nao_vazias = 0
areas = []
for f in ds["test"].arquivos:
    im = cv2.imread(f, cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    m = rasteriza(rotulo_de(f), w, h)
    if m.sum() == 0:
        vazias += 1
    else:
        nao_vazias += 1
        areas.append(100.0 * m.sum() / (h * w))
check(nao_vazias > 0 and vazias >= 0, f"test: {nao_vazias} with a mask, {vazias} empty")
check(all(0 < a < 100 for a in areas), f"area %: median {np.median(areas):.1f} · "
      f"min {min(areas):.1f} · max {max(areas):.1f}")
sem_rotulo = sum(1 for f in ds["test"].arquivos if not os.path.isfile(rotulo_de(f)))
check(True, f"images with no label file: {sem_rotulo} (treated as negatives)")

# ----------------------------------------------------- 3. reversible letterbox
print("\n3. reversible letterbox  (if this fails, the area comes out wrong)")
piores = []
for f in ds["test"].arquivos[:40]:
    im = cv2.imread(f, cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    m = rasteriza(rotulo_de(f), w, h)
    if m.sum() == 0:
        continue
    _, m640, lb = letterbox(im, m, 640, 0)
    volta = desfaz_letterbox(m640, lb)
    check_shape = volta.shape == m.shape
    # int() IS MANDATORY: summing a uint8 array returns an UNSIGNED integer, and
    # a1 - a0 wraps around when the area decreases — it gave a 1e16 % error.
    a0, a1 = int(m.sum()), int(volta.sum())
    err = abs(a1 - a0) / a0
    piores.append((err, os.path.basename(f), check_shape, m.shape, volta.shape))
piores.sort(reverse=True)
check(all(p[2] for p in piores), "restored shape = original shape")
check(piores[0][0] < 0.05,
      f"round-trip area error: worst {100*piores[0][0]:.2f}% · "
      f"median {100*np.median([p[0] for p in piores]):.2f}%",
      "(the loss comes from resampling 2452→640→2452; expected and small)")

# -------------------------------------------------- 4. coherent augmentation
print("\n4. augmentation preserves image↔mask")
x, y, nome, lb = ds["train"][0]
check(x.shape == (3, 640, 640) and y.shape == (1, 640, 640),
      f"shapes: x{tuple(x.shape)} y{tuple(y.shape)}")
check(0.0 <= float(x.min()) and float(x.max()) <= 1.0,
      f"image normalised [{float(x.min()):.2f}, {float(x.max()):.2f}]")
check(set(np.unique(y.numpy()).tolist()) <= {0.0, 1.0}, "binary mask {0,1}")

# THIS BLOCK WAS WRONG UNTIL 2026-07-28. It asserted "same sample, same output"
# as the desired property — and so CERTIFIED the defect that augmentation did not
# vary between epochs. Determinism has to be over (seed, epoch, index), not over
# (seed, index).
ds["train"].set_epoca(1)
a1 = ds["train"][0][0]
ds["train"].set_epoca(1)
a1b = ds["train"][0][0]
check(torch.equal(a1, a1b), "determinism: same epoch + same index = identical")

ds["train"].set_epoca(2)
a2 = ds["train"][0][0]
check(not torch.equal(a1, a2),
      "augmentation DOES vary between epochs (the defect the old test hid)")

ds["train"].set_epoca(1)
check(not torch.equal(ds["train"][0][0], ds["train"][1][0]), "distinct samples do differ")

# and augmentation must not vary with the training seed — the same as Ultralytics,
# otherwise the U-Net arm would carry one more source of variance than the YOLO arm
outro = WoundDataset(sp["train"], 640, treino=True, fill=0, seed=999)
outro.set_epoca(1)
check(torch.equal(a1, outro[0][0]),
      "augmentation is independent of the training seed (symmetry with the YOLO arm)")

# ------------------------------------------------------------- 5. modelo
print("\n5. model")
m = UNet()
npar = sum(p.numel() for p in m.parameters())
with torch.no_grad():
    saida = m(torch.zeros(1, 3, 640, 640))
check(tuple(saida.shape) == (1, 1, 640, 640), f"output {tuple(saida.shape)}",
      f"· {npar/1e6:.1f} M parameters")

# --------------------------------------------------- 6. one training pass
print("\n6. one training iteration (CPU, 1 batch of 2)")
crit = BCEDiceLoss()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
xb = torch.stack([ds["train"][i][0] for i in range(2)])
yb = torch.stack([ds["train"][i][1] for i in range(2)])
p0 = crit(m(xb), yb).item()
opt.zero_grad()
perda = crit(m(xb), yb)
perda.backward()
gnorm = sum(float(q.grad.norm()) for q in m.parameters() if q.grad is not None)
opt.step()
p1 = crit(m(xb), yb).item()
check(np.isfinite(p0) and np.isfinite(p1), f"finite loss: {p0:.4f} -> {p1:.4f}")
check(gnorm > 0, f"non-zero gradient (summed norm {gnorm:.1f})")
check(p1 < p0, "the loss fell after one step")

# the loss on an all-empty mask (negatives) must not become NaN
p_vazio = crit(m(xb), torch.zeros_like(yb)).item()
check(np.isfinite(p_vazio), f"loss on an empty mask = {p_vazio:.4f} (negatives do not break it)")

print("\n" + "=" * 66)
if falhas:
    print(f"{len(falhas)} CHECKS FAILED:")
    for f in falhas:
        print("   -", f)
    sys.exit(1)
print("Every check passed. The comparator is ready to run.")
print("Do NOT start before the YOLO grid finishes — run_unet_grid.py has a guard,")
print("but the guard is heuristic.")
