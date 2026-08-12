# -*- coding: utf-8 -*-
"""
stage3/estratifica_bracos.py — WHERE each arm fails, not HOW MUCH.

An average over 234 images (or 97 observations) hides regime. The hypothesis is
concrete and testable: at 0 h the wound is large and both arms get it right; at the
late timepoints it is small, and that is where everything falls apart — including
the U-Net's false positives in an already closed well.

Two stratifications, over data that already exists:

  A. Mask IoU by WOUND SIZE (the reference standard rasterised at the original
     resolution). Source: stage3/iou_per_image.csv
  B. Closure-fraction error by TIMEPOINT and by CELL LINE.
     Source: stage3/cmp_*_paired_new_long.csv + data/whst_series_analysis.csv

The closure error is reported WITH ITS SIGN, not in absolute value: the sign is the
information. Positive = the method reports MORE closure than the reference standard.

    python stage3/estratifica_bracos.py
"""
import csv
import glob
import os
import statistics as st
import sys
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(RAIZ, "unet_comparator"))
os.chdir(RAIZ)

import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402

from unet_data import carrega_splits, rasteriza, rotulo_de   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEEDS = [42, 43, 44, 45, 46]


def faixa_area(pct):
    if pct == 0:
        return "0 · fechada"
    for lo, hi, rot in ((0, 2, "1 · <2%"), (2, 5, "2 · 2-5%"), (5, 10, "3 · 5-10%"),
                        (10, 20, "4 · 10-20%")):
        if lo < pct <= hi:
            return rot
    return "5 · >20%"


# ══════════════════════════════ A · IoU by wound size
print("=" * 74)
print("A · MASK IoU BY WOUND SIZE (reference standard)")
print("=" * 74)

p_iou = os.path.join("stage3", "iou_per_image.csv")
if not os.path.isfile(p_iou):
    sys.exit(f"could not find {p_iou} — run stage3/iou_per_image.py")
linhas = list(csv.DictReader(open(p_iou, encoding="utf-8")))

d = carrega_splits("data.yaml")["test"]
imagens = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                 for f in glob.glob(os.path.join(d, e)))
area_pct = {}
for f in imagens:
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    m = rasteriza(rotulo_de(f), w, h)
    area_pct[os.path.basename(f)] = 100.0 * float(m.sum()) / (h * w)
print(f"{len(area_pct)} images · wound area rasterised at the original resolution\n")

# the native operating points only, which is what is used in practice
PONTOS = [("YOLO M", "conf=0.8"), ("U-Net", "sigmoid=0.5")]
por = defaultdict(lambda: defaultdict(list))
for L in linhas:
    k = (L["braco"], L["ponto"])
    if k not in PONTOS:
        continue
    por[k][faixa_area(area_pct[L["arquivo"]])].append(float(L["iou"]))

faixas = sorted({f for k in por for f in por[k]})
print(f"{'area band':<14} {'n/seed':>7} " +
      "".join(f"{b:>16}" for b, _ in PONTOS))
print("-" * 74)
for fx in faixas:
    n = len(por[PONTOS[0]][fx]) // len(SEEDS)
    cel = []
    for k in PONTOS:
        v = por[k][fx]
        cel.append(f"{st.mean(v):.4f}" if v else "—")
    print(f"{fx:<14} {n:>7} " + "".join(f"{c:>16}" for c in cel))

# ══════════════════════════════ B · closure error by timepoint and cell line
print("\n" + "=" * 74)
print("B · CLOSURE-FRACTION ERROR (method − reference), SIGNED")
print("=" * 74)

cel_de = {}
for r in csv.DictReader(open("data/whst_series_analysis.csv", encoding="utf-8-sig")):
    cel_de[r["series_key"]] = r["cell_line"]

BRACOS = {"YOLO M": "yolo11m-seg_black_coco_seed{}", "U-Net": "unet_black_seed{}"}
obs = defaultdict(lambda: defaultdict(list))       # (braço, estrato) -> difs
chaves_por_run = {}
for braco, molde in BRACOS.items():
    for s in SEEDS:
        p = os.path.join("stage3", f"cmp_{molde.format(s)}_paired_new_long.csv")
        if not os.path.isfile(p):
            continue
        r = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        chaves_por_run[(braco, s)] = {
            (x["series_key"], x["campo"], x["timepoint_h"]): x for x in r}
comuns = None
for d_ in chaves_por_run.values():
    comuns = set(d_) if comuns is None else (comuns & set(d_))
print(f"{len(comuns)} observations common to all {len(chaves_por_run)} runs\n")

for (braco, s), d_ in chaves_por_run.items():
    for k in comuns:
        x = d_[k]
        dif = float(x["ai"]) - float(x["referencia"])
        obs[braco][f"t = {int(k[2]):>2} h"].append(dif)
        obs[braco][cel_de.get(k[0], "?")].append(dif)
        obs[braco]["ALL"].append(dif)

for titulo, chaves in (("by timepoint", sorted(k for k in obs["YOLO M"] if k.startswith("t ="))),
                       ("by cell line", sorted(k for k in obs["YOLO M"]
                                               if k in ("HUVEC", "SKOV-3"))),
                       ("total", ["ALL"])):
    print(f"── {titulo} ──")
    print(f"{'stratum':<12} {'n/seed':>7} {'YOLO M':>20} {'U-Net':>20}")
    for k in chaves:
        n = len(obs["YOLO M"][k]) // len(SEEDS)
        cel = []
        for b in ("YOLO M", "U-Net"):
            v = obs[b][k]
            cel.append(f"{st.mean(v):+.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else "—")
        print(f"{k:<12} {n:>7} {cel[0]:>20} {cel[1]:>20}")
    print()

print("""HOW TO READ THIS. The sign matters: positive = the method reports MORE
closure than the reference standard. If the bias grows with time, the cause is
under-segmentation of the small wound — and the error runs in the direction the
experiment wants to see success.

The ± is the standard deviation of the differences within the stratum (dispersion
between observations), not the standard error of the estimator.""")
