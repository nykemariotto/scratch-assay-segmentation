# -*- coding: utf-8 -*-
"""
stage3/regua_humana.py — do the automated methods reach the level at which the
observer agrees with themselves?

The right question is not "0.794 against 0.867", it is **where the ceiling is**. If
the observer does not reproduce their own delineation above ~0.86, no method can be
held to a higher standard: the limit stops being the architecture and becomes the
definition of the wound border.

TWO PRECAUTIONS, both mandatory:

1. The "empty over empty = 1.0" convention inflates both sides. Of the 14 blinded
   re-correction pairs, THREE have zero area on both passes and score 1.0 — they do
   not measure contour reproducibility. A fourth (`f8f76efe0c`) has an identical
   area on both passes and an IoU of exactly 1.0000, which for two independent
   manual corrections of a real wound is implausible: it looks like ROI reuse.

2. IoU depends strongly on wound size (0.36 in the 2-5% band, 0.90 in the 10-20%
   band). The re-correction pairs have areas between 2.5% and 22%, a different
   distribution from the test set as a whole. Comparing without matching the band
   would compare difficulties, not methods.

This script reports both sides with and without the degenerate pairs, and restricts
the automated methods to the same area band.

    python stage3/regua_humana.py
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


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ═══════════════════════════ 1 · o observador contra si mesmo
r = [x for x in csv.DictReader(open("data/correction_agreement.csv", encoding="utf-8-sig"))
     if x.get("iou_pass1_vs_pass2", "").strip() not in ("", "nan")]
pares = [{"base": x["base"], "a1": f(x["area_pct_pass1"]),
          "a2": f(x["area_pct_pass2"]), "iou": f(x["iou_pass1_vs_pass2"])} for x in r]

vazio = [p for p in pares if p["a1"] == 0 and p["a2"] == 0]
identico = [p for p in pares if p not in vazio and p["iou"] >= 1.0 - 1e-9]
reais = [p for p in pares if p not in vazio]
limpos = [p for p in reais if p not in identico]

print("=" * 72)
print("1 · THE OBSERVER AGAINST THEMSELVES (blinded re-correction)")
print("=" * 72)
for rot, conj in (("all pairs", pares),
                  ("excluding empty↔empty", reais),
                  ("also excluding IoU exactly 1.0000", limpos)):
    v = [p["iou"] for p in conj]
    print(f"  {rot:<38} n={len(v):>2}  median {st.median(v):.4f}  "
          f"mean {st.mean(v):.4f}  min {min(v):.4f}")
print(f"\n  empty↔empty: {len(vazio)} pairs, all IoU 1.0 by convention")
for p in identico:
    print(f"  IoU exactly 1.0000 with a real area: {p['base'][:34]} "
          f"({p['a1']:.3f}% on both passes)")
print("\n  The manuscript reported a median of 0.894 and a mean of 0.878 — the numbers")
print("  in the first row, which include the three empty↔empty pairs.")

FAIXA_LO = min(p["a1"] for p in limpos)
FAIXA_HI = max(p["a1"] for p in limpos)
print(f"\n  area band of the clean pairs: {FAIXA_LO:.2f}% to {FAIXA_HI:.2f}%")

# ═══════════════════════════ 2 · os métodos, na MESMA faixa
print("\n" + "=" * 72)
print("2 · THE AUTOMATED METHODS, RESTRICTED TO THE SAME AREA BAND")
print("=" * 72)

p_iou = os.path.join("stage3", "iou_per_image.csv")
if not os.path.isfile(p_iou):
    sys.exit(f"could not find {p_iou}")
linhas = list(csv.DictReader(open(p_iou, encoding="utf-8")))

d = carrega_splits("data.yaml")["test"]
imagens = sorted(f_ for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                 for f_ in glob.glob(os.path.join(d, e)))
area = {}
for f_ in imagens:
    im = cv2.imread(f_, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    area[os.path.basename(f_)] = 100.0 * float(rasteriza(rotulo_de(f_), w, h).sum()) / (h * w)

PONTOS = [("YOLO M", "conf=0.8"), ("U-Net", "sigmoid=0.5")]
print(f"{'set':<34} {'n/seed':>7} " + "".join(f"{b:>13}" for b, _ in PONTOS))
print("-" * 72)
for rot, filtro in (
        ("whole test set (234)", lambda a: True),
        ("with a wound only (area > 0)", lambda a: a > 0),
        (f"observer band ({FAIXA_LO:.1f}–{FAIXA_HI:.1f}%)",
         lambda a: FAIXA_LO <= a <= FAIXA_HI)):
    cel, n = [], 0
    for k in PONTOS:
        v = [float(L["iou"]) for L in linhas
             if (L["braco"], L["ponto"]) == k and filtro(area[L["arquivo"]])]
        n = len(v) // 5
        cel.append(f"{st.mean(v):.4f}" if v else "—")
    print(f"{rot:<34} {n:>7} " + "".join(f"{c:>13}" for c in cel))

print(f"""
HOW TO READ THIS. The comparison that counts is the last row against the third row
of section 1 — the same area band, and neither side inflated by the empty-mask
convention.

If the methods reach the observer's level, the limit of the task is not the
architecture: it is the definition of the wound border, which not even the human
reproduces. That is a much stronger claim — and a more useful one to the reader —
than any ranking between models.

A caveat to declare: n = {len(limpos)} pairs from a single observer. The interval
is wide and the result is indicative, not a precise estimate of the ceiling.""")
