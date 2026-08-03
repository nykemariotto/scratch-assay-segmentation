# -*- coding: utf-8 -*-
"""
stage4/classify_failure_mode.py — refines the binary over/under triage into FAILURE
MODES, without requiring a new visual inspection.

MOTIVATION (observed by the operator during the triage): the "over" box ended up
grouping three mechanically distinct situations:
  (a) real excess      — the mask contains the wound plus more (multiplicative bias)
  (b) displaced mask   — the mask is OUTSIDE the wound (localisation error)
  (c) spurious mask    — the wound has already closed (real area ~ 0) and WHST
                         marked something anyway
This matters because the bias-cancellation test assumes
measured_area = k * real_area, a model that only holds in (a): in (b) there is no
relation to the truth, and in (c) k is undefined.

METRIC: CONTAINMENT = |M_t ∩ M_0| / |M_t|
  the fraction of the mask at t>0 that falls inside the wound's footprint at t0.
  Physical basis: in a scratch assay the wound only NARROWS and stays in the same
  place; therefore the mask at t>0 should be contained in the one at t0.

DISCRIMINANT VALIDITY (measured, over series whose t0 was classified OK):
  images scored OK    -> median containment 0.92  (5% below 0.5)
  images scored 'over'-> median containment 0.42  (66% below 0.5)
  The high containment of the OK images also rules out field drift (FOV shift) as
  the dominant explanation: if the field moved between timepoints, the OK images
  would show low containment too.

CLASSIFICATION (only for frames scored SEG_RUIM with computable containment):
  containment >= 0.5                     -> excesso
  containment <  0.5 and area_pct <  5.0 -> espuria_fechada
  containment <  0.5 and area_pct >= 5.0 -> deslocada
Frames with no t0 in the series, or t0 itself, stay 'nao_avaliavel'.

It does NOT change the triage category/subtype nor the correction list: it appends
the columns 'contencao_t0' and 'modo_falha' to data/inspecao_visual.csv.
"""
import csv, os
import numpy as np
from PIL import Image
from collections import defaultdict, Counter

HUM = "data/inspecao_visual.csv"
AUTO = "data/whst_pass1_qc.csv"
MASKS = os.path.join("whst_output", "masks")
THR_CONT, THR_AREA = 0.5, 5.0

hum = list(csv.DictReader(open(HUM, encoding="utf-8-sig")))
humk = {r["whst_input_file"]: r for r in hum}
auto = {r["whst_input_file"]: r for r in csv.DictReader(open(AUTO, encoding="utf-8-sig"))}


def base(f):
    for e in (".tiff", ".tif"):
        if f.lower().endswith(e):
            return f[: -len(e)]
    return os.path.splitext(f)[0]


def mask(f):
    p = os.path.join(MASKS, base(f) + "_mask.png")
    if not os.path.exists(p):
        return None
    return np.asarray(Image.open(p).convert("L")) > 127


by_ser = defaultdict(list)
for k in humk:
    by_ser[auto[k]["series_key"]].append((int(auto[k]["timepoint_h"]), k))

cont = {}
sem_ref = 0
for sk, v in by_ser.items():
    b = [x for x in v if x[0] == 0]
    m0 = mask(b[0][1]) if b else None
    if m0 is None or m0.sum() == 0:
        sem_ref += sum(1 for tp, _ in v if tp > 0)
        continue
    for tp, k in v:
        if tp == 0:
            continue
        m = mask(k)
        if m is None or m.sum() == 0 or m.shape != m0.shape:
            continue
        cont[k] = float(np.logical_and(m, m0).sum()) / float(m.sum())


def modo(r):
    k = r["whst_input_file"]
    if r["categoria"] != "SEG_RUIM":
        return ""
    if k not in cont:
        return "nao_avaliavel"
    c = cont[k]
    if c >= THR_CONT:
        return "excesso"
    return "espuria_fechada" if float(auto[k]["area_pct"]) < THR_AREA else "deslocada"


for r in hum:
    k = r["whst_input_file"]
    r["contencao_t0"] = f"{cont[k]:.4f}" if k in cont else ""
    r["modo_falha"] = modo(r)

with open(HUM, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hum[0].keys()))
    w.writeheader(); w.writerows(hum)

print(f"containment computed for {len(cont)} images at t>0 "
      f"({sem_ref} sem t0 de referencia na serie)")
seg = [r for r in hum if r["categoria"] == "SEG_RUIM"]
c = Counter(r["modo_falha"] for r in seg)
print(f"\n=== FAILURE MODE among the {len(seg)} SEG_RUIM images ===")
for m, n in sorted(c.items(), key=lambda x: -x[1]):
    print(f"  {m:<18} {n:>4}  ({n/len(seg):.0%})")
av = [r for r in seg if r["modo_falha"] not in ("", "nao_avaliavel")]
if av:
    ca = Counter(r["modo_falha"] for r in av)
    print(f"\n  among the {len(av)} EVALUABLE ones:")
    for m, n in sorted(ca.items(), key=lambda x: -x[1]):
        print(f"    {m:<18} {n:>4}  ({n/len(av):.0%})")
print(f"\nSaved: {HUM} (+ contencao_t0, modo_falha)")
