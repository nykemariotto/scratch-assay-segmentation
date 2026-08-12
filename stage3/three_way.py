
# -*- coding: utf-8 -*-
"""
stage3/three_way.py — classical × YOLO × U-Net, on the same outcome and the same set.

`stage3/benchmark_classical_long.csv` carries the closure fraction of the AUTOMATIC
WHST against the reference standard — the same quantity and the same reference that
`stage3/compara_bracos_closure.py` uses for the deep models. That makes a
three-way comparison possible.

⚠️ INCORPORATION BIAS — READ BEFORE INTERPRETING
The reference standard was built by **manually correcting the output of the
automatic WHST**. That is: the classical arm is compared against a standard that
IS, in part, its own output. In the frames the observer did not need to correct,
the two sides are identical by construction, and the classical arm's agreement
rises without it having got anything right.

The deep models do not have that property: they never saw the reference standard,
neither in training nor in its construction. The comparison is therefore
**unfavourable to them by design** — which makes any advantage of theirs stronger,
and any advantage of the classical arm uninterpretable.

It is the trap STARD calls incorporation bias.

    python stage3/three_way.py
"""
import csv
import math
import os
import statistics as st
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(AQUI))
sys.path.insert(0, AQUI)

import numpy as np                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEEDS = [42, 43, 44, 45, 46]
BRACOS = {"YOLO M": "yolo11m-seg_black_coco_seed{}", "U-Net": "unet_black_seed{}"}


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * ((x - x.mean()) * (y - y.mean())).mean() / den) if den else math.nan


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return math.nan if x.std() == 0 or y.std() == 0 else float(
        ((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))


def stats(pares):
    ref = [p[0] for p in pares]
    m = [p[1] for p in pares]
    dif = [b - a for a, b in pares]
    dp = st.stdev(dif) if len(dif) > 1 else 0.0
    return {"n": len(pares), "pearson": pearson(ref, m), "ccc": ccc(ref, m),
            "vies": st.mean(dif), "mae": st.mean(abs(d) for d in dif),
            "loa": 2 * 1.96 * dp}


# ── classical ───────────────────────────────────────────────────────────────
cls = {}
for r in csv.DictReader(open("stage3/benchmark_classical_long.csv", encoding="utf-8-sig")):
    k = (r["series_key"], r["campo"], str(int(float(r["timepoint_h"]))))
    try:
        cls[k] = (float(r["closure_reference"]), float(r["closure_whst_auto"]))
    except (TypeError, ValueError):
        pass

# ── models ──────────────────────────────────────────────────────────────────
mod = {}
for braco, molde in BRACOS.items():
    for s in SEEDS:
        p = os.path.join("stage3", f"cmp_{molde.format(s)}_paired_new_long.csv")
        if not os.path.isfile(p):
            continue
        mod[(braco, s)] = {
            (x["series_key"], x["campo"], str(int(float(x["timepoint_h"])))):
            (float(x["referencia"]), float(x["ai"]))
            for x in csv.DictReader(open(p, encoding="utf-8-sig"))}

comuns = set(cls)
for d in mod.values():
    comuns &= set(d)
comuns = sorted(comuns)
print(f"classical: {len(cls)} obs · models: {len(next(iter(mod.values())))} obs")
print(f"COMMON to the three arms and to all {len(mod)} runs: {len(comuns)}\n")
if len(comuns) < 20:
    sys.exit("ABORTED: the intersection is too small to compare")

# ── implausibility: the PRE-SPECIFIED criterion of stage 4 ──────────────────
# The [-0.05, 1.05] range comes from `stage4/whst_series_analysis.py::closure`,
# frozen before any analysis. Applying it here is not a post-hoc filter.
LO, HI = -0.05, 1.05


def fora(v):
    return v < LO or v > HI


print("=" * 78)
print("BEFORE AGREEMENT: how many values are even on scale?")
print("=" * 78)
n_cls = sum(1 for k in comuns if fora(cls[k][1]))
print(f"  classical (automatic WHST)   {n_cls:>5} of {len(comuns)} outside "
      f"[{LO}, {HI}]  ({100*n_cls/len(comuns):.1f}%)")
print(f"     three smallest: "
      f"{', '.join(f'{v:.1f}' for v in sorted(cls[k][1] for k in comuns)[:3])}")
for braco in BRACOS:
    n = st.mean(sum(1 for k in comuns if fora(mod[(braco, s)][k][1]))
                for s in SEEDS if (braco, s) in mod)
    print(f"  {braco:<27} {n:>5.1f} of {len(comuns)} outside the range "
          f"({100*n/len(comuns):.1f}%)")
print(f"  {'reference standard':<27} "
      f"{sum(1 for k in comuns if fora(cls[k][0])):>5} of {len(comuns)}")
print("""
  Mechanism: closure = (a0 - at)/a0. If the method under-segments the BASELINE
  frame, a0 becomes tiny and the ratio blows up. A single error at t=0 contaminates
  the whole series — which is why the manual correction step exists.

  Reporting the classical CCC without this line would say "it does not agree", when
  what actually happens is "it does not produce an on-scale number for part of the
  observations". Those are different claims, and the second is the true one.""")

plaus = [k for k in comuns if not fora(cls[k][1])]

print("\n" + "=" * 78)
print("AGREEMENT WITH THE REFERENCE STANDARD — closure fraction, same set")
print("=" * 78)
print(f"{'arm':<12} {'n':>4} {'Pearson':>16} {'Lin CCC':>16} "
      f"{'bias':>16} {'MAE':>8}")
print("-" * 78)

s_cls = stats([cls[k] for k in comuns])
print(f"{'classical*':<12} {s_cls['n']:>4} {s_cls['pearson']:>+16.4f} "
      f"{s_cls['ccc']:>+16.4f} {s_cls['vies']:>+16.4f} {s_cls['mae']:>8.4f}")

for braco in BRACOS:
    ss = [stats([mod[(braco, s)][k] for k in comuns])
          for s in SEEDS if (braco, s) in mod]
    cel = {}
    for c in ("pearson", "ccc", "vies", "mae"):
        m = st.mean(x[c] for x in ss)
        d = st.stdev([x[c] for x in ss]) if len(ss) > 1 else 0.0
        cel[c] = f"{m:+.4f}±{d:.4f}" if c != "mae" else f"{m:.4f}"
    print(f"{braco:<12} {ss[0]['n']:>4} {cel['pearson']:>16} {cel['ccc']:>16} "
          f"{cel['vies']:>16} {cel['mae']:>8}")

print(f"""
* ⚠️ THE CLASSICAL ARM SUFFERS FROM INCORPORATION BIAS. The reference standard was
  obtained by CORRECTING the output of the automatic WHST. In the frames the
  observer did not need to correct, the two sides are identical by construction.
  The classical arm's agreement is inflated by design and is not interpretable as
  accuracy.

  The deep models never saw the reference standard. The comparison is unfavourable
  to them by construction — which makes any advantage of theirs stronger, and any
  advantage of the classical arm meaningless.

  This is STARD's incorporation bias, which is why `benchmark_classical.py` already
  declares it.""")

# ── the classical arm restricted to what is on scale ────────────────────────
print("\n" + "-" * 78)
print(f"SENSITIVITY — classical restricted to the {len(plaus)} on-scale observations")
print("-" * 78)
sp = stats([cls[k] for k in plaus])
print(f"  {'classical':<12} {sp['n']:>4} {sp['pearson']:>+16.4f} "
      f"{sp['ccc']:>+16.4f} {sp['vies']:>+16.4f} {sp['mae']:>8.4f}")
print("""
  ⚠️ NOT THE PRIMARY ANALYSIS. Excluding the observations where the method failed
  is selection on the outcome and inflates agreement — the same caveat
  `stage3/paired_new.py` makes for the model's implausible series. The primary
  analysis is the table above; this one only says how the classical arm behaves
  WHEN it produces an on-scale number.""")

n_iguais = sum(1 for k in comuns if abs(cls[k][0] - cls[k][1]) < 1e-9)
print(f"\n  observations where classical and reference coincide exactly: "
      f"{n_iguais} de {len(comuns)}")
print("""  Incorporation does not show up as an identical value: closure depends on TWO
  frames (t=0 and t), and one of them having been corrected is enough for the
  values to differ. The bias operates at the level of the FRAME, not of the
  observation — in the frames the observer did not correct, the reference area is
  literally the classical output. An earlier wording of this script overstated the
  mechanism.""")
