# -*- coding: utf-8 -*-
"""
stage4/provenance_effect_on_closure.py — resolves the grey zone of the validation.

The a-priori criterion said that, between 5% and 20%, the EFFECT ON THE CLOSURE is
what decides. Two distinct questions, which |delta| alone conflates:

  (a) IS THERE SYSTEMATIC BIAS?  For a ratio, what biases the closure is a
      difference with a consistent DIRECTION (all automatic areas larger, or all
      smaller). Symmetric dispersion around zero does not bias the closure, it only
      adds noise. Tested with a sign test plus Wilcoxon.

  (b) HOW MUCH DOES THE CLOSURE MOVE IN PRACTICE?  Recomputes the closure of the
      affected series, substituting the automatic area of the drawn frame with the
      corrected area, and measures the absolute shift in closure. Baselines (t0)
      are handled separately, because an error in a0 propagates to every timepoint
      of the series.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np
import statistics as st

VAL = "stage4/validation_provenance.csv"
AREAS = "data/whst_areas_final.csv"
if not os.path.isfile(VAL):
    sys.exit("run stage4/validate_provenance.py first")

V = list(csv.DictReader(open(VAL, encoding="utf-8-sig")))
A = list(csv.DictReader(open(AREAS, encoding="utf-8-sig")))
byk = {r["whst_input_file"]: r for r in A}
bys = defaultdict(list)
for r in A:
    if r["area_pct_final"] != "":
        bys[r["series_key"]].append(r)

d = np.array([float(r["delta_rel"]) for r in V])
print("=== (a) IS THERE SYSTEMATIC BIAS? ===")
pos, neg = int((d > 0).sum()), int((d < 0).sum())
n = pos + neg
print(f"  positive deltas (corrected > automatic): {pos}")
print(f"  negative deltas (corrected < automatic): {neg}")
# exact sign test (binomial p=0.5, two-sided)
from math import comb
k = min(pos, neg)
p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)
print(f"  sign test (two-sided): p = {p_sign:.3f}")
try:
    from scipy.stats import wilcoxon
    w = wilcoxon(d)
    print(f"  Wilcoxon (median=0): W={w.statistic:.1f}  p = {w.pvalue:.3f}")
except Exception as e:
    print(f"  (scipy unavailable: {e})")
print(f"  signed median = {np.median(d):+.1%}   mean = {d.mean():+.1%}")
print(f"  -> {'NO detectable bias: symmetric dispersion' if p_sign > 0.05 else 'DIRECTIONAL bias detected'}")

# ---------- (b) effect on the closure ----------
print("\n=== (b) THE REAL EFFECT ON THE CLOSURE ===")
subs = {r["whst_input_file"]: float(r["area_pct_corrigida"]) for r in V}


def curva(rs, sub=None):
    dd = defaultdict(list)
    for r in rs:
        a = float(r["area_pct_final"])
        if sub and r["whst_input_file"] in sub:
            a = sub[r["whst_input_file"]]
        dd[int(r["timepoint_h"])].append(a)
    return {tp: st.median(v) for tp, v in dd.items()}


def closure(ab):
    if 0 not in ab or ab[0] <= 0:
        return None
    a0 = ab[0]
    return {tp: (a0 - ab[tp]) / a0 for tp in sorted(ab)}


linhas, desl_all, desl_base, desl_nb = [], [], [], []
afetadas = sorted({byk[k]["series_key"] for k in subs})
for sk in afetadas:
    rs = bys[sk]
    c0 = closure(curva(rs))
    c1 = closure(curva(rs, subs))
    if not c0 or not c1:
        continue
    alvos = [r for r in rs if r["whst_input_file"] in subs]
    tem_base = any(int(r["timepoint_h"]) == 0 for r in alvos)
    difs = [abs(c1[t] - c0[t]) for t in c0 if t in c1 and t > 0]
    if not difs:
        continue
    mx = max(difs)
    desl_all.append(mx)
    (desl_base if tem_base else desl_nb).append(mx)
    linhas.append({"series_key": sk, "frame_alterado_e_baseline": "sim" if tem_base else "nao",
                   "closure_antes": ";".join(f"{t}h:{v:.4f}" for t, v in sorted(c0.items())),
                   "closure_depois": ";".join(f"{t}h:{v:.4f}" for t, v in sorted(c1.items())),
                   "deslocamento_max": round(mx, 4)})

if linhas:
    with open("stage4/provenance_effect.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader(); w.writerows(linhas)

da = np.array(desl_all)
print(f"  affected series: {len(da)}")
print(f"  ABSOLUTE shift in closure (closure units, 0-1):")
print(f"    median = {np.median(da):.4f}   max = {da.max():.4f}   IQR=[{np.percentile(da,25):.4f}, {np.percentile(da,75):.4f}]")
if desl_base:
    print(f"    when the swapped frame is the BASELINE (n={len(desl_base)}): median={np.median(desl_base):.4f} max={max(desl_base):.4f}")
if desl_nb:
    print(f"    when it is NOT the baseline           (n={len(desl_nb)}): median={np.median(desl_nb):.4f} max={max(desl_nb):.4f}")

print("\n=== VERDICT ===")
sem_vies = p_sign > 0.05
peq = float(np.median(da)) < 0.05
print(f"  directional bias : {'NO' if sem_vies else 'YES'}  (sign test p={p_sign:.3f})")
print(f"  median shift     : {np.median(da):.4f} ({'<' if peq else '>='} 0.05 of closure)")
if sem_vies and peq:
    print("\n  -> MISTURA VALIDADA na pratica.")
    print("     The 10% |delta| is DISPERSION, not bias: it does not shift the closure")
    print("     systematically. Mixed series stay analysable; declare the")
    print("     no Methods a magnitude do ruido residual.")
elif not sem_vies:
    print("\n  -> DIRECTIONAL BIAS: correct all mixed series or restrict")
    print("     the analysis to the homogeneous ones.")
else:
    print("\n  -> No bias, but a large offset: report it as uncertainty")
    print("     and consider correcting the mixed series.")
