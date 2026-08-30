# -*- coding: utf-8 -*-
"""
stage4/validate_provenance.py — testa a premissa da procedencia mista.

Compares, over the N drawn frames (classified OK in the triage, and therefore with an
AUTOMATICA nas series mistas), a area automatica contra a area corrigida
manualmente pelo mesmo observador, as cegas.

CRITERIO DECLARADO A PRIORI (stage4/build_validation_worklist.py):
  |delta| mediano  < 5%   -> mistura VALIDADA
  |delta| mediano >= 20%  -> viés; decidir entre corrigir todas as mistas ou
                             restringir a analise as series homogeneas
  entre 5% e 20%          -> zona cinzenta; decidir pelo efeito na closure

Reporta tambem o IoU(automatico, corrigido) nesses frames, comparavel ao 0,262
observed on the frames judged BAD — if it is much higher here, that confirms the
visual triage separated the good from the bad well.
"""
import csv, os, sys
import numpy as np
from PIL import Image

VAL = "stage4/manual_correction_validation.csv"
GAB = "stage4/.validacao_gabarito.csv"
AUTO_MASKS = "whst_output/masks"
VAL_MASKS = "whst_output/rois_validation/masks"
OUT = "stage4/validation_provenance.csv"

if not os.path.isfile(VAL):
    sys.exit(f"could not find {VAL} — run pass '3 - validacao' in Fiji")

val = {r["whst_input_file"]: r for r in csv.DictReader(open(VAL, encoding="utf-8-sig"))}
gab = {r["whst_input_file"]: r for r in csv.DictReader(open(GAB, encoding="utf-8-sig"))}


def base(f):
    for e in (".tiff", ".tif"):
        if f.lower().endswith(e):
            return f[: -len(e)]
    return os.path.splitext(f)[0]


def mask(d, f):
    p = os.path.join(d, base(f) + "_mask.png")
    return (np.asarray(Image.open(p).convert("L")) > 127) if os.path.exists(p) else None


def iou(a, b):
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return 1.0 if union == 0 else float(inter / union)


rows, deltas, ious = [], [], []
for k, g in sorted(gab.items()):
    v = val.get(k)
    if not v:
        print(f"  [pending] not yet corrected: {k[:60]}")
        continue
    aa = float(g["area_pct_auto"])
    if v["status"] == "ok":
        ac = float(v["area_pct_corrigida"])
    elif v["status"] == "fechada":
        ac = 0.0
    else:
        print(f"  [nota] status='{v['status']}' em {k[:56]} — fora do calculo de delta")
        continue
    d_abs = ac - aa
    d_rel = (ac - aa) / aa if aa > 0 else float("nan")
    ma, mc = mask(AUTO_MASKS, k), mask(VAL_MASKS, k)
    j = iou(ma, mc) if (ma is not None and mc is not None and ma.shape == mc.shape) else float("nan")
    deltas.append(d_rel); ious.append(j)
    rows.append({"whst_input_file": k, "series_key": g["series_key"],
                 "area_pct_auto": round(aa, 3), "area_pct_corrigida": round(ac, 3),
                 "delta_pp": round(d_abs, 3), "delta_rel": round(d_rel, 4),
                 "iou_auto_vs_corr": ("" if np.isnan(j) else round(j, 4)),
                 "status": v["status"]})

if not rows:
    sys.exit("no comparable frame yet.")

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

dr = np.array([d for d in deltas if not np.isnan(d)])
ad = np.abs(dr)
ii = np.array([x for x in ious if not np.isnan(x)])

print(f"\n=== VALIDATION OF MIXED PROVENANCE (n={len(dr)}) ===")
print(f"  drawn frames: classified OK in the triage (automatic area),")
print(f"  corrigidos as cegas pelo mesmo observador.\n")
print(f"  delta relativo (corrigida vs automatica):")
print(f"    mediana |delta| = {np.median(ad):.1%}")
print(f"    median   delta  = {np.median(dr):+.1%}   (sign: <0 = the correction reduced the area)")
print(f"    IQR |delta|     = [{np.percentile(ad,25):.1%}, {np.percentile(ad,75):.1%}]")
print(f"    min/max delta   = {dr.min():+.1%} / {dr.max():+.1%}")
if len(ii):
    print(f"\n  IoU(automatico, corrigido) nestes frames OK: mediana={np.median(ii):.3f}")
    print(f"    (compare with the 0.262 of the frames judged BAD)")

m = float(np.median(ad))
print("\n  === VEREDITO (criterio declarado a priori) ===")
if m < 0.05:
    print(f"    median |delta| = {m:.1%} < 5%  ->  MIXTURE VALIDATED")
    print("    Frames 'OK' sao tao acurados quanto os corrigidos; series mistas")
    print("    are not biased. Declare it in the Methods.")
elif m >= 0.20:
    print(f"    median |delta| = {m:.1%} >= 20%  ->  BIAS DETECTED")
    print("    Decide: (a) correct every frame of the 25 mixed series, or")
    print("             (b) restrict the analysis to the homogeneous series.")
else:
    print(f"    median |delta| = {m:.1%}  ->  GREY ZONE (5%-20%)")
    print("    Assess the effect on closure before deciding.")
print(f"\nSalvo: {OUT}")
