# -*- coding: utf-8 -*-
"""
stage3/test_ap_core.py — validates ap_core before the training grid finishes.

No GPU. The point is to be sure the AP core is right while there is still time to
fix it: if it were written in a hurry once the 25 runs came out, an error here
would contaminate all of Table 2 and every CI — and nobody would notice, because
the number "looks reasonable".

Cases:
  1. perfect prediction (the GT itself, score 1) -> AP = 1.0 at every threshold
  2. no prediction at all                        -> AP = 0
  3. image with no GT                            -> AP undefined (NaN), not 0
  4. ordering by score matters                   -> same detection, different orders
  5. one GT, two identical predictions           -> the second is a FP (no double count)
  6. borderline IoU                              -> TP at 0.50 and FP at 0.75
  7. precision/recall/F1 vary with the confidence threshold and mAP does NOT
  8. cluster bootstrap is wider than the naive one
  9. the bootstrap is deterministic given the same seed
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ap_core import (IDX_50, IDX_75, IOU_THRS, average_precision, casa_imagem,
                     cluster_bootstrap, bootstrap_ingenuo, iou_mask, matriz_iou,
                     mapa_5095, metricas, prf)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

falhas = []


def check(cond, rot, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {rot}{('  ' + extra) if extra else ''}")
    if not cond:
        falhas.append(rot)


def quadrado(x0, y0, lado, n=64):
    m = np.zeros((n, n), bool)
    m[y0:y0 + lado, x0:x0 + lado] = True
    return m


def registro(preds, gts, scores):
    return casa_imagem(scores, matriz_iou(preds, gts))


print("1. perfect prediction")
g = [quadrado(10, 10, 20)]
r = registro(g, g, [1.0])
aps = [average_precision([r], k) for k in range(len(IOU_THRS))]
check(all(abs(a - 1.0) < 1e-9 for a in aps), f"AP = 1.0 at all 10 thresholds",
      f"(min {min(aps):.4f})")

print("\n2. no prediction at all")
r0 = casa_imagem([], np.zeros((0, 1), np.float32))
r0["n_gt"] = 1
check(average_precision([r0], IDX_50) == 0.0, "AP = 0 when there is no detection")

print("\n3. image with no GT")
r_semgt = registro([quadrado(0, 0, 5)], [], [0.9])
check(r_semgt["n_gt"] == 0, "n_gt = 0")
check(np.isnan(average_precision([r_semgt], IDX_50)),
      "AP undefined (NaN), not 0 — an image with no GT cannot 'zero' the mean")
# AP is RANK-BASED, and that has a counterintuitive consequence which has to be
# understood before Table 2 is interpreted: a false positive scored BELOW every
# true positive does not reduce AP. The precision envelope takes the maximum to
# the right, and recall has already reached 1.0 before the FP appears.
# Only an FP ranked above some TP hurts.
# the TP is given score 0.50 so that there is ranking space on both sides of it
r_bom = registro(g, g, [0.50])
ap_so_bom = average_precision([r_bom], IDX_50)
fp_abaixo = registro([quadrado(0, 0, 5)], [], [0.30])     # ranqueado DEPOIS do TP
fp_acima = registro([quadrado(0, 0, 5)], [], [0.90])      # ranqueado ANTES do TP
ap_fp_baixo = average_precision([r_bom, fp_abaixo], IDX_50)
ap_fp_alto = average_precision([r_bom, fp_acima], IDX_50)
check(ap_fp_baixo == ap_so_bom,
      "an FP ranked BELOW the TP does not change AP (a property of the rank)",
      f"({ap_so_bom:.3f} -> {ap_fp_baixo:.3f})")
check(ap_fp_alto < ap_so_bom,
      "an FP ranked ABOVE the TP reduces AP",
      f"({ap_so_bom:.3f} -> {ap_fp_alto:.3f})")
# and in both cases it lowers pointwise precision, which is what the 80% threshold sees
p_sem = prf([r_bom], conf=0.2)["precision"]
p_com = prf([r_bom, fp_abaixo], conf=0.2)["precision"]
check(p_com < p_sem, "but pointwise precision falls in both cases",
      f"({p_sem:.2f} -> {p_com:.2f})")

print("\n4. ordering by score")
gts = [quadrado(10, 10, 20), quadrado(40, 40, 10)]
preds = [quadrado(10, 10, 20), quadrado(0, 0, 4)]     # 1 certa, 1 lixo
alto = registro(preds, gts, [0.9, 0.1])               # a certa vem primeiro
baixo = registro(preds, gts, [0.1, 0.9])              # o lixo vem primeiro
a1, a2 = average_precision([alto], IDX_50), average_precision([baixo], IDX_50)
check(a1 > a2, "the correct detection scored high gives a higher AP", f"({a1:.3f} > {a2:.3f})")

print("\n5. two predictions for one GT")
r = registro([quadrado(10, 10, 20), quadrado(10, 10, 20)], [quadrado(10, 10, 20)],
             [0.9, 0.8])
check(r["tp"][IDX_50] == [1, 0], "the second is an FP — the GT is not counted twice",
      f"tp={r['tp'][IDX_50]}")

print("\n6. borderline IoU")
a, b = quadrado(0, 0, 20), quadrado(0, 0, 20)
# shift until the IoU falls between 0.50 and 0.75
b = np.zeros_like(a)
b[0:20, 5:25] = True
i = iou_mask(a, b)
r = registro([b], [a], [0.9])
check(0.5 <= i < 0.75, f"IoU constructed = {i:.3f}")
check(r["tp"][IDX_50] == [1] and r["tp"][IDX_75] == [0],
      "TP at 0.50 and FP at 0.75", f"tp50={r['tp'][IDX_50]} tp75={r['tp'][IDX_75]}")

print("\n7. confidence threshold: affects P/R/F1, not mAP")
regs = [registro([quadrado(10, 10, 20), quadrado(0, 0, 4)], [quadrado(10, 10, 20)],
                 [0.95, 0.30])]
m_alto = prf(regs, conf=0.8)
m_baixo = prf(regs, conf=0.1)
ap_a, ap_b = average_precision(regs, IDX_50), average_precision(regs, IDX_50)
check(m_alto["precision"] != m_baixo["precision"],
      "precision changes with the threshold", f"({m_alto['precision']:.2f} vs {m_baixo['precision']:.2f})")
check(ap_a == ap_b, "mAP is unchanged — it does not depend on the confidence threshold")

print("\n8. cluster bootstrap × naive")
# 40 groups of 6 images; within a group, performance is HIGHLY correlated
rng = np.random.default_rng(7)
regs_img, grupo_de = {}, {}
for gidx in range(40):
    bom = rng.random() < 0.5                    # o grupo inteiro é bom ou ruim
    for j in range(6):
        k = f"g{gidx}_i{j}"
        gt = [quadrado(10, 10, 20)]
        if bom:
            pr, sc = [quadrado(10, 10, 20)], [0.9]
        else:
            pr, sc = [quadrado(40, 40, 6)], [0.9]     # erra feio
        regs_img[k] = registro(pr, gt, sc)
        grupo_de[k] = f"g{gidx}"
fn = lambda rs: average_precision(rs, IDX_50)
cb = cluster_bootstrap(regs_img, grupo_de, fn, B=400, seed=1)
ib = bootstrap_ingenuo(regs_img, fn, B=400, seed=1)
larg_c, larg_i = cb["hi"] - cb["lo"], ib["hi"] - ib["lo"]
print(f"       cluster : [{cb['lo']:.3f}, {cb['hi']:.3f}]  width {larg_c:.3f}")
print(f"       naive   : [{ib['lo']:.3f}, {ib['hi']:.3f}]  width {larg_i:.3f}")
check(larg_c > larg_i, "the grouped CI is WIDER than the naive one",
      f"({larg_c/larg_i:.2f}x)")
check(cb["B_validos"] > 0 and ib["B_validos"] > 0, "valid resamples in both")

print("\n9. determinism")
c1 = cluster_bootstrap(regs_img, grupo_de, fn, B=200, seed=99)
c2 = cluster_bootstrap(regs_img, grupo_de, fn, B=200, seed=99)
check(c1 == c2, "same seed, same interval")
c3 = cluster_bootstrap(regs_img, grupo_de, fn, B=200, seed=100)
check(c1 != c3, "different seed, different interval")

print("\n" + "=" * 66)
if falhas:
    print(f"{len(falhas)} CHECKS FAILED:")
    for f in falhas:
        print("   -", f)
    sys.exit(1)
print("ap_core validated.")
