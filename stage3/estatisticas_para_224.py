
# -*- coding: utf-8 -*-
"""
stage3/estatisticas_para_224.py — everything the manuscript's equivalence claim
needs in order to be stated.

That passage used to say the AI workflow is "a quantitatively reliable replacement
for ImageJ-based scratch assay analysis", supported by r = 0.840, CCC = 0.838,
bias −0.014 and TOST equivalence at ±0.10 and ±0.05.

Those numbers came from the model trained on the partition that LEAKED, compared
against three annotators' measurements. The new pairing is against the stage-4
reference standard, with the leakage-free model. This script recomputes all of it,
including TOST at both pre-specified margins, so the sentence can be rewritten with
numbers that exist.

TOST: two one-sided tests on the paired difference. Equivalence at a margin δ is
declared when the (1−2α) CI of the mean difference falls entirely inside
[−δ, +δ]. Margins pre-specified in §2.8: ±0.10 (primary) and ±0.05 (sensitivity).

    python stage3/estatisticas_para_224.py
"""
import csv
import math
import os
import statistics as st
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(AQUI))

import numpy as np                                  # noqa: E402
from scipy import stats as sps                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEEDS = [42, 43, 44, 45, 46]
BRACOS = {"YOLO M": "yolo11m-seg_black_coco_seed{}", "U-Net": "unet_black_seed{}"}
MARGENS = [0.10, 0.05]


sys.path.insert(0, AQUI)
from _concordancia_estat import ccc, tost      # noqa: E402  (definição única)

dados = {}
for braco, molde in BRACOS.items():
    for s in SEEDS:
        p = os.path.join("stage3", f"cmp_{molde.format(s)}_paired_new_longo.csv")
        if os.path.isfile(p):
            dados[(braco, s)] = {
                (x["series_key"], x["campo"], x["timepoint_h"]):
                (float(x["referencia"]), float(x["ai"]))
                for x in csv.DictReader(open(p, encoding="utf-8-sig"))}
comuns = None
for d in dados.values():
    comuns = set(d) if comuns is None else comuns & set(d)
comuns = sorted(comuns)
print(f"{len(comuns)} observations common to all {len(dados)} runs\n")

for braco in BRACOS:
    print("=" * 70)
    print(f"{braco} × the stage-4 reference standard")
    print("=" * 70)
    acc = {k: [] for k in ("r", "rho", "ccc", "vies", "dp", "loa_lo", "loa_hi", "n")}
    tosts = {d: [] for d in MARGENS}
    for s in SEEDS:
        if (braco, s) not in dados:
            continue
        par = [dados[(braco, s)][k] for k in comuns]
        ref = [a for a, _ in par]
        ai = [b for _, b in par]
        dif = [b - a for a, b in par]
        dp = st.stdev(dif)
        acc["n"].append(len(par))
        acc["r"].append(float(sps.pearsonr(ref, ai)[0]))
        acc["rho"].append(float(sps.spearmanr(ref, ai)[0]))
        acc["ccc"].append(ccc(ref, ai))
        acc["vies"].append(st.mean(dif))
        acc["dp"].append(dp)
        acc["loa_lo"].append(st.mean(dif) - 1.96 * dp)
        acc["loa_hi"].append(st.mean(dif) + 1.96 * dp)
        for d in MARGENS:
            tosts[d].append(tost(dif, d))

    def mm(k, f="{:+.4f}"):
        return f"{f.format(st.mean(acc[k]))} ± {st.stdev(acc[k]):.4f}"

    print(f"  n .................. {acc['n'][0]}")
    print(f"  Pearson r .......... {mm('r')}")
    print(f"  Spearman rho ....... {mm('rho')}")
    print(f"  CCC de Lin ......... {mm('ccc')}")
    print(f"  bias (AI − ref) .... {mm('vies')}")
    print(f"  SD of differences .. {mm('dp', '{:.4f}')}")
    print(f"  LoA 95% ............ [{st.mean(acc['loa_lo']):+.4f}, "
          f"{st.mean(acc['loa_hi']):+.4f}]  largura "
          f"{st.mean(acc['loa_hi'])-st.mean(acc['loa_lo']):.4f}")
    for d in MARGENS:
        eq = sum(1 for t in tosts[d] if t["equivalente"])
        print(f"  TOST ±{d:.2f} ......... equivalente em {eq}/{len(tosts[d])} seeds"
              f"   p máx {max(t['p'] for t in tosts[d]):.4f}"
              f"   IC90% da média [{st.mean(t['lo'] for t in tosts[d]):+.4f}, "
              f"{st.mean(t['hi'] for t in tosts[d]):+.4f}]")
    print()

print("""LEITURA. A equivalência TOST responde "a diferença MÉDIA é pequena?". A LoA
answers "how wrong can ONE measurement be?". The two can disagree, and that is
exactly what is worth reporting: an unbiased mean with LoA of ±0.3 means the
method does not shift the result of the experiment, but does not replace the
individual.""")
