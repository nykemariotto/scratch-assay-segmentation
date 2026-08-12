# -*- coding: utf-8 -*-
"""
stage3/padding_ablation.py — the padding ablation, paired by seed, PERSISTED to disk.

WHY. The ablation statistics (mean difference, SD, paired t, how many seeds each
arm wins) were computed once by stage3/aggregate.py and existed only on the
console. The manuscript quotes those numbers, and there was no file backing them.
A reported number with no file that generates it is not reproducible.

This script recomputes the pairing and WRITES `stage3/padding_ablation.csv` and
`stage3/padding_ablation.json`. It runs no bootstrap: the seed-paired statistic is
cheap — it reads arrays already saved by stage3/eval_test.py, with no model
inference — and can be run while the GPU is busy. The paired cluster-bootstrap CI
remains the responsibility of stage3/aggregate.py.

    python stage3/padding_ablation.py
"""
import csv
import json
import math
import os
import re
import statistics as st
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
os.chdir(os.path.dirname(AQUI))

from ap_core import IDX_50, IDX_75, average_precision, mapa_5095   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REGS = os.path.join("stage3", "registros")
# black vs white padding, everything else identical: M against M-white
A = "yolo11m-seg_black_coco"
B = "yolo11m-seg_white_coco"
SEEDS = [42, 43, 44, 45, 46]


def metricas(nome):
    p = os.path.join(REGS, f"{nome}.json")
    if not os.path.isfile(p):
        sys.exit(f"ABORTED: could not find {p}")
    d = json.load(open(p, encoding="utf-8"))
    if d.get("n_imagens") != 234:
        sys.exit(f"ABORTED: {nome} has n_imagens={d.get('n_imagens')}, expected 234")
    regs = list(d["registros"].values())
    return {"mAP50": 100 * average_precision(regs, IDX_50),
            "mAP75": 100 * average_precision(regs, IDX_75),
            "mAP50_95": 100 * mapa_5095(regs)}


print(f"seed-paired padding ablation · {A} (black) vs {B} (white)")
print(f"test set: 234 images · {len(SEEDS)} pairs\n")

linhas, difs = [], {k: [] for k in ("mAP50", "mAP75", "mAP50_95")}
for s in SEEDS:
    ma, mb = metricas(f"{A}_seed{s}"), metricas(f"{B}_seed{s}")
    linha = {"seed": s}
    for k in difs:
        linha[f"{k}_black"] = round(ma[k], 4)
        linha[f"{k}_white"] = round(mb[k], 4)
        d = ma[k] - mb[k]
        linha[f"{k}_dif"] = round(d, 4)
        difs[k].append(d)
    linhas.append(linha)
    print(f"  seed {s}: mAP@50 black {ma['mAP50']:.2f}  white {mb['mAP50']:.2f}  "
          f"dif {ma['mAP50']-mb['mAP50']:+.2f} pp")

resumo = {}
print()
for k, ds in difs.items():
    n = len(ds)
    media, dp = st.mean(ds), st.stdev(ds)
    # paired t: mean of the difference / standard error of the difference
    t = media / (dp / math.sqrt(n)) if dp > 0 else float("nan")
    vence_preto = sum(1 for d in ds if d > 0)
    resumo[k] = {"n_pares": n, "dif_media_pp": round(media, 4),
                 "dif_dp_pp": round(dp, 4), "t_pareado": round(t, 4), "gl": n - 1,
                 "seeds_em_que_o_preto_vence": vence_preto,
                 "amplitude_braco_preto_pp": [round(min(l[f"{k}_black"] for l in linhas), 4),
                                              round(max(l[f"{k}_black"] for l in linhas), 4)],
                 "amplitude_das_diferencas_pp": [round(min(ds), 4), round(max(ds), 4)]}
    print(f"{k}: difference (black - white) = {media:+.3f} pp · SD {dp:.3f} pp · "
          f"t({n-1}) = {t:+.3f} · black wins {vence_preto}/{n}")
    lo, hi = resumo[k]["amplitude_braco_preto_pp"]
    print(f"{'':11s}range of the black arm across seeds: {lo:.2f}–{hi:.2f} pp "
          f"({hi-lo:.2f} pp of spread)")

# the central argument: the spread across seeds WITHIN an arm exceeds the effect
for k in difs:
    lo, hi = resumo[k]["amplitude_braco_preto_pp"]
    resumo[k]["faixa_entre_seeds_maior_que_efeito"] = bool(
        (hi - lo) > abs(resumo[k]["dif_media_pp"]))

with open(os.path.join("stage3", "padding_ablation.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
    w.writeheader()
    w.writerows(linhas)
json.dump({"contraste": f"{A} (black) - {B} (white)", "n_imagens_teste": 234,
           "seeds": SEEDS, "por_seed": linhas, "resumo": resumo,
           "nota": "the paired cluster-bootstrap CI is NOT computed here; it "
                   "belongs to stage3/aggregate.py, which persists it."},
          open(os.path.join("stage3", "padding_ablation.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\nwrote stage3/padding_ablation.csv and .json")
