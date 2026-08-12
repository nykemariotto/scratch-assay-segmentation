# -*- coding: utf-8 -*-
"""
stage3/compara_bracos_closure.py — YOLO × U-Net on the ASSAY OUTCOME, not on the mask.

WHY THIS CAN OVERTURN THE IoU VERDICT. What the assay produces is not the mask: it
is the closure fraction, `(a₀ − aₜ) / a₀`. That is a RATIO between areas of the same
field, and it therefore **cancels constant multiplicative bias**. A model that
overestimates every area by 20% has a worse IoU and an identical closure fraction; a
model that overestimates only at the late timepoints has a similar IoU and a biased
closure fraction.

So: which of the two arms agrees better with the reference standard is **not
deducible** from IoU. It has to be measured.

WHAT IT DOES. Runs `stage3/paired_new.py` for the 5 seeds of the reference YOLO and
the 5 of the U-Net, each with its own prefix, and aggregates the agreement
statistics per arm — mean ± SD over the seeds, not one seed picked at random.

Comparison paired by SERIES: the same observations enter both arms, so the
difference between them is measured on the same set and the variance of the
reference standard cancels.

    python stage3/compara_bracos_closure.py
"""
import argparse
import csv
import glob
import math
import os
import statistics as st
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
os.chdir(RAIZ)

import numpy as np                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PY = sys.executable
BRACOS = {"YOLO M": "yolo11m-seg_black_coco_seed{}",
          "U-Net": "unet_black_seed{}"}
SEEDS = [42, 43, 44, 45, 46]


def ccc(x, y):
    """Lin's concordance. It penalises systematic bias; Pearson does not."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    den = vx + vy + (mx - my) ** 2
    return float(2 * cov / den) if den else float("nan")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float((((x - x.mean()) * (y - y.mean())).mean()) / (sx * sy))


def stats_de(pares):
    ref = [p["ref"] for p in pares]
    ai = [p["ai"] for p in pares]
    dif = [a - r for a, r in zip(ai, ref)]
    dp = st.stdev(dif) if len(dif) > 1 else 0.0
    return {"n": len(pares), "pearson": pearson(ref, ai), "ccc": ccc(ref, ai),
            "vies": st.mean(dif), "dp_dif": dp,
            "loa_lo": st.mean(dif) - 1.96 * dp, "loa_hi": st.mean(dif) + 1.96 * dp,
            "mae": st.mean(abs(d) for d in dif)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="reroda o stage3/paired_new.py mesmo se o CSV já existir")
    args = ap.parse_args()

    por_run = {}
    for braco, molde in BRACOS.items():
        for s in SEEDS:
            run = molde.format(s)
            if not os.path.isfile(os.path.join("stage3", "areas", f"{run}.csv")):
                print(f"  [SKIPPED] {run}: no areas in stage3/areas/")
                continue
            pref = f"cmp_{run}_"
            saida = os.path.join("stage3", f"{pref}paired_new_long.csv")
            if args.force or not os.path.isfile(saida):
                r = subprocess.run(
                    [PY, os.path.join("stage3", "paired_new.py"), "--run", run,
                     "--out-prefix", pref],
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
                if r.returncode != 0:
                    print(f"  [FAILED] {run}\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
                    continue
            linhas = list(csv.DictReader(open(saida, encoding="utf-8-sig")))
            por_run[(braco, s)] = {
                (l["series_key"], l["campo"], l["timepoint_h"]):
                {"ref": float(l["referencia"]), "ai": float(l["ai"])}
                for l in linhas}
            print(f"  {braco:<8} seed {s}: {len(linhas)} observations")

    if not por_run:
        sys.exit("no run was processed")

    # ── the observations common to all TEN runs ─────────────────────────────
    comuns = None
    for k, d in por_run.items():
        comuns = set(d) if comuns is None else (comuns & set(d))
    print(f"\nobservations present in ALL {len(por_run)} runs: {len(comuns)}")
    print("The comparison uses only these, so both arms are measured on the same set")
    print("and the variance of the reference standard cancels.\n")

    # ── per arm: mean ± SD of the statistics over the seeds ─────────────────
    resumo = {}
    for braco in BRACOS:
        seeds = [s for (b, s) in por_run if b == braco]
        ss = [stats_de([por_run[(braco, s)][k] for k in sorted(comuns)]) for s in seeds]
        resumo[braco] = {c: (st.mean(x[c] for x in ss),
                             st.stdev([x[c] for x in ss]) if len(ss) > 1 else 0.0)
                         for c in ("pearson", "ccc", "vies", "dp_dif", "mae",
                                   "loa_lo", "loa_hi")}
        resumo[braco]["n"] = ss[0]["n"]
        resumo[braco]["seeds"] = len(ss)

    print("AGREEMENT WITH THE REFERENCE STANDARD — mean ± SD over the seeds")
    print("=" * 70)
    print(f"{'':<12} {'YOLO M':>22} {'U-Net':>22}")
    print("-" * 70)
    for rot, c, f in (("n", None, None),
                      ("Pearson r", "pearson", "{:+.4f}"),
                      ("Lin CCC", "ccc", "{:+.4f}"),
                      ("bias", "vies", "{:+.4f}"),
                      ("SD of diff.", "dp_dif", "{:.4f}"),
                      ("MAE", "mae", "{:.4f}")):
        if c is None:
            print(f"{rot:<12} {resumo['YOLO M']['n']:>22} {resumo['U-Net']['n']:>22}")
            continue
        cel = []
        for b in ("YOLO M", "U-Net"):
            m, d = resumo[b][c]
            cel.append(f"{f.format(m)} ± {d:.4f}")
        print(f"{rot:<12} {cel[0]:>22} {cel[1]:>22}")
    for b in ("YOLO M", "U-Net"):
        lo, hi = resumo[b]["loa_lo"][0], resumo[b]["loa_hi"][0]
        print(f"  95% LoA {b:<8}: [{lo:+.4f}, {hi:+.4f}]  (width {hi-lo:.4f})")

    # ── contrast paired by seed, on the same set ────────────────────────────
    print("\n" + "=" * 70)
    print("PAIRED CONTRAST — same seed, same observations")
    print("=" * 70)
    difs = {}
    for c in ("ccc", "pearson", "mae"):
        d = []
        for s in SEEDS:
            if ("YOLO M", s) in por_run and ("U-Net", s) in por_run:
                a = stats_de([por_run[("YOLO M", s)][k] for k in sorted(comuns)])
                b = stats_de([por_run[("U-Net", s)][k] for k in sorted(comuns)])
                d.append(a[c] - b[c])
        difs[c] = d
        if not d:
            continue
        m = st.mean(d)
        sd = st.stdev(d) if len(d) > 1 else 0.0
        t = m / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
        venc = "YOLO" if (m > 0 if c != "mae" else m < 0) else "U-Net"
        print(f"  {c:<9} YOLO − U-Net = {m:+.4f} ± {sd:.4f}   t({len(d)-1}) = {t:+.3f}"
              f"   favours {venc} in {sum(1 for x in d if (x>0 if c!='mae' else x<0))}/{len(d)}")

    print("""
HOW TO READ THIS. The CCC is the statistic that decides: it penalises systematic
bias, Pearson does not. A high Pearson with a low CCC means the model tracks the
trend but gets the level wrong — and then a claim of interchangeability does not
hold, whatever the IoU.

The MAE is in units of closure fraction: 0.05 = five percentage points of closure.
It is the error the user of the tool actually sees.""")


if __name__ == "__main__":
    main()
