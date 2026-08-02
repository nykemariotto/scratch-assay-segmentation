# -*- coding: utf-8 -*-
"""
stage3/make_area_fixture.py — synthetic areas to validate the JOIN in stage3/paired_new.py.

The real risk in stage3/paired_new.py is not the statistics, it is KEY MATCHING:
three files with different naming conventions (the export name in the test set, the
whst_input name with an md5 prefix, and series_key with pipe separators). If the
join is wrong the script does not crash — it simply returns fewer pairs, and nobody
notices.

This generator builds a `stage3/areas/<run>.csv` from the REAL areas of the
reference standard, propagated backwards through the join. With controlled noise:

  --ruido 0.0   -> the AI equals the reference exactly. paired_new MUST return
                   CCC = 1.000 and bias = 0. Anything else is a matching error,
                   not a model error.
  --ruido 0.05  -> high but imperfect agreement, to exercise the arithmetic.
  --vies 0.10   -> shifts the AI: Pearson barely moves and the CCC collapses. That
                   is the demonstration of why agreement needs CCC, not correlation.

The flag names stay in Portuguese: they are the documented command-line contract.
"""
import argparse
import csv
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(AQUI))

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def ler(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="FIXTURE_perfeito")
    ap.add_argument("--ruido", type=float, default=0.0, help="relative SD on the area")
    ap.add_argument("--vies", type=float, default=0.0, help="relative bias on the area")
    ap.add_argument("--vies-por-tp", type=float, default=0.0,
                    help="bias that GROWS with the timepoint (per 24 h). Constant bias "
                         "cancels in the ratio (a0-at)/a0 and does not affect closure; "
                         "only bias that varies with time survives — and that is what happens")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    INSP = ler("data/inspecao_visual.csv")
    WA = {r["whst_input_file"]: r for r in ler("data/whst_areas_final.csv")}
    rng = np.random.default_rng(args.seed)

    linhas, faltando = [], 0
    for r in INSP:
        t = r.get("test_image")
        if not t:
            continue
        w = WA.get(r.get("whst_input_file", ""))
        if not w:
            faltando += 1
            continue
        a = w.get("area_pct_final")
        try:
            a = float(a)
        except (TypeError, ValueError):
            continue
        try:
            tp = float(w.get("timepoint_h", 0) or 0)
        except (TypeError, ValueError):
            tp = 0.0
        k = 1 + args.vies + args.vies_por_tp * (tp / 24.0)
        a2 = a * k * (1 + rng.normal(0, args.ruido))
        a2 = max(0.0, a2)
        linhas.append({"arquivo": t, "area_px": int(round(a2 * 100)),
                       "area_pct": round(a2, 6), "n_instancias": 1 if a2 > 0 else 0,
                       "orig_w": 640, "orig_h": 640})

    os.makedirs(os.path.join("stage3", "areas"), exist_ok=True)
    dest = os.path.join("stage3", "areas", f"{args.run}.csv")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"{len(linhas)} areas written to {dest}  (no reference: {faltando})")
    print(f"noise={args.ruido} · bias={args.vies}")
    if args.vies and not args.vies_por_tp:
        print("\nNOTE: CONSTANT bias on the area cancels in the closure fraction — the")
        print("ratio (a0-at)/a0 is invariant to a multiplicative factor. Use --vies-por-tp")
        print("for the case that survives, which is the one that actually happens.")
    if args.ruido == 0 and args.vies == 0 and args.vies_por_tp == 0:
        print("\nThis is the control case: stage3/paired_new.py MUST return")
        print("CCC = 1.000 and bias = 0. Anything else = a matching error.")


if __name__ == "__main__":
    main()
