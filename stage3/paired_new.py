# -*- coding: utf-8 -*-
"""
stage3/paired_new.py — the paired analysis rebuilt: new model × reference standard.

IT REPLACES the old pairing rather than updating it. `paired_imagej_vs_ai_clean.csv`
holds 225 observations whose "AI" side came from the model trained on the partition
that LEAKED, and whose "ImageJ" side came from three annotators under a protocol
that stage 4 replaced. Editing the numbers in that file would be a patch; the thing
worth doing is redoing it.

WHAT CHANGES RELATIVE TO THE OLD ONE
  - AI side       : retrained leakage-free, predicted on the held-out test set
  - reference side: the supervised reference standard of stage 4 (59 series), not
                    measurements by 3 annotators
  - coverage      : HUVEC **and SKOV-3** — the old one was HUVEC only, which is
                    what the coverage criticism was about
  - independence  : the models never saw the reference standard, neither in
                    training nor in its construction. The classical arm (automatic
                    WHST) does NOT have that property — see the incorporation bias
                    in `benchmark_classico.py`

WHAT IT PRODUCES
  1. stage3/paired_new_longo.csv       — one row per (series, field, timepoint)
  2. stage3/supplementary_table_S2.csv — the observations with negative closure, in
                                  the format of the manuscript table (regenerated,
                                  not corrected)
  3. an agreement report        — Pearson, Spearman, Lin's CCC, Bland-Altman
  4. the systematic-vs-noise diagnosis, via stage3/diagnostico_c7.py

  python stage3/paired_new.py --run yolo11m-seg_black_coco_seed42
"""
import argparse
import csv
import math
import os
import statistics as st
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
os.chdir(os.path.dirname(AQUI))

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def ler(p):
    if not os.path.isfile(p):
        sys.exit(f"could not find {p}")
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ estatísticas
def lin_ccc(x, y):
    mx, my = st.mean(x), st.mean(y)
    vx = sum((a - mx) ** 2 for a in x) / len(x)
    vy = sum((b - my) ** 2 for b in y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / len(x)
    den = vx + vy + (mx - my) ** 2
    return 2 * cov / den if den else float("nan")


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num_ = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num_ / den if den else float("nan")


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            media = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = media
            i = j + 1
        return r
    return pearson(rank(x), rank(y))


def bland_altman(x, y):
    """x = referência, y = AI. Diferença = y − x."""
    d = [b - a for a, b in zip(x, y)]
    mu = st.mean(d)
    sd = st.stdev(d) if len(d) > 1 else float("nan")
    return {"vies": mu, "dp": sd, "loa_inf": mu - 1.96 * sd, "loa_sup": mu + 1.96 * sd}


# --------------------------------------------------- plausibility of a series
# PRE-SPECIFIED CRITERION. These three conditions and these constants come from
# `stage4/whst_series_analysis.py::closure`, frozen in stage 4 BEFORE any model
# output existed. They are not a cutpoint chosen now while looking at the result —
# reusing them is what keeps this filter from becoming p-hacking.
FAIXA_MIN, FAIXA_MAX = -0.05, 1.05     # physically admissible closure
TOL_MONOTONIA = 0.10                   # a wound only closes, within a noise tolerance
SALTO_ABSURDO = 1.5                    # |closure| above this is a failure, not a measurement


def serie_plausivel(seq):
    """seq = [(tp, closure)] in order, including tp=0 with closure 0.

    Returns (plausible, reason) exactly as in stage 4. The reason strings are kept
    in Portuguese on purpose: they are written to the `ai_motivo` column of the
    deposited CSVs.
    """
    cvals = [c for tp, c in seq if tp > 0]
    if not cvals:
        return False, "serie_1_ponto"
    vals = [c for _, c in seq]
    faixa = all(FAIXA_MIN <= c <= FAIXA_MAX for c in cvals)
    mono = all(vals[i] >= vals[i - 1] - TOL_MONOTONIA for i in range(1, len(vals)))
    sem_salto = all(abs(c) <= SALTO_ABSURDO for c in cvals)
    motivo = []
    if not faixa:
        motivo.append("fora_[0,1]")
    if not mono:
        motivo.append("nao_monotonica")
    if not sem_salto:
        motivo.append("salto_absurdo")
    return (faixa and mono and sem_salto), ("+".join(motivo) or "plausivel")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name (reads stage3/areas/<run>.csv)")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    areas_p = os.path.join("stage3", "areas", f"{args.run}.csv")
    AREAS = {r["arquivo"]: r for r in ler(areas_p)}
    INSP = ler("data/inspecao_visual.csv")
    REF = ler("data/closure_final_longo.csv")
    print(f"run              : {args.run}")
    print(f"predicted areas  : {len(AREAS)}")
    print(f"reference (long) : {len(REF)} observations")

    # ---- test-set image -> (series, field, timepoint)
    # data/inspecao_visual.csv has NO series_key, but it has whst_input_file, which is the
    # primary key of data/whst_areas_final.csv, where series_key lives. Direct join
    # on that field; the md5 prefix is only the fallback, in case the name varied.
    WA = {r["whst_input_file"]: r for r in ler("data/whst_areas_final.csv")}
    por_md5 = {os.path.basename(k).split("__")[0]: r for k, r in WA.items()}

    chave, via = {}, {"direto": 0, "md5": 0, "sem_padrao": 0}
    for r in INSP:
        t = r.get("test_image")
        if not t:
            continue
        w = WA.get(r.get("whst_input_file", ""))
        if w:
            via["direto"] += 1
        else:
            pref = os.path.basename(r["sorted_basename"]).rsplit("__", 1)[-1].split(".")[0]
            w = por_md5.get(pref)
            via["md5" if w else "sem_padrao"] += 1
        if w:
            chave[t] = (w["series_key"], w["campo"], w["timepoint_h"], r["categoria"])
    print(f"join with the reference: {via}")

    ligadas = {k: v for k, v in chave.items() if v[0] and k in AREAS}
    print(f"images linked    : {len(ligadas)} (reference ∩ test set ∩ predictions)")
    if not ligadas:
        sys.exit("no links at all — check the test_image column of data/inspecao_visual.csv")

    # ---- predicted area per (series, field, tp)
    ai_area = {}
    for arq, (sk, campo, tp, cat) in ligadas.items():
        ai_area[(sk, str(campo), str(int(float(tp))))] = num(AREAS[arq]["area_pct"])

    # ---- AI closure, the SAME formula and the SAME unit as the reference
    por_serie = {}
    for (sk, campo, tp), a in ai_area.items():
        por_serie.setdefault((sk, campo), {})[tp] = a
    # DEGENERATE BASELINE (raised in the code review of 2026-07-28).
    # The `a0 <= 0` guard does not catch the case that actually happens: a0
    # POSITIVE but tiny — the model barely detected the wound at t=0 — which makes
    # the ratio (a0-at)/a0 blow up. That is how automatic WHST reached -179.
    #
    # The answer is NOT to exclude: discarding the series where the model does
    # badly is selecting on the outcome, and would inflate agreement artificially.
    # Here the series is FLAGGED by the pre-specified criterion of stage 4, enters
    # the primary analysis, and the restricted analysis appears only as a declared
    # sensitivity.
    ai_closure, plaus_serie = {}, {}
    sem_base = 0
    for (sk, campo), tps in por_serie.items():
        a0 = tps.get("0")
        if a0 is None or a0 <= 0:
            sem_base += 1
            continue
        seq = [(0, 0.0)]
        for tp, at in sorted(tps.items(), key=lambda kv: int(kv[0])):
            if tp == "0":
                continue
            c = (a0 - at) / a0
            ai_closure[(sk, campo, tp)] = c
            seq.append((int(tp), c))
        ok, motivo = serie_plausivel(seq)
        plaus_serie[(sk, campo)] = (ok, motivo, a0)
    print(f"series with no predictable baseline: {sem_base}")

    ruins = {k: v for k, v in plaus_serie.items() if not v[0]}
    print(f"series with an implausible AI trajectory: {len(ruins)} of {len(plaus_serie)}"
          f"  (criterion frozen in stage 4, not chosen now)")
    for (sk, campo), (_, motivo, a0) in sorted(ruins.items())[:6]:
        print(f"   {sk[:38]:<38} field {campo:<3} {motivo:<26} area₀ = {a0:.3f}%")

    ref_closure = {(r["series_key"], str(r["campo"]), str(int(float(r["timepoint_h"]))))
                   : num(r["closure_fraction"])
                   for r in REF if r.get("analisavel") == "sim"}

    pares = []
    for k, v in ai_closure.items():
        rv = ref_closure.get(k)
        if rv is None or v is None:
            continue
        ok, motivo, _ = plaus_serie.get((k[0], k[1]), (True, "plausivel", None))
        pares.append({"series_key": k[0], "campo": k[1], "timepoint_h": k[2],
                      "referencia": rv, "ai": v, "diferenca": v - rv,
                      "ai_serie_plausivel": "sim" if ok else "nao",
                      "ai_motivo": motivo})
    pares.sort(key=lambda p: (p["series_key"], p["campo"], int(p["timepoint_h"])))
    print(f"\nPAIRED OBSERVATIONS: {len(pares)}")
    if len(pares) < 3:
        sys.exit("not enough pairs")

    cl = {p["series_key"].split("||")[0] for p in pares}
    linhas_cel = {}
    for r in REF:
        linhas_cel[r["series_key"]] = r["cell_line"]
    from collections import Counter
    print("by cell line:",
          dict(Counter(linhas_cel.get(p["series_key"], "?") for p in pares)))

    x = [p["referencia"] for p in pares]
    y = [p["ai"] for p in pares]
    ba = bland_altman(x, y)
    print("\n" + "=" * 70)
    print("AGREEMENT — AI × reference standard")
    print("=" * 70)
    print(f"  n .................. {len(pares)}")
    print(f"  Pearson r .......... {pearson(x, y):+.4f}")
    print(f"  Spearman rho ....... {spearman(x, y):+.4f}")
    print(f"  Lin's CCC .......... {lin_ccc(x, y):+.4f}")
    print(f"  Bland-Altman bias .. {ba['vies']:+.4f}  (AI − reference)")
    print(f"  LoA 95% ............ [{ba['loa_inf']:+.4f}, {ba['loa_sup']:+.4f}]")
    print("\n  The CCC penalises systematic bias; Pearson does not. If Pearson is high")
    print("  and the CCC low, the model tracks the trend but gets the level wrong — and")
    print("  a claim of interchangeability does not hold.")

    # ---- SENSITIVITY, not the primary analysis -----------------------------
    sub = [p for p in pares if p["ai_serie_plausivel"] == "sim"]
    print("\n" + "-" * 70)
    print("SENSITIVITY — restricted to series with a plausible AI trajectory")
    print("-" * 70)
    if len(sub) == len(pares):
        print("  Every series is plausible; the sensitivity coincides with the primary.")
    elif len(sub) < 3:
        print(f"  Only {len(sub)} pair(s) survive — not enough for statistics.")
    else:
        xs = [p["referencia"] for p in sub]
        ys = [p["ai"] for p in sub]
        bs = bland_altman(xs, ys)
        print(f"  n .......... {len(sub)} of {len(pares)} "
              f"({100*len(sub)/len(pares):.0f}%)")
        print(f"  Pearson r .. {pearson(xs, ys):+.4f}   (primary: {pearson(x, y):+.4f})")
        print(f"  Lin's CCC .. {lin_ccc(xs, ys):+.4f}   (primary: {lin_ccc(x, y):+.4f})")
        print(f"  bias ....... {bs['vies']:+.4f}   (primary: {ba['vies']:+.4f})")
        print("\n  ⚠️ THIS IS NOT THE PRIMARY ANALYSIS. Excluding the series where the")
        print("  model failed is selection on the outcome and inflates agreement. Report")
        print("  it as a declared sensitivity, and the primary as the manuscript number.")

    pref = args.out_prefix or ""
    dest = os.path.join("stage3", f"{pref}paired_new_longo.csv")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["series_key", "campo", "timepoint_h",
                                          "referencia", "ai", "diferenca",
                                          "ai_serie_plausivel", "ai_motivo"])
        w.writeheader()
        for p in pares:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in p.items()})
    print(f"\nwritten: {dest}")

    # -------------------------------------------------- Supplementary Table S2
    neg = [p for p in pares if p["referencia"] < 0 or p["ai"] < 0]
    s2 = os.path.join("stage3", f"{pref}supplementary_table_S2.csv")
    with open(s2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Series", "Field", "Timepoint (h)", "Reference standard", "AI"])
        for p in neg:
            w.writerow([p["series_key"], p["campo"], p["timepoint_h"],
                        f"{p['referencia']:+.4f}", f"{p['ai']:+.4f}"])
    conc = sum(1 for p in neg if p["referencia"] * p["ai"] > 0)
    print(f"written: {s2}")
    print(f"\nSUPPLEMENTARY TABLE S2 — {len(neg)} of {len(pares)} observations "
          f"({100*len(neg)/len(pares):.1f}%) with negative closure")
    print(f"   agreeing in sign (noise)      : {conc}")
    print(f"   disagreeing                   : {len(neg)-conc}")
    if not neg:
        print("\n   EMPTY TABLE. Under the leakage-free partition there was no negative")
        print("   closure — the corresponding limitation and discussion paragraphs have to")
        print("   be REWRITTEN, not updated. Do not leave old text carrying new numbers.")

    # ------------------------------------- systematic-vs-noise diagnosis
    d7 = os.path.join("stage3", f"{pref}paired_para_c7.csv")
    with open(d7, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["analysis_unit", "campo", "timepoint_h", "imagej", "ai"])
        for p in pares:
            w.writerow([p["series_key"], p["campo"], p["timepoint_h"],
                        p["referencia"], p["ai"]])
    print(f"written: {d7}")
    print(f"\nNext:  python stage3/diagnostico_c7.py {d7}")


if __name__ == "__main__":
    main()
