
# -*- coding: utf-8 -*-
"""
stage3/agreement_final.py — every agreement number the manuscript states.

WHY THIS SCRIPT EXISTS. The agreement block of the Results, §2.8 of the Methods
and the closing sentence of the Abstract still carried the 225-observation
analysis: the DEPLOYED model (trained on the partition that leaked) against three
annotators' ImageJ measurements, with no restriction to the held-out partition.
The `analysis/paired_analysis.py` that produced it does not mention `train` once
and does not filter by partition — the number measures the model on data it may
have seen. That is the leakage defect applied to the agreement analysis, not a
merely superseded result.

`stage3/paired_new.py` had already replaced that pairing: a leakage-free model
predicted on the held-out set, against the supervised reference standard of stage
4, and covering both cell lines. The Discussion and the Limitations had been
migrated; the Methods, the Results, the captions and the Abstract had not. The
manuscript ended up asserting and denying the same equivalence in different
sections.

This script produces the complete set needed to rewrite them — what
`stage3/stats_for_224.py` already gave (Pearson, Spearman, CCC, bias, LoA,
TOST) plus what only the Results needs: regression, % within the LoA, descriptives
by timepoint and cell line, a paired test with Cohen's d, and the count of
negative closures that feeds Supplementary Table S2.

ONE DEFINITION ONLY. `ccc` and `tost` come from `stage3/_agreement_stats.py`,
the same module `stage3/stats_for_224.py` imports. Two implementations of
the same statistic would diverge silently, and the divergence would surface as the
Results and the Discussion disagreeing with each other.

THE SET ANALYSED. The 97 observations present in all 10 runs (5 seeds × 2 arms).
Measuring the arms on different sets would make the difference between them
uninterpretable: part of it would be composition, not performance.

    python stage3/agreement_final.py
"""
import csv
import json
import os
import statistics as st
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(AQUI))
sys.path.insert(0, AQUI)

import numpy as np                                      # noqa: E402
from scipy import stats as sps                          # noqa: E402

from _agreement_stats import ccc, tost               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEEDS = [42, 43, 44, 45, 46]
BRACOS = {"YOLO M": "yolo11m-seg_black_coco_seed{}", "U-Net": "unet_black_seed{}"}
MARGENS = [0.10, 0.05]
DESTINO = os.path.join("stage3", "agreement_final.json")
DEST_CSV = os.path.join("stage3", "agreement_final_long.csv")

# Values the manuscript's Discussion already reports, from
# `stage3/stats_for_224.py`. This script reconstructs the selection of
# observations by its own route; if the reconstruction diverged, the two passages
# of the paper would start disagreeing. The guard fails loudly rather than let
# that through.
ESPERADO_YOLO = {"r": 0.8196, "ccc": 0.8032, "vies": 0.0534,
                 "loa_lo": -0.2883, "loa_hi": 0.3950}
TOL = 5e-4


def carrega():
    """(arm, seed) -> {key: (reference, ai)}, plus plausibility and cell line."""
    dados, plaus = {}, {}
    for braco, molde in BRACOS.items():
        for s in SEEDS:
            p = os.path.join("stage3", f"cmp_{molde.format(s)}_paired_new_long.csv")
            if not os.path.isfile(p):
                sys.exit(f"missing: {p}")
            with open(p, encoding="utf-8-sig") as fh:
                linhas = list(csv.DictReader(fh))
            dados[(braco, s)] = {
                (x["series_key"], x["campo"], x["timepoint_h"]):
                (float(x["referencia"]), float(x["ai"])) for x in linhas}
            plaus[(braco, s)] = {
                (x["series_key"], x["campo"], x["timepoint_h"]):
                (x["ai_serie_plausivel"] == "sim") for x in linhas}
    with open("data/closure_final_long.csv", encoding="utf-8-sig") as fh:
        linhagem = {(r["series_key"], r["campo"], r["timepoint_h"]): r["cell_line"]
                    for r in csv.DictReader(fh)}
    return dados, plaus, linhagem


def por_seed(par):
    """statistics for one seed over a list of (reference, ai) pairs."""
    ref = [a for a, _ in par]
    ai = [b for _, b in par]
    dif = [b - a for a, b in par]
    m, dp = st.mean(dif), st.stdev(dif)
    lo, hi = m - 1.96 * dp, m + 1.96 * dp
    reg = sps.linregress(ref, ai)
    return {
        "n": len(par),
        "r": float(sps.pearsonr(ref, ai)[0]),
        "rho": float(sps.spearmanr(ref, ai)[0]),
        "ccc": ccc(ref, ai),
        "vies": m,
        "dp": dp,
        "mae": float(np.mean(np.abs(dif))),
        "loa_lo": lo,
        "loa_hi": hi,
        "dentro_loa": 100.0 * sum(1 for d in dif if lo <= d <= hi) / len(dif),
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "tost": {f"{d:.2f}": tost(dif, d) for d in MARGENS},
    }


def ms(vals, casas=4):
    """mean ± SD OVER THE SEEDS, plus the range.

    The 5 estimates come from the SAME 97 observations, changing only the
    initialisation — they are not independent samples to be pooled
    meta-analytically. Hence the arithmetic mean, and not Fisher's z transform for
    r: what is being described is the seed-to-seed variability of the estimator,
    which is a statement about reproducibility. For p-values the mean is not
    interpretable, which is why the range travels with it — the range is what the
    text should quote.
    """
    return {"media": round(st.mean(vals), casas),
            "dp": round(st.stdev(vals), casas) if len(vals) > 1 else 0.0,
            "min": round(min(vals), casas), "max": round(max(vals), casas)}


def main():
    dados, plaus, linhagem = carrega()
    comuns = None
    for d in dados.values():
        comuns = set(d) if comuns is None else comuns & set(d)
    comuns = sorted(comuns)
    print(f"{len(comuns)} observations common to all {len(dados)} runs")

    sem_linhagem = [k for k in comuns if k not in linhagem]
    if sem_linhagem:
        sys.exit(f"{len(sem_linhagem)} observation(s) with no cell line in "
                 f"data/closure_final_long.csv — the join is broken, e.g.: {sem_linhagem[:2]}")

    # the reference standard depends on neither the seed nor the arm; the
    # per-timepoint descriptives use the one from a single run, so this has to be
    # true rather than assumed
    base = dados[("YOLO M", SEEDS[0])]
    divergentes = [k for k in comuns for d in dados.values()
                   if abs(d[k][0] - base[k][0]) > 1e-9]
    if divergentes:
        sys.exit(f"the reference standard differs between runs in "
                 f"{len(set(divergentes))} observation(s) — e.g.: {divergentes[0]}")

    # ── composition ───────────────────────────────────────────────────────────
    comp = {}
    for k in comuns:
        lg = linhagem[k]
        c = comp.setdefault(lg, {"n": 0, "series": set(), "timepoints": {}})
        c["n"] += 1
        c["series"].add(k[0])
        c["timepoints"][k[2]] = c["timepoints"].get(k[2], 0) + 1
    for lg, c in comp.items():
        c["series"] = len(c["series"])
        c["timepoints"] = dict(sorted(c["timepoints"].items(), key=lambda x: int(x[0])))
        print(f"  {lg}: {c['n']} observations · {c['series']} series · {c['timepoints']}")

    out = {
        "fonte": "stage3/cmp_*_paired_new_long.csv (stage3/paired_new.py)",
        "padrao": "supervised reference standard, stage 4",
        "n_observacoes": len(comuns),
        "n_series": len({k[0] for k in comuns}),
        "seeds": SEEDS,
        "margens_tost": MARGENS,
        "composicao": comp,
        "bracos": {},
        "por_tempo": {},
        "por_linhagem": {},
        "negativas": {},
        "sensibilidade_plausiveis": {},
    }

    # ── per arm ───────────────────────────────────────────────────────────────
    for braco in BRACOS:
        seeds = {s: por_seed([dados[(braco, s)][k] for k in comuns]) for s in SEEDS}
        ag = {}
        for campo in ("r", "rho", "ccc", "vies", "dp", "mae", "loa_lo", "loa_hi",
                      "dentro_loa", "slope", "intercept"):
            ag[campo] = ms([seeds[s][campo] for s in SEEDS])
        ag["n"] = seeds[SEEDS[0]]["n"]
        ag["tost"] = {}
        for d in MARGENS:
            k = f"{d:.2f}"
            ts = [seeds[s]["tost"][k] for s in SEEDS]
            ag["tost"][k] = {
                "equivalentes": sum(1 for t in ts if t["equivalente"]),
                "de": len(ts),
                "p_max": round(max(t["p"] for t in ts), 4),
                "ic90_media": [round(st.mean(t["lo"] for t in ts), 4),
                               round(st.mean(t["hi"] for t in ts), 4)],
            }
        ag["por_seed"] = {str(s): {c: round(v, 4) for c, v in seeds[s].items()
                                   if c != "tost"} for s in SEEDS}
        out["bracos"][braco] = ag
        print(f"\n{braco}: r {ag['r']['media']:+.4f}±{ag['r']['dp']:.4f} · "
              f"CCC {ag['ccc']['media']:+.4f}±{ag['ccc']['dp']:.4f} · "
              f"bias {ag['vies']['media']:+.4f}±{ag['vies']['dp']:.4f} · "
              f"LoA [{ag['loa_lo']['media']:+.4f}, {ag['loa_hi']['media']:+.4f}] · "
              f"within {ag['dentro_loa']['media']:.1f}% · "
              f"slope {ag['slope']['media']:.4f} intercept {ag['intercept']['media']:.4f}")
        for d in MARGENS:
            t = ag["tost"][f"{d:.2f}"]
            print(f"    TOST ±{d:.2f}: {t['equivalentes']}/{t['de']} seeds  "
                  f"max p {t['p_max']:.4f}")

    # the guard: the reconstruction has to reproduce what the paper already states
    y = out["bracos"]["YOLO M"]
    ruins = {c: (y[c]["media"], v) for c, v in ESPERADO_YOLO.items()
             if abs(y[c]["media"] - v) > TOL}
    if ruins:
        sys.exit(f"DIVERGES from what the manuscript reports: {ruins}")
    print(f"\n  guard: matches the manuscript on {len(ESPERADO_YOLO)} statistics "
          f"(tolerance {TOL})")

    # ── descriptives by cell line × timepoint ─────────────────────────────────
    for braco in BRACOS:
        bloco = {}
        for lg in sorted(comp):
            for tp in comp[lg]["timepoints"]:
                ks = [k for k in comuns if linhagem[k] == lg and k[2] == tp]
                ref = [dados[(braco, SEEDS[0])][k][0] for k in ks]
                # the reference does not depend on the seed; the AI does — mean over seeds
                ai_por_seed = [[dados[(braco, s)][k][1] for k in ks] for s in SEEDS]
                dif_por_seed = [[a - r for a, r in zip(v, ref)] for v in ai_por_seed]
                ps, ds = [], []
                for ai, dif in zip(ai_por_seed, dif_por_seed):
                    if len(dif) > 1 and st.stdev(dif) > 0:
                        ps.append(float(sps.ttest_rel(ai, ref)[1]))
                        ds.append(st.mean(dif) / st.stdev(dif))   # paired d_z
                if len(ks) < 2:
                    print(f"  ⚠️  {lg} {tp}h has {len(ks)} observation — no SD")
                bloco[f"{lg} {tp}h"] = {
                    "n": len(ks),
                    "ref_media_pct": round(100 * st.mean(ref), 1),
                    "ref_dp_pct": round(100 * st.stdev(ref), 1) if len(ref) > 1 else 0.0,
                    "ai_media_pct": ms([100 * st.mean(v) for v in ai_por_seed], 1),
                    "ai_dp_pct": (ms([100 * st.stdev(v) for v in ai_por_seed], 1)
                                  if len(ks) > 1 else None),
                    "dif_media": ms([st.mean(v) for v in dif_por_seed]),
                    "p_pareado": ms(ps) if ps else None,
                    "cohen_dz": ms(ds) if ds else None,
                }
        out["por_tempo"][braco] = bloco

    # ── full agreement statistics BY CELL LINE ───────────────────────────────
    # R1.5 asked for the reliability claims to be disaggregated by cell line, and
    # the earlier draft of the response letter said they were when they were not.
    # Same procedure as the pooled block, so the two are comparable: each statistic
    # is computed WITHIN a seed over that line's observations, then averaged over
    # the five seeds. SKOV-3 carries 15 observations; its concordance coefficient
    # and limits of agreement are reported with the n beside them and should be
    # read as indicative, not as an estimate of the same precision as the HUVEC one.
    for braco in BRACOS:
        bloco = {}
        for lg in sorted(comp):
            ks = [k for k in comuns if linhagem[k] == lg]
            porseed = {s_: por_seed([dados[(braco, s_)][k] for k in ks]) for s_ in SEEDS}
            ag = {c: ms([porseed[s_][c] for s_ in SEEDS])
                  for c in ("r", "rho", "ccc", "vies", "dp",
                            "loa_lo", "loa_hi", "dentro_loa")}
            ag["n"] = len(ks)
            bloco[lg] = ag
        out["por_linhagem"][braco] = bloco

    print("")
    print("=== AGREEMENT BY CELL LINE ===")
    for braco in BRACOS:
        for lg, v in out["por_linhagem"][braco].items():
            print(f"  {braco:<8} {lg:<8} n={v['n']:<4} "
                  f"r {v['r']['media']:+.3f}+-{v['r']['dp']:.3f}  "
                  f"CCC {v['ccc']['media']:+.3f}+-{v['ccc']['dp']:.3f}  "
                  f"bias {v['vies']['media']:+.4f}  "
                  f"LoA {v['loa_lo']['media']:+.3f} to {v['loa_hi']['media']:+.3f}")

    # ── negative closures (feeds Supplementary Table S2) ──────────────────────
    neg_ref = sorted(k for k in comuns if dados[("YOLO M", SEEDS[0])][k][0] < 0)
    for braco in BRACOS:
        por = {}
        for s in SEEDS:
            ks = [k for k in comuns if dados[(braco, s)][k][1] < 0]
            por[str(s)] = [{"series_key": k[0], "campo": k[1], "timepoint_h": k[2],
                            "linhagem": linhagem[k],
                            "referencia": round(dados[(braco, s)][k][0], 4),
                            "ai": round(dados[(braco, s)][k][1], 4)} for k in sorted(ks)]
        out["negativas"][braco] = {"por_seed": por,
                                   "faixa": [min(len(v) for v in por.values()),
                                             max(len(v) for v in por.values())]}
    out["negativas"]["referencia"] = [
        {"series_key": k[0], "campo": k[1], "timepoint_h": k[2],
         "linhagem": linhagem[k],
         "referencia": round(dados[("YOLO M", SEEDS[0])][k][0], 4)} for k in neg_ref]
    print(f"\n  negative closures — reference: {len(neg_ref)}; "
          + " · ".join(f"{b}: {out['negativas'][b]['faixa'][0]}–"
                       f"{out['negativas'][b]['faixa'][1]} of {len(comuns)}"
                       for b in BRACOS))

    # ── sensitivity: the plausible series only ────────────────────────────────
    # `stage3/paired_new.py` flags as implausible any series whose closure
    # trajectory is not monotonic — the sign of a badly segmented frame, typically
    # the baseline, which enters the denominator of every later timepoint. The
    # primary analysis RETAINS those series: discarding them would select the cases
    # where the model happened to be right, which is choosing the result. The
    # question here is a different and legitimate one — does the conclusion depend
    # on them?
    plausivel_sempre = [k for k in comuns if all(plaus[bs][k] for bs in dados)]
    print(f"\n  sensitivity: {len(plausivel_sempre)} of {len(comuns)} observations "
          f"in series plausible across all 10 runs")
    for braco in BRACOS:
        if len(plausivel_sempre) < 3:
            out["sensibilidade_plausiveis"][braco] = None
            continue
        seeds = {s: por_seed([dados[(braco, s)][k] for k in plausivel_sempre])
                 for s in SEEDS}
        bloco = {c: ms([seeds[s][c] for s in SEEDS])
                 for c in ("r", "ccc", "vies", "loa_lo", "loa_hi")}
        bloco["n"] = len(plausivel_sempre)
        bloco["tost"] = {}
        for d in MARGENS:
            ts = [seeds[s]["tost"][f"{d:.2f}"] for s in SEEDS]
            bloco["tost"][f"{d:.2f}"] = {
                "equivalentes": sum(1 for t in ts if t["equivalente"]), "de": len(ts)}
        out["sensibilidade_plausiveis"][braco] = bloco
        print(f"    {braco}: r {bloco['r']['media']:+.4f} · "
              f"CCC {bloco['ccc']['media']:+.4f} · bias {bloco['vies']['media']:+.4f} · "
              + " · ".join(f"TOST ±{d:.2f} {bloco['tost'][f'{d:.2f}']['equivalentes']}/5"
                           for d in MARGENS))

    # ── long CSV, for Figures 3-5 ─────────────────────────────────────────────
    with open(DEST_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series_key", "campo", "timepoint_h", "cell_line", "braco",
                    "seed", "referencia", "ai", "diferenca"])
        for braco in BRACOS:
            for s in SEEDS:
                for k in comuns:
                    r, a = dados[(braco, s)][k]
                    w.writerow([k[0], k[1], k[2], linhagem[k], braco, s,
                                f"{r:.6f}", f"{a:.6f}", f"{a-r:.6f}"])
    print(f"  wrote {DEST_CSV} "
          f"({len(comuns)*len(SEEDS)*len(BRACOS)} rows — the data behind Figures 3-5)")

    with open(DESTINO, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"  wrote {DESTINO}")


if __name__ == "__main__":
    main()
