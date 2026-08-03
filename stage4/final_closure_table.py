# -*- coding: utf-8 -*-
"""
stage4/final_closure_table.py — the final deliverable of the WHST measurement:
closure fraction per series, computed from the CORRECTED AREAS
(data/whst_areas_final.csv).

closure(t) = (a0 - a_t) / a0     ; a0 = the baseline area (t=0)
  0   = the wound is at its initial size
  1   = complete closure (area 0)

Each series is labelled ANALYSABLE or not, with the reason, and the PROVENANCE of
the area is declared (corrected / automatic / mixed), so the statistical analysis
can run over any subset and test robustness.

Outputs:
  stage4/closure_final_por_serie.csv   (one row per series, with the curve)
  data/closure_final_longo.csv         (long format: 1 row per measurement, for stats)
"""
import csv, os, sys
from collections import defaultdict, Counter
import statistics as st

AREAS = "data/whst_areas_final.csv"
SER = "data/whst_series_analysis.csv"
if not os.path.isfile(AREAS):
    sys.exit(f"could not find {AREAS} (run stage4/apply_corrections.py)")

A = list(csv.DictReader(open(AREAS, encoding="utf-8-sig")))
meta = {r["series_key"]: r for r in csv.DictReader(open(SER, encoding="utf-8-sig"))}

bys = defaultdict(list)
for r in A:
    if r["area_pct_final"] == "":          # frame invalido: fora da serie
        continue
    bys[r["series_key"]].append(r)

TOL = 0.10


def curva(rs):
    d = defaultdict(list)
    for r in rs:
        d[int(r["timepoint_h"])].append(float(r["area_pct_final"]))
    return {tp: st.median(v) for tp, v in d.items()}


longo, linhas = [], []
for sk, rs in sorted(bys.items()):
    m = meta.get(sk, {})
    ab = curva(rs)
    tps = sorted(ab)
    fontes = Counter(r["fonte_area"] for r in rs)
    proc = ("corrigida" if all(f.startswith("corrigida") for f in fontes)
            else "automatica" if all(f.startswith("automatica") for f in fontes)
            else "mista")
    # RULE R2 (PROTOCOL section 3.1): t0 invalid with a valid sibling field at t0
    # -> use the sibling as the baseline. stage4/whst_series_analysis.py already
    # resolved WHICH field to borrow; here it has to be APPLIED, otherwise the
    # series is discarded as 'sem_baseline' and the two analyses disagree.
    emprestado = m.get("baseline_emprestado", "nao")
    if 0 not in ab and emprestado.startswith("sim"):
        c_irm = emprestado[emprestado.find("(") + 2: emprestado.find(")")]  # 'sim(c2)' -> '2'
        au = rs[0]["analysis_unit"]
        irm = [float(r["area_pct_final"]) for r in A
               if r["analysis_unit"] == au and int(r["timepoint_h"]) == 0
               and str(r["campo"]) == c_irm and r["area_pct_final"] != ""]
        if irm:
            ab[0] = st.median(irm)
            tps = sorted(ab)
    if 0 not in ab or ab[0] <= 0 or len(tps) < 2:
        linhas.append({"series_key": sk, "cell_line": rs[0]["cell_line"],
                       "analysis_unit": rs[0]["analysis_unit"], "campo": rs[0]["campo"],
                       "n_timepoints": len(tps), "timepoints": ";".join(map(str, tps)),
                       "area0_pct": round(ab.get(0, float("nan")), 3) if 0 in ab else "",
                       "closure_seq": "", "closure_final": "", "analisavel": "nao",
                       "motivo": "sem_baseline" if 0 not in ab else
                                 ("baseline_zero" if ab[0] <= 0 else "menos_de_2_timepoints"),
                       "procedencia_area": proc, "baseline_emprestado": emprestado})
        continue
    a0 = ab[0]
    seq = [(tp, (a0 - ab[tp]) / a0) for tp in tps]
    cv = [c for _, c in seq]
    range_ok = all(-0.05 <= c <= 1.05 for c in cv)
    mono_ok = all(cv[i] >= cv[i - 1] - TOL for i in range(1, len(cv)))
    plaus = range_ok and mono_ok
    mot = [] if plaus else ([] if range_ok else ["fora_[0,1]"]) + ([] if mono_ok else ["nao_monotonica"])
    linhas.append({"series_key": sk, "cell_line": rs[0]["cell_line"],
                   "analysis_unit": rs[0]["analysis_unit"], "campo": rs[0]["campo"],
                   "n_timepoints": len(tps), "timepoints": ";".join(map(str, tps)),
                   "area0_pct": round(a0, 3),
                   "closure_seq": ";".join(f"{tp}h:{c:.4f}" for tp, c in seq),
                   "closure_final": round(seq[-1][1], 4),
                   "analisavel": "sim" if plaus else "nao",
                   "motivo": "plausivel" if plaus else "+".join(mot),
                   "procedencia_area": proc, "baseline_emprestado": emprestado})
    for tp, c in seq:
        longo.append({"series_key": sk, "cell_line": rs[0]["cell_line"],
                      "analysis_unit": rs[0]["analysis_unit"], "campo": rs[0]["campo"],
                      "timepoint_h": tp, "area_pct": round(ab[tp], 3),
                      "closure_fraction": round(c, 4),
                      "procedencia_area": proc,
                      "analisavel": "sim" if plaus else "nao"})

for nome, dados in ((("stage4/closure_final_por_serie.csv"), linhas),
                    (("data/closure_final_longo.csv"), longo)):
    if dados:
        with open(nome, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dados[0].keys()))
            w.writeheader(); w.writerows(dados)

an = [x for x in linhas if x["analisavel"] == "sim"]
print(f"=== FINAL CLOSURE (corrected areas) ===")
print(f"  total series       : {len(linhas)}")
print(f"  ANALYSABLE         : {len(an)}")
print(f"  not analysable     : {len(linhas)-len(an)}  -> {dict(Counter(x['motivo'] for x in linhas if x['analisavel']=='nao'))}")
print(f"\n  by cell line       : {dict(Counter(x['cell_line'] for x in an))}")
print(f"  area provenance    : {dict(Counter(x['procedencia_area'] for x in an))}")
print(f"  borrowed baseline  : {sum(1 for x in an if x['baseline_emprestado']!='nao')}")
cf = [x["closure_final"] for x in an if x["closure_final"] != ""]
if cf:
    print(f"\n  closure at the last timepoint: median={st.median(cf):.3f}  "
          f"min={min(cf):.3f}  max={max(cf):.3f}")
    print(f"  series reaching complete closure (>=0.99): {sum(1 for c in cf if c >= 0.99)}")
print(f"\nSaved: stage4/closure_final_por_serie.csv ({len(linhas)}) and data/closure_final_longo.csv ({len(longo)})")
