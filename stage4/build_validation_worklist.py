# -*- coding: utf-8 -*-
"""
stage4/build_validation_worklist.py — TESTING THE MIXED-PROVENANCE ASSUMPTION.

PROBLEM: 25 of the 46 analysable series mix timepoints with a CORRECTED area
(frames the triage judged badly segmented) and an AUTOMATIC area (frames judged
OK). Since closure is a ratio, a systematic difference between the two sources
biases mixed series — homogeneous ones it does not. The implicit assumption ("OK
in the visual triage ~ as accurate as corrected") had never been tested, and the
auto-vs-manual IoU of 0.262 on the bad frames justifies checking.

DESIGN: draws N frames classified OK (and therefore carrying an automatic area)
from ANALYSABLE MIXED series, with a fixed seed, and puts them in a worklist
identical in form to the others. The operator corrects them without knowing they
are 'OK' — the worklist does not carry the category. Afterwards,
stage4/validate_provenance.py compares the automatic against the corrected area.

DECISION CRITERION (declared BEFORE measuring):
  median |area| delta < 5%    -> the mixing is VALIDATED; declared in the Methods
  median delta >= 20%         -> there is bias; choose between correcting all the
                                 mixed series or restricting the analysis to the
                                 homogeneous ones
  between 5% and 20%          -> grey zone; report it and decide from the observed
                                 effect on the closure

Produces: stage4/validation_worklist.csv  (+ stage4/.validacao_gabarito.csv, do not
open beforehand)

⚠️ RUN ONCE, NOT REGENERABLE — IT DRAWS. Re-running it redraws the frames from the
current state and would break the correspondence with the corrections already made.
Same class as stage4/build_correction_worklist.py.
"""
import csv, os, random, sys
from collections import Counter

AREAS = "data/whst_areas_final.csv"
CLOS = "stage4/closure_final_por_serie.csv"
HUM = "data/inspecao_visual.csv"
AUTO = "data/whst_pass1_qc.csv"
OUT = "stage4/validation_worklist.csv"
GAB = "stage4/.validacao_gabarito.csv"
SEED = 4336348
N = 10

A = list(csv.DictReader(open(AREAS, encoding="utf-8-sig")))
C = {r["series_key"]: r for r in csv.DictReader(open(CLOS, encoding="utf-8-sig"))}
H = {r["whst_input_file"]: r for r in csv.DictReader(open(HUM, encoding="utf-8-sig"))}
auto = {r["whst_input_file"]: r for r in csv.DictReader(open(AUTO, encoding="utf-8-sig"))}

mistas = {k for k, v in C.items() if v["procedencia_area"] == "mista" and v["analisavel"] == "sim"}
elig = [r for r in A if r["series_key"] in mistas
        and r["fonte_area"] == "automatica"
        and H[r["whst_input_file"]]["categoria"] == "OK"]
print(f"analysable mixed series: {len(mistas)}")
print(f"eligible frames (OK + automatic): {len(elig)}")
if len(elig) < N:
    sys.exit(f"only {len(elig)} eligible, fewer than N={N}")

# deterministic draw, stratified by cell line so it does not concentrate
rng = random.Random(SEED)
por_linha = {}
for r in elig:
    por_linha.setdefault(r["cell_line"], []).append(r)
sel = []
# proportional quota, minimum 1 per cell line present
linhas = sorted(por_linha)
cotas = {cl: max(1, round(N * len(por_linha[cl]) / len(elig))) for cl in linhas}
while sum(cotas.values()) > N:
    cotas[max(cotas, key=lambda c: cotas[c])] -= 1
while sum(cotas.values()) < N:
    cotas[max(cotas, key=lambda c: len(por_linha[c]) - cotas[c])] += 1
for cl in linhas:
    pool = sorted(por_linha[cl], key=lambda r: r["whst_input_file"])
    sel += rng.sample(pool, min(cotas[cl], len(pool)))
sel.sort(key=lambda r: (r["cell_line"], r["analysis_unit"], int(r["timepoint_h"]), str(r["campo"])))

# ---- BLIND worklist: the same visible columns as the others, without the category ----
rows = []
for i, r in enumerate(sel, 1):
    a = auto[r["whst_input_file"]]
    rows.append({"ordem": i, "whst_input_file": r["whst_input_file"],
                 "cell_line": r["cell_line"], "analysis_unit": r["analysis_unit"],
                 "timepoint_h": r["timepoint_h"], "campo": r["campo"],
                 "eh_baseline": "SIM" if int(r["timepoint_h"]) == 0 else "",
                 "tarefa": "corrigir contorno",
                 "roi_auto": os.path.splitext(r["whst_input_file"])[0] + ".roi"})
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

with open(GAB, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["whst_input_file", "series_key", "area_pct_auto", "categoria_triagem"])
    for r in sel:
        w.writerow([r["whst_input_file"], r["series_key"], r["area_pct_auto"],
                    H[r["whst_input_file"]]["categoria"]])

print(f"\nproduced {OUT}: {len(rows)} frames (SEED={SEED})")
print(f"  by cell line: {dict(Counter(r['cell_line'] for r in sel))}")
print(f"  by timepoint: {dict(sorted(Counter(int(r['timepoint_h']) for r in sel).items()))}")
print(f"  distinct series: {len(set(r['series_key'] for r in sel))}")
print(f"  gabarito em {GAB} — NAO abrir antes de corrigir")
print("\nNo Fiji: stage4/whst_manual_correction.ijm -> passada '3 - validacao'")
print("Depois:  python stage4/validate_provenance.py")
