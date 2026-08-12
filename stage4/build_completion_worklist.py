# -*- coding: utf-8 -*-
"""
stage4/build_completion_worklist.py — completes the correction of the MIXED series.

CONTEXT: the provenance validation (pass 3, n=10) showed that frames judged OK
carry no directional bias relative to the manual correction (sign test p=0.754),
but they do carry dispersion that shifts the closure by ~0.079 (median). Since
that is ~28% of the between-series standard deviation, the choice was to eliminate
the noise by correcting all automatic frames of the analysable mixed series,
rather than restricting the analysis to the homogeneous ones (which would have
cost half the series).

Of the 34 automatic frames in those series, 10 were already corrected in pass 3;
this worklist carries the remaining 24. Ordered by series (temporal context), with
the baselines first within each series — an error in a0 propagates to every
timepoint.

Produces: stage4/completion_worklist.csv   (Fiji: pass '4 - completar mistas')

⚠️ RUN ONCE, NOT REGENERABLE. It rebuilds the list from the CURRENT state of the
areas, in which the corrections are already applied. The deposited file is the
record of what was handed to the operator.
"""
import csv, os
from collections import Counter

AREAS = "data/whst_areas_final.csv"
CLOS = "stage4/closure_final_por_serie.csv"
VALP = "stage4/validation_provenance.csv"
AUTO = "data/whst_pass1_qc.csv"
OUT = "stage4/completion_worklist.csv"

A = list(csv.DictReader(open(AREAS, encoding="utf-8-sig")))
C = {r["series_key"]: r for r in csv.DictReader(open(CLOS, encoding="utf-8-sig"))}
auto = {r["whst_input_file"]: r for r in csv.DictReader(open(AUTO, encoding="utf-8-sig"))}
feitos = ({r["whst_input_file"] for r in csv.DictReader(open(VALP, encoding="utf-8-sig"))}
          if os.path.isfile(VALP) else set())

mist = {k for k, v in C.items() if v["analisavel"] == "sim" and v["procedencia_area"] == "mista"}
rest = [r for r in A if r["series_key"] in mist
        and r["fonte_area"].startswith("automatica")
        and r["whst_input_file"] not in feitos]
rest.sort(key=lambda r: (r["cell_line"], r["analysis_unit"], str(r["campo"]),
                         int(r["timepoint_h"])))   # t0 primeiro dentro da serie

rows = []
for i, r in enumerate(rest, 1):
    rows.append({"ordem": i, "whst_input_file": r["whst_input_file"],
                 "cell_line": r["cell_line"], "analysis_unit": r["analysis_unit"],
                 "timepoint_h": r["timepoint_h"], "campo": r["campo"],
                 "eh_baseline": "SIM" if int(r["timepoint_h"]) == 0 else "",
                 "tarefa": "corrigir contorno",
                 "roi_auto": os.path.splitext(r["whst_input_file"])[0] + ".roi"})
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"produced {OUT}: {len(rows)} frames")
print(f"  analysable mixed series: {len(mist)}  | touched by this list: "
      f"{len(set(r['series_key'] for r in rest))}")
print(f"  already corrected in pass 3: {len(feitos)}")
print(f"  baselines (t0): {sum(1 for r in rows if r['eh_baseline'])}  <- first: a0 propagates")
print(f"  by timepoint: {dict(sorted(Counter(int(r['timepoint_h']) for r in rows).items()))}")
print(f"  by cell line: {dict(Counter(r['cell_line'] for r in rows))}")
print("\nIn Fiji: stage4/whst_manual_correction.ijm -> pass '4 - completar mistas'")
print("Then:    python stage4/apply_corrections.py && python stage4/whst_series_analysis.py "
      "&& python stage4/final_closure_table.py")
