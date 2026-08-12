# -*- coding: utf-8 -*-
"""
stage4/build_correction_worklist.py — builds the final manual-correction list,
ordered by series (temporal context), and draws the BLINDED RE-CORRECTION subset.

REQUIREMENT — intra-observer reproducibility:
  15 images are drawn with a FIXED seed (SEED=4336348, the manuscript number) to be
  corrected TWICE, with an interval, without the observer knowing which they are.
  -> the worklist handed to Fiji (stage4/correction_worklist.csv) does not mark them.
  -> the drawn list lives in stage4/.recorrecao_oculta.csv (a separate file the
     observer must not open); the second-pass macro reads that file.
  Pass 1 -> whst_output/rois_corrected/
  Pass 2 -> whst_output/rois_blind_repeat/
  Agreement (IoU + CCC) is computed by stage4/correction_agreement.py.

Ordering: cell_line -> analysis_unit -> timepoint -> field, so the observer walks
the whole series in sequence and keeps the temporal context.

Outputs:
  stage4/correction_worklist.csv      (handed to Fiji; NO re-correction marking)
  stage4/.recorrecao_oculta.csv       (the 15 drawn; do not open before finishing)

⚠️ RUN ONCE, NOT REGENERABLE. The worklist is the record of what was actually
handed to the operator. Re-running it today rebuilds the list from the CURRENT
state of data/visual_triage.csv, in which most corrections are already applied,
so it produces a shorter and different list. The deposited file is the historical
artefact; do not overwrite it.
"""
import csv, os, random

HUM = "data/visual_triage.csv"
AUTO = "data/whst_pass1_qc.csv"
OUT = "stage4/correction_worklist.csv"
HIDDEN = "stage4/.recorrecao_oculta.csv"
SEED = 4336348          # the manuscript number -> a deterministic draw
N_RECOR = 15

hum = list(csv.DictReader(open(HUM, encoding="utf-8-sig")))
auto = {r["whst_input_file"]: r for r in csv.DictReader(open(AUTO, encoding="utf-8-sig"))}

if any(r["categoria"] == "AMBIGUO" for r in hum):
    raise SystemExit("ABORTADO: ainda ha AMBIGUO nao adjudicado. "
                     "Rode stage1/adjudicate_ambiguous.py --template/--apply e "
                     "stage4/whst_series_analysis.py antes desta etapa.")
if "na_lista_correcao" not in hum[0]:
    raise SystemExit("ABORTADO: rode stage4/whst_series_analysis.py primeiro.")

sel = [r for r in hum if r["na_lista_correcao"] == "sim"]
rev = [r for r in hum if r.get("na_lista_revisao") == "sim"]

# frames whose validity is DECIDED during the correction itself (annotated as
# "noisy but segmentable"): tracing the contour = valid; skipping = invalid
RESOLVER = set()
if os.path.isfile("data/annotation_report.csv"):
    for _r in csv.DictReader(open("data/annotation_report.csv", encoding="utf-8-sig")):
        if _r.get("motivo_classificado") == "ruidosa_recuperavel":
            RESOLVER.add(_r["whst_input_file"])


def key(r):
    a = auto[r["whst_input_file"]]
    return (r["cell_line"], a["analysis_unit"], int(r["timepoint_h"]), str(r["campo"]))


sel.sort(key=key)

# ---- deterministic and BLIND draw ----
rng = random.Random(SEED)
n_rec = min(N_RECOR, len(sel))
recor = sorted(rng.sample([r["whst_input_file"] for r in sel], n_rec))

rows = []
for i, r in enumerate(sel, 1):
    a = auto[r["whst_input_file"]]
    rows.append({"ordem": i, "whst_input_file": r["whst_input_file"],
                 "cell_line": r["cell_line"], "analysis_unit": a["analysis_unit"],
                 "timepoint_h": r["timepoint_h"], "campo": r["campo"],
                 "eh_baseline": "SIM" if int(r["timepoint_h"]) == 0 else "",
                 "categoria_visual": r["categoria"] + ("/" + r["subtipo"] if r["subtipo"] else ""),
                 "tarefa": ("DECIDIR VALIDADE: se der p/ tracar o contorno e valida; "
                            "se nao, Select None (registra 'pulada')"
                            if r["whst_input_file"] in RESOLVER else "corrigir contorno"),
                 "serie_decisao": r["serie_decisao"], "area_pct_auto": a["area_pct"],
                 "roi_auto": os.path.splitext(r["whst_input_file"])[0] + ".roi"})
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- REVIEW list (series with an invalid frame): a human decision, not a correction ----
rev.sort(key=key)
rrows = []
for i, r in enumerate(rev, 1):
    a = auto[r["whst_input_file"]]
    rrows.append({"ordem": i, "whst_input_file": r["whst_input_file"],
                  "cell_line": r["cell_line"], "analysis_unit": a["analysis_unit"],
                  "timepoint_h": r["timepoint_h"], "campo": r["campo"],
                  "eh_baseline": "SIM" if int(r["timepoint_h"]) == 0 else "",
                  "categoria_visual": r["categoria"] + ("/" + r["subtipo"] if r["subtipo"] else ""),
                  "acao": ("decidir manter/excluir o frame" if r["categoria"] == "IMG_INVALIDA"
                           else "corrigir SE a serie sobreviver a decisao acima"),
                  "area_pct_auto": a["area_pct"]})
if rrows:
    with open("stage4/revision_worklist.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rrows[0].keys())); w.writeheader(); w.writerows(rrows)

with open(HIDDEN, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["whst_input_file"])
    for k in recor:
        w.writerow([k])

print(f"correction list: {OUT}  ({len(rows)} images, ordered by series)")
print(f"  baselines (t0) in the list: {sum(1 for r in rows if r['eh_baseline'])}"
      f"  <- a wrong t0 invalidates the whole series")
print(f"  REVIEW list (invalid frame): stage4/revision_worklist.csv ({len(rev)} images)")
print(f"  blinded re-correction: {n_rec} drawn (SEED={SEED}) -> {HIDDEN}")
print(f"  >>> DO NOT OPEN {HIDDEN} before both passes are finished.")
from collections import Counter
print("  by cell line:", dict(Counter(r["cell_line"] for r in rows)))
print("  by series decision:", dict(Counter(r["serie_decisao"] for r in rows)))
print(f"  distinct series: {len(set(r['analysis_unit'] for r in rows))}")
