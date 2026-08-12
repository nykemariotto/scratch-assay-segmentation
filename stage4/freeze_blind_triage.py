# -*- coding: utf-8 -*-
"""
stage4/freeze_blind_triage.py — freezes the BLIND VISUAL TRIAGE as an immutable artefact.

WHY THIS FILE EXISTS
  The blind triage is the only assessment made without knowledge of the automatic
  QC output. That is what makes it the valid reference for one specific claim: the
  sensitivity and specificity of the automatic QC. Any reassessment made after
  seeing the QC results suffers incorporation bias (STARD) and cannot replace it in
  that comparison.

  For EVERYTHING ELSE (the correction list, the WHST failure rate, the exclusion
  rate, the failure mode), the better-informed assessment is more accurate and
  therefore preferable — and the final reference standard of the paired analysis is
  the MANUALLY CORRECTED ROI, not this triage.

PROVENANCE
  Extracted from data/visual_triage.csv.pre_adjudicacao.bak, which preserves the
  state before the 12 AMBIGUO cases were adjudicated. The adjudication was done
  AFTER the operator saw the QC analysis, so it is not blind and does not enter here.

Produces: data/visual_triage_TRIAGEM_CEGA.csv  (read-only; do not edit)
"""
import csv, os, hashlib, stat, sys
from collections import Counter

SRC = "data/visual_triage.csv.pre_adjudicacao.bak"
OUT = "data/visual_triage_TRIAGEM_CEGA.csv"
# only the triage fields themselves plus identification. Nothing derived from analysis.
KEEP = ["whst_input_file", "sorted_basename", "categoria", "subtipo",
        "cell_line", "group_key", "timepoint_h", "campo",
        "raw_file_original", "raw_md5", "test_image"]

if not os.path.isfile(SRC):
    sys.exit(f"could not find {SRC}")
rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
assert len(rows) == 223, f"{len(rows)} linhas, esperava 223"

cnt = Counter(r["categoria"] + ("/" + r["subtipo"] if r["subtipo"] else "") for r in rows)
assert cnt["OK"] == 65 and cnt["SEG_RUIM/super"] == 118 and cnt["SEG_RUIM/sub"] == 14 \
       and cnt["IMG_INVALIDA"] == 14 and cnt["AMBIGUO"] == 12, f"contagens inesperadas: {cnt}"

if os.path.exists(OUT):                      # ja congelado: nao sobrescrever
    os.chmod(OUT, stat.S_IWRITE)

out = [{k: r.get(k, "") for k in KEEP} for r in rows]
out.sort(key=lambda r: r["whst_input_file"])
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=KEEP); w.writeheader(); w.writerows(out)

h = hashlib.md5(open(OUT, "rb").read()).hexdigest()
os.chmod(OUT, stat.S_IREAD)                  # somente leitura

print(f"frozen: {OUT}")
print(f"  {len(out)} images | md5={h}")
print(f"  categories: {dict(cnt)}")
print("  file marked READ-ONLY.")
print("\n  VALID USE   : a blind reference for the sensitivity/specificity of the QC.")
print("  INVALID USE : replacing it with the informed reassessment in that comparison.")
print("  The reference standard of the paired analysis = the manually corrected ROI.")
