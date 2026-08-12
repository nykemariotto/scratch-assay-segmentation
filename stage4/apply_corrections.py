# -*- coding: utf-8 -*-
"""
stage4/apply_corrections.py — folds in the outcomes of pass 1 (manual correction).

TWO THINGS:
 (1) CLASSIFICATION: status 'invalida' -> IMG_INVALIDA in data/visual_triage.csv.
 (2) FINAL AREA: writes data/whst_areas_final.csv with the area to be used in the
     paired analysis, per image, with its source declared:

     status 'ok'       -> area_pct_corrigida        fonte=corrigida
     status 'fechada'  -> 0.0  (a valid measurement) fonte=corrigida_fechada
     status 'invalida' -> absent                    fonte=invalida
     status 'pulada'   -> automatic area + flag     fonte=automatica_PENDENTE
     not corrected     -> automatic area            fonte=automatica

MIXING RULE (why it is valid to mix corrected and automatic within one series):
corrected frames are the ones the triage judged badly segmented; uncorrected
frames are the ones judged correct, so their automatic measurement is already the
good one. In the USAVEL_sem_correcao series no frame was corrected and every area
stays automatic — there the closure is valid because the multiplicative bias
cancels in the ratio, even though the absolute areas are inflated. This is recorded
per series so the final analysis can tell the two situations apart.
"""
import csv, os, shutil, sys
from collections import Counter, defaultdict

HUM = "data/visual_triage.csv"
AUTO = "data/whst_pass1_qc.csv"
P1 = "stage4/manual_correction_pass1.csv"
OUT = "data/whst_areas_final.csv"

if not os.path.isfile(P1):
    sys.exit(f"could not find {P1} (run pass 1 in Fiji)")

# The macro's CSV is APPEND-only: reprocessing a 'pulada' image creates a second
# row for the same file. It is resolved by explicit priority, not by read order:
# a resolved outcome always beats 'pulada'.
PRIO = {"ok": 3, "fechada": 3, "invalida": 2, "pulada": 1}


def melhor(cur, novo):
    return novo if (cur is None or
                    PRIO.get(novo["status"], 0) >= PRIO.get(cur["status"], 0)) else cur


def carrega(path):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        k = r["whst_input_file"]
        d[k] = melhor(d.get(k), r)
    return d


p1 = carrega(P1)
print(f"pass 1: {len(p1)} images | {dict(Counter(r['status'] for r in p1.values()))}")

# Passes 3 (provenance validation) and 4 (completion of the mixed series) also
# produce valid corrections and enter the final area. Pass 2 does NOT: it is the
# blinded re-correction of the same subset, it exists to measure reproducibility,
# and using it would replace the primary measurement with the repeat.
for extra, rot in (("stage4/manual_correction_validation.csv", "pass 3 (validation)"),
                   ("stage4/manual_correction_completion.csv", "pass 4 (completion)"),
                   ("stage4/manual_correction_baselines.csv", "pass 5 (recovered baselines)")):
    if os.path.isfile(extra):
        e = carrega(extra)
        novos = {k: v for k, v in e.items() if k not in p1}
        p1.update(novos)
        print(f"{rot}: {len(e)} records ({len(novos)} new) | "
              f"{dict(Counter(r['status'] for r in e.values()))}")

hum = list(csv.DictReader(open(HUM, encoding="utf-8-sig")))
auto = {r["whst_input_file"]: r for r in csv.DictReader(open(AUTO, encoding="utf-8-sig"))}

# ---------- (1) classification ----------
mud, prom = [], []
for r in hum:                                    # coluna em TODAS as linhas
    r["status_correcao"] = ""
for r in hum:
    k = r["whst_input_file"]
    c = p1.get(k)
    if not c:
        continue
    r["status_correcao"] = c["status"]
    if c["status"] == "invalida" and r["categoria"] != "IMG_INVALIDA":
        mud.append((r["categoria"] + ("/" + r["subtipo"] if r["subtipo"] else ""), k))
        r["categoria"], r["subtipo"] = "IMG_INVALIDA", ""
        r["origem_categoria"] = "correcao_manual"
    # PROMOTION: a frame labelled IMG_INVALIDA that the operator MANAGED to segment.
    # Tracing the contour is the proof that the image is usable; the correction
    # overrides the earlier label (which came from the triage).
    # Without this, stage4/whst_series_analysis.py would drop the frame from the
    # series while stage4/final_closure_table.py would include it — they would
    # disagree.
    elif c["status"] in ("ok", "fechada") and r["categoria"] in ("IMG_INVALIDA", "NAO_TRIADA"):
        # NAO_TRIADA = baselines recovered afterwards, which never went through the
        # visual triage; a successful correction is what establishes their validity.
        antes = r["categoria"]                    # ler ANTES de sobrescrever
        prom.append((k, f"{antes}->{c['status']}"))
        r["categoria"], r["subtipo"] = "OK", ""
        r["origem_categoria"] = ("baseline_recuperado_corrigido"
                                 if antes == "NAO_TRIADA" else "correcao_manual")
# ATOMIC write: build a temporary file and only then replace. Avoids truncating the
# source file if something fails midway (and does not clobber an existing backup).
assert len(hum) > 100, f"guard: data/visual_triage.csv with only {len(hum)} rows"
if not os.path.isfile(HUM + ".bak_precorr"):
    shutil.copy2(HUM, HUM + ".bak_precorr")
_tmp = HUM + ".tmp"
with open(_tmp, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hum[0].keys()))
    w.writeheader(); w.writerows(hum)
os.replace(_tmp, HUM)
print(f"\n(1) reclassified as IMG_INVALIDA by the correction: {len(mud)}")
for a, k in mud:
    print(f"    {a:<18} -> IMG_INVALIDA   {k[:56]}")
print(f"    PROMOTED to OK (invalid but successfully segmented): {len(prom)}")
for k, stt in prom:
    print(f"    IMG_INVALIDA -> OK  (status={stt})  {k[:52]}")

# ---------- (2) final area ----------
rows, pend = [], []
for r in hum:
    k = r["whst_input_file"]
    a = auto[k]
    c = p1.get(k)
    # a frame judged INVALID (and not promoted by the correction) has no final area:
    # an invalid image yields no measurement. Without this,
    # stage4/whst_series_analysis.py (which drops IMG_INVALIDA) and
    # stage4/final_closure_table.py (which only looks at an empty area) would
    # disagree about which frames belong to the series.
    if r["categoria"] == "IMG_INVALIDA":
        area, fonte = None, ("invalida" if (c and c["status"] == "invalida")
                             else "invalida_triagem")
    elif r["categoria"] == "NAO_TRIADA":
        # a baseline recovered afterwards, still without manual correction. It must
        # NOT enter with the automatic area: that area is precisely the suspect one
        # (up to 4x the batch median) and it would become the series' a0, propagating
        # the error to every timepoint. It stays without an area until pass 5 decides.
        area, fonte = None, "pendente_correcao"
    elif c is None:
        area, fonte = float(a["area_pct"]), "automatica"
    elif c["status"] == "ok":
        area, fonte = float(c["area_pct_corrigida"]), "corrigida"
    elif c["status"] == "fechada":
        area, fonte = 0.0, "corrigida_fechada"
    elif c["status"] == "invalida":
        area, fonte = None, "invalida"
    else:                                        # pulada
        area, fonte = float(a["area_pct"]), "automatica_PENDENTE"
        pend.append(k)
    rows.append({"whst_input_file": k, "cell_line": r["cell_line"],
                 "analysis_unit": a["analysis_unit"], "series_key": a["series_key"],
                 "campo": a["campo"], "timepoint_h": int(r["timepoint_h"]),
                 "area_pct_final": "" if area is None else round(area, 3),
                 "area_pct_auto": a["area_pct"], "fonte_area": fonte,
                 "categoria": r["categoria"] + ("/" + r["subtipo"] if r["subtipo"] else "")})
rows.sort(key=lambda x: (x["cell_line"], x["analysis_unit"], x["timepoint_h"], str(x["campo"])))
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

cf = Counter(x["fonte_area"] for x in rows)
print(f"\n(2) {OUT}: {len(rows)} images")
for k, v in cf.most_common():
    print(f"    {k:<22} {v:>4}")
if pend:
    print(f"\n    [PENDING] {len(pend)} 'pulada' image(s) using the automatic area:")
    for k in pend:
        print(f"      {k[:64]}")

# ---------- 100% automatic series (bias cancels) vs mixed ----------
bys = defaultdict(list)
for x in rows:
    bys[x["series_key"]].append(x["fonte_area"])
puro_auto = [s for s, v in bys.items() if all(f.startswith("automatica") for f in v)]
misto = [s for s, v in bys.items()
         if any(f.startswith("corrigida") for f in v) and any(f == "automatica" for f in v)]
puro_corr = [s for s, v in bys.items()
             if all(f.startswith("corrigida") or f == "invalida" for f in v)]
print(f"\n(3) composition of the {len(bys)} series:")
print(f"    100% automatic area (bias cancels in the ratio) : {len(puro_auto)}")
print(f"    mixed (corrected where it was wrong)            : {len(misto)}")
print(f"    100% corrected                                  : {len(puro_corr)}")
