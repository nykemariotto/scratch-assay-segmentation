# -*- coding: utf-8 -*-
"""
stage1/build_annotation_sheet.py — a spreadsheet for the operator to justify WHY they
put each image in the folder they did, during the visual triage of the overlays
AUTOMATICOS.

OBJETIVO
  1. To audit the triage: does the justification confirm the folder chosen?
  2. Descobrir categorias faltantes: o esquema de 4 caixas (super/sub/invalida/
     ambiguous) was imposed from outside; if several justifications describe the
     same situation that fits none of the boxes, that is a new category.
  3. To validate the 'modo_falha' field independently, since it was computed
     geometrically (containment at t0) and not by inspection.

CEGUEIRA DELIBERADA
  The sheet does NOT show 'modo_falha', 'contencao_t0', nor any QC output
  automatico. Se mostrasse, a justificativa passaria a repetir a metrica em vez
  de testa-la. So aparecem: a pasta escolhida pelo operador, a identificacao da
  the image and the area WHST measured (which was already visible in the overlay).

ESCOPO
  Only the images the operator MOVED into some folder (the ones left in the
  raiz = OK nao precisam de justificativa). Ordenadas por serie, para a
  justificativa poder usar o contexto temporal.

Produces: stage1/annotation_sheet.csv
"""
import csv, os, sys
from collections import Counter

TRIAGEM = "data/inspecao_visual_TRIAGEM_CEGA.csv"   # estado das PASTAS (congelado)
ATUAL = "data/inspecao_visual.csv"                  # pos-adjudicacao (p/ contexto)
AUTO = "data/whst_pass1_qc.csv"
MAP = "whst_output/overlays_sorted_map.csv"
OUT = "stage1/annotation_sheet.csv"

if not os.path.isfile(TRIAGEM):
    sys.exit(f"could not find {TRIAGEM} (run stage4/freeze_blind_triage.py)")

tri = list(csv.DictReader(open(TRIAGEM, encoding="utf-8-sig")))
atual = {r["whst_input_file"]: r for r in csv.DictReader(open(ATUAL, encoding="utf-8-sig"))}
auto = {r["whst_input_file"]: r for r in csv.DictReader(open(AUTO, encoding="utf-8-sig"))}
mp = {r["whst_input_file"]: r for r in csv.DictReader(open(MAP, encoding="utf-8-sig"))}

PASTA = {("OK", ""): "",                       # ficou na raiz -> sem justificativa
         ("SEG_RUIM", "super"): "_SEG_RUIM/_super",
         ("SEG_RUIM", "sub"): "_SEG_RUIM/_sub",
         ("IMG_INVALIDA", ""): "_IMG_INVALIDA",
         ("AMBIGUO", ""): "_AMBIGUO"}

rows = []
for r in tri:
    pasta = PASTA.get((r["categoria"], r["subtipo"]), "?")
    if not pasta:                              # OK: it was not moved
        continue
    k = r["whst_input_file"]
    a = auto[k]
    # if it was AMBIGUO and adjudicated later, record it for the operator's context
    adj = ""
    if r["categoria"] == "AMBIGUO":
        cur = atual.get(k, {})
        adj = cur.get("categoria", "") + ("/" + cur["subtipo"] if cur.get("subtipo") else "")
    rows.append({
        "pasta_triagem": pasta,
        "adjudicado_depois_como": adj,
        "cell_line": r["cell_line"],
        "analysis_unit": a["analysis_unit"],
        "timepoint_h": int(r["timepoint_h"]),
        "campo": r["campo"],
        "area_pct_whst": a["area_pct"],
        "arquivo_overlay": mp[k]["sorted_basename"] if k in mp else "",
        "whst_input_file": k,
        # ---- a preencher ----
        "motivo": "",
        "pasta_correta?": "",
        "categoria_sugerida": "",
        "confianca": "",
    })

# Sort BY FOLDER first (the operator works one folder at a time, without
# entrando e saindo delas), e dentro da pasta por serie (contexto temporal).
# Pastas menores antes: definem os extremos do criterio e dao ritmo.
ORDEM_PASTA = {"_IMG_INVALIDA": 0, "_SEG_RUIM/_sub": 1, "_AMBIGUO": 2,
               "_SEG_RUIM/_super": 3}
rows.sort(key=lambda r: (ORDEM_PASTA.get(r["pasta_triagem"], 9), r["cell_line"],
                         r["analysis_unit"], r["timepoint_h"], str(r["campo"])))
for i, r in enumerate(rows, 1):
    r["ordem"] = i
cols = ["ordem"] + [c for c in rows[0] if c != "ordem"]
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

print(f"produced: {OUT}  ({len(rows)} images you moved into some folder)")
print("  distribuicao:", dict(Counter(r["pasta_triagem"] for r in rows)))
print(f"  (the {len(tri)-len(rows)} left in the root = OK need no justification)")
print("""
COMO PREENCHER
  motivo            : why you put it in that folder. Free text, can be short
                      ('the contour caught the well border', 'wound already closed but
                      caught debris', 'contour only on the top half of the wound').
  pasta_correta?    : sim | nao | duvida   (looking again, was the folder right?)
  categoria_sugerida: if 'nao' or 'duvida', which category would describe it better.
                      You may invent new names — that is exactly what we want
                      descobrir. Ex.: 'pegou_fundo_do_poco', 'dividiu_a_ferida'.
  confianca         : alta | media | baixa  (opcional)

  Use o painel/arquivo em whst_output/overlays_sorted/ (coluna arquivo_overlay).
  Blank rows are allowed: note down whatever is informative.

  Depois:  python stage1/read_annotations.py
""")
