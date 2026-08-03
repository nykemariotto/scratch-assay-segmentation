# -*- coding: utf-8 -*-
"""
stage1/apply_annotation_reclass.py — promotes the operator's ANNOTATION to the classification
primaria de data/inspecao_visual.csv.

Rationale: the original triage used 4 boxes, with no room for a justification; the
annotation was made with the overlay in view and free text. Where the two diverge,
the annotation is the better-informed reading (see PROTOCOL section 8: for the
substantive analysis the more accurate assessment counts; the blind triage stays
frozen and intact for the comparison against the automatic QC).

It APPLIES only UNAMBIGUOUS mappings:
    segmentacao_ok       -> OK
    imagem_invalida      -> IMG_INVALIDA
    excesso              -> SEG_RUIM/super
    espuria_fechada      -> SEG_RUIM/super   (a mask where there is no wound)
    deslocada            -> SEG_RUIM/super   (same; it is not 'sub')
    sub                  -> SEG_RUIM/sub
    excesso_e_sub        -> SEG_RUIM/super   (predomina o excesso; marcado)
It does NOT apply (ambiguous -> keeps the folder and reports):
    ruidosa_recuperavel  (poor but segmentable image: it says nothing about the
                          quality of the segmentation, so the category cannot be
                          inferred)
    outro / vazio

It writes the column 'origem_categoria' = triagem | anotacao, and makes a backup.
"""
import csv, shutil
from collections import Counter

HUM = "data/inspecao_visual.csv"
REP = "data/annotation_report.csv"

MAP = {"segmentacao_ok": ("OK", ""),
       "imagem_invalida": ("IMG_INVALIDA", ""),
       "excesso": ("SEG_RUIM", "super"),
       "espuria_fechada": ("SEG_RUIM", "super"),
       "deslocada": ("SEG_RUIM", "super"),
       "sub": ("SEG_RUIM", "sub"),
       "excesso_e_sub": ("SEG_RUIM", "super")}
AMBIGUO = {"ruidosa_recuperavel", "outro", ""}

hum = list(csv.DictReader(open(HUM, encoding="utf-8-sig")))
rep = {r["whst_input_file"]: r for r in csv.DictReader(open(REP, encoding="utf-8-sig"))}

mud, mantidas, nao_anot = [], [], 0
for r in hum:
    k = r["whst_input_file"]
    a = rep.get(k)
    r.setdefault("origem_categoria", "triagem")
    r.setdefault("motivo_operador", "")
    if not a:
        nao_anot += 1
        continue
    mc = a.get("motivo_classificado", "")
    r["motivo_operador"] = a.get("motivo", "")
    if mc in AMBIGUO:
        if mc == "ruidosa_recuperavel":
            mantidas.append((k, mc, r["categoria"]))
        continue
    novo = MAP.get(mc)
    if not novo:
        continue
    antes = r["categoria"] + ("/" + r["subtipo"] if r["subtipo"] else "")
    if (r["categoria"], r["subtipo"]) != novo:
        mud.append((antes, novo[0] + ("/" + novo[1] if novo[1] else ""), k))
        r["categoria"], r["subtipo"] = novo
        r["origem_categoria"] = "anotacao"

shutil.copy2(HUM, HUM + ".bak_prereclass")
with open(HUM, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hum[0].keys()))
    w.writeheader(); w.writerows(hum)

print(f"rows: {len(hum)}  | without annotation: {nao_anot}")
print(f"RECLASSIFICADAS: {len(mud)}")
for (a, b), n in Counter((x[0], x[1]) for x in mud).most_common():
    print(f"  {a:<18} -> {b:<18} {n:>3}")
if mantidas:
    print(f"\nNOT reclassified (ambiguous annotation) — folder kept: {len(mantidas)}")
    for k, mc, cat in mantidas:
        print(f"  [{mc}] mantida como {cat}")
print(f"\ndistribuicao final: {dict(Counter(r['categoria'] + ('/' + r['subtipo'] if r['subtipo'] else '') for r in hum))}")
print(f"backup: {HUM}.bak_prereclass")
