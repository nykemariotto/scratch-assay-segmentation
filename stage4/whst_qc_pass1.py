# -*- coding: utf-8 -*-
"""
stage4/whst_qc_pass1.py (v2) — classifies the 223 automatic WHST measurements (pass 1).
It corrects nothing; it only classifies.

v2 corrige dois problemas da v1:
 (1) FALSE NEGATIVE: absolute thresholds plus between-field disagreement do not
     catch failures detectable from the TIME SERIES. Two criteria were added that
     independem de valor absoluto:
       FALHA_CLOSURE_NEG   : area(t>0) > area(baseline t=0) -> the wound "grew"
       FALHA_NAO_MONOTONICO: area sobe > 5 pontos percentuais vs o timepoint
                             anterior da mesma serie
     Quando o baseline tambem esta sinalizado, marca baseline E timepoint.
 (2) FALSE POSITIVE: the merged groups (Controle+Saudavel) mix two experiments;
     the disagreement fired on between-plate variation and the baseline became
     the mean of different plates. For the PAIRED ANALYSIS the unit becomes
     experiment × treatment × well (the split still uses the merged form — that
     does not change the training dataset).

Unidades:
  analysis_unit : HUVEC -> lote_ORIGINAL || tratamento || well
                  SKOV  -> experimento | tratamento | Ffield   (ja e por campo)
  series_key    : analysis_unit (+ campo, no HUVEC) -> serie temporal de 1 campo
"""
import csv, re
from collections import defaultdict, Counter

SUB_HUVEC_TP = {0, 8, 12}
SUB_HUVEC, SUB_SKOV0, SUPER, DISCORD, MONO_PP = 3.0, 10.0, 85.0, 3.0, 5.0

corr = list(csv.DictReader(open("whst_input/correspondencia.csv", encoding="utf-8-sig")))
area = {r["filename"]: float(r["area_pct"]) for r in csv.DictReader(open("data/whst_batch_results.csv", encoding="utf-8-sig"))}
hmap = {r["arquivo_b"]: r for r in csv.DictReader(open("data/mapping_huvec_final.csv", encoding="utf-8"))}


def campo_from_raw(raw):
    """ultimo numero do nome cru HUVEC: 'A5 0HR 1.tiff' -> '1'."""
    stem = re.sub(r"\.(tiff?|png)$", "", raw or "", flags=re.I)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)          # remove sufixo de colisao
    m = re.findall(r"(\d+)", stem)
    return m[-1] if m else "?"


M = []
for r in corr:
    wf = r["whst_input_file"]
    if wf not in area:
        continue
    cell, tp, gk = r["cell_line"], int(r["timepoint_h"]), r["group_key"]
    is_bl = r.get("is_baseline", "no")
    if cell == "HUVEC":
        m = hmap.get(r["test_image"], {})
        lote = (m.get("lote") or "").strip()
        if not lote:                                  # orfao/ambiguo: cai no canonico
            lote = (m.get("lote_canon") or gk.split("||")[0]).strip()
        trat = (m.get("trat_huvec") or "None").strip() or "None"
        well = (m.get("well_campo") or gk.split("||")[-1]).strip()
        campo = (m.get("campo") or "").strip() or campo_from_raw(r["raw_file_original"])
        unit = f"{lote}||{trat}||{well}"
        series = f"{unit}||c{campo}"
    else:                                             # SKOV ja e por campo
        unit = gk
        series = gk
        campo = gk.split("|")[-1]
    M.append({"wf": wf, "cell": cell, "tp": tp, "grp_split": gk, "unit": unit,
              "series": series, "campo": campo, "is_baseline": is_bl,
              "test_image": r["test_image"], "raw": r["raw_file_original"],
              "pct": area[wf], "flags": []})
print(f"measurements: {len(M)}")

# ---------- absolutos ----------
for m in M:
    if m["cell"] == "HUVEC" and m["tp"] in SUB_HUVEC_TP and m["pct"] < SUB_HUVEC:
        m["flags"].append("FALHA_SUB")
    if m["cell"] == "SKOV-3" and m["tp"] == 0 and m["pct"] < SUB_SKOV0:
        m["flags"].append("FALHA_SUB")
    if m["pct"] > SUPER:
        m["flags"].append("FALHA_SUPER")

# ---------- discordancia entre campos, agora DENTRO da unidade desdobrada ----------
ut = defaultdict(list)
for m in M:
    ut[(m["unit"], m["tp"])].append(m)
for k, ms in ut.items():
    p = [x["pct"] for x in ms if x["pct"] > 0]
    if len(ms) >= 2 and p and max(p) / min(p) > DISCORD:
        for x in ms:
            x["flags"].append("DISCORDANCIA_CAMPO")

# ---------- criterios de SERIE ----------
ser = defaultdict(list)
for m in M:
    ser[m["series"]].append(m)

def suspect(x):
    return any(f in x["flags"] for f in ("FALHA_SUB", "FALHA_SUPER", "DISCORDANCIA_CAMPO"))

for s, ms in ser.items():
    ms.sort(key=lambda x: x["tp"])
    base = [x for x in ms if x["tp"] == 0]
    if base:
        bmax = max(x["pct"] for x in base)
        bsusp = any(suspect(x) for x in base)
        for x in ms:
            if x["tp"] > 0 and x["pct"] > bmax:
                x["flags"].append("FALHA_CLOSURE_NEG")
                if bsusp:                              # cannot tell which one failed
                    for b in base:
                        if "FALHA_CLOSURE_NEG" not in b["flags"]:
                            b["flags"].append("FALHA_CLOSURE_NEG")
    # monotonicidade entre timepoints consecutivos (usa o max por timepoint)
    bytp = defaultdict(list)
    for x in ms:
        bytp[x["tp"]].append(x)
    tps = sorted(bytp)
    for i in range(1, len(tps)):
        prev = max(x["pct"] for x in bytp[tps[i - 1]])
        for x in bytp[tps[i]]:
            if x["pct"] - prev > MONO_PP:
                x["flags"].append("FALHA_NAO_MONOTONICO")

for m in M:
    m["flags"] = sorted(set(m["flags"]))
    m["categoria"] = "OK" if not m["flags"] else "+".join(m["flags"])
    m["needs_correction"] = 1 if m["flags"] else 0

with open("data/whst_pass1_qc.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["whst_input_file", "cell_line", "timepoint_h", "analysis_unit", "series_key",
                "campo", "grupo_split", "is_baseline", "area_pct", "categoria",
                "needs_correction", "raw_file", "test_image"])
    for m in sorted(M, key=lambda m: (m["cell"], m["unit"], m["campo"], m["tp"])):
        w.writerow([m["wf"], m["cell"], m["tp"], m["unit"], m["series"], m["campo"],
                    m["grp_split"], m["is_baseline"], round(m["pct"], 3), m["categoria"],
                    m["needs_correction"], m["raw"], m["test_image"]])

# ================= RELATORIO =================
print("\n=== COUNT BY CATEGORY (flags co-occur) ===")
for c in ("FALHA_SUB", "FALHA_SUPER", "DISCORDANCIA_CAMPO", "FALHA_CLOSURE_NEG", "FALHA_NAO_MONOTONICO"):
    print(f"  {c}: {sum(1 for m in M if c in m['flags'])}")
print(f"  OK: {sum(1 for m in M if not m['flags'])}")

need = [m for m in M if m["needs_correction"]]
print(f"\n=== UNIQUE IMAGES FOR MANUAL CORRECTION: {len(need)} of {len(M)} ===")
print(f"  by cell line: {dict(Counter(m['cell'] for m in need))}")
print(f"  by timepoint: {dict(sorted(Counter(m['tp'] for m in need).items()))}")
print(f"  new (from the series criteria alone): "
      f"{sum(1 for m in need if set(m['flags']) <= {'FALHA_CLOSURE_NEG','FALHA_NAO_MONOTONICO'})}")

# unidades de analise
utp = defaultdict(set)
umeds = defaultdict(list)
for m in M:
    utp[m["unit"]].add(m["tp"]); umeds[m["unit"]].append(m)
analyz = [u for u, t in utp.items() if 0 in t and any(x > 0 for x in t)]
print(f"\n=== ANALYSIS UNITS AFTER THE SPLIT-OUT ===")
print(f"  total units: {len(utp)}  |  analysable (0h + >=1 tp): {len(analyz)}")
print(f"  by cell line: {dict(Counter(('SKOV-3' if '|' in u and '||' not in u else 'HUVEC') for u in analyz))}")
ok_u = [u for u in analyz if all(not m["needs_correction"] for m in umeds[u])]
print(f"  intact (no failure): {len(ok_u)}  |  need correction: {len(analyz)-len(ok_u)}")

crit = [u for u in analyz if any(m["needs_correction"] for m in umeds[u] if m["tp"] == 0)]
print(f"\n=== CRITICAL: units with a 0h BASELINE among the failures: {len(crit)} ===")
for u in sorted(crit):
    z = [m for m in umeds[u] if m["tp"] == 0]
    print(f"    {u:<46} 0h={[round(m['pct'],2) for m in z]} cat={[m['categoria'] for m in z]}")

print("\nSalvo: data/whst_pass1_qc.csv")
