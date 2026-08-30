# -*- coding: utf-8 -*-
"""
benchmark_classical.py — o braço clássico do A4 já está medido.

A revisão pede comparação contra ≥1 ferramenta automática clássica, no mesmo test
set. That would normally mean installing and running a new tool. It does not:
a Etapa 4 já rodou o Wound Healing Size Tool em modo AUTOMÁTICO sobre as mesmas
images that were later corrected by hand to become the reference standard.

  data/whst_batch_results.csv  -> saída AUTOMÁTICA do WHST (a ferramenta clássica)
  data/whst_areas_final.csv    -> a mesma imagem após correção supervisionada (o padrão)
  data/visual_triage.csv     -> blind visual triage of the automatic outputs

And 213 of the 223 images fall in the test set (40 of 41 groups), so the comparison
está na partição certa.

Este script quantifica o braço clássico em dois níveis:
  (1) per image  — how much the automatic area differs from the corrected one;
  (2) per series — how much the closure fraction derived from the automatic
                   measurement differs from the one derived from the reference. It
                   is the endpoint the paper reports, and it is
                    onde a comparação com o modelo vai acontecer.

It does not compare against the AI: it produces the classical side so that, when
the grid finishes, the comparison is a single merge.
"""
import csv
import os
import statistics as st
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def ler(p):
    if not os.path.isfile(p):
        sys.exit(f"could not find {p}")
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


AUTO = ler("data/whst_batch_results.csv")
FINAL = ler("data/whst_areas_final.csv")
INSP = ler("data/visual_triage.csv")
MAP = ler("data/mapping_dataset_final_strat.csv")

# partition by group
gk_part = {}
for r in MAP:
    if r["partition"] in ("train", "val", "test"):
        gk_part[r["group_key"]] = r["partition"]

# CHAVE DE JUNCAO: o prefixo md5 de 10 caracteres, presente nos tres arquivos com
# different name formats. Matching by name substring would fail (which is what
# happened in the first version: 198 pairs, all without a partition).
#   whst_batch_results.filename   004fc0a19e__HUVEC__…tiff
#   whst_areas_final.whst_input_file  121a1c1fcd__HUVEC__…tiff
#   visual_triage.sorted_basename   HUVEC__…__121a1c1fcd.jpg
def md5pref(nome):
    base = os.path.basename(str(nome))
    for pedaco in os.path.splitext(base)[0].split("__"):
        if len(pedaco) == 10 and all(c in "0123456789abcdef" for c in pedaco.lower()):
            return pedaco.lower()
    return None


part_img, cat_img, test_img = {}, {}, {}
for r in INSP:
    k = md5pref(r.get("whst_input_file") or r["sorted_basename"])
    if not k:
        continue
    part_img[k] = gk_part.get(r["group_key"], "?")
    cat_img[k] = r["categoria"]
    test_img[k] = r.get("test_image", "")

auto_por = {}
for r in AUTO:
    k = md5pref(r["filename"])
    if k:
        auto_por[k] = r

print("=" * 74)
print("BRAÇO CLÁSSICO DO A4 — Wound Healing Size Tool em modo automático")
print("=" * 74)
print(f"\nsaídas automáticas : {len(AUTO)}")
print(f"áreas corrigidas   : {len(FINAL)}")

# ---------------------------------------------------------------- por imagem
pares = []
for r in FINAL:
    k = md5pref(r["whst_input_file"])
    a = auto_por.get(k) if k else None
    if not a:
        continue
    va, vf = num(a.get("area_pct")), num(r.get("area_pct_final"))
    if va is None or vf is None:
        continue
    pares.append({"k": k, "auto": va, "final": vf, "dif": va - vf,
                  "part": part_img.get(k, "?"), "cat": cat_img.get(k, "?")})

print(f"pares automático×corrigido formados : {len(pares)}")
if not pares:
    sys.exit("no pairs formed, check the name matching")

from collections import Counter
print("\npor partição :", dict(Counter(p["part"] for p in pares)))
print("by triage    :", dict(Counter(p["cat"] for p in pares)))


def resumo(rot, sub):
    if len(sub) < 2:
        print(f"  {rot:<26} n={len(sub)}  (poucos dados)")
        return
    d = [p["dif"] for p in sub]
    ad = [abs(v) for v in d]
    inal = sum(1 for p in sub if abs(p["dif"]) < 1e-9)
    print(f"  {rot:<26} n={len(sub):>3}  viés {st.mean(d):>+7.2f} pp  "
          f"|erro| mediano {st.median(ad):>6.2f} pp  máx {max(ad):>6.2f} pp  "
          f"inalteradas {100*inal/len(sub):.0f}%")


print("\n--- automatic − corrected difference (percentage points of area) ---")
resumo("todas", pares)
for c in ("OK", "SEG_RUIM", "IMG_INVALIDA"):
    resumo(f"triage {c}", [p for p in pares if p["cat"] == c])
resumo("só test set", [p for p in pares if p["part"] == "test"])

# ---------------------------------------------------------------- por serie
LONGO = ler("data/closure_final_long.csv")
ref_cl = {(r["series_key"], r["campo"], r["timepoint_h"]): num(r["closure_fraction"])
          for r in LONGO}

# recompute closure from the AUTOMATIC areas, same formula
areas_auto = {}
for r in FINAL:
    k = md5pref(r["whst_input_file"])
    a = auto_por.get(k) if k else None
    if not a:
        continue
    v = num(a.get("area_pct"))
    if v is None:
        continue
    areas_auto.setdefault((r["series_key"], r["campo"]), {})[r["timepoint_h"]] = v

comp = []
for (sk, campo), tps in areas_auto.items():
    if "0" not in tps:
        continue
    a0 = tps["0"]
    if a0 <= 0:
        continue
    for tp, at in tps.items():
        if tp == "0":
            continue
        cl_auto = (a0 - at) / a0
        cl_ref = ref_cl.get((sk, campo, tp))
        if cl_ref is None:
            continue
        comp.append({"sk": sk, "campo": campo, "tp": tp,
                     "auto": cl_auto, "ref": cl_ref, "dif": cl_auto - cl_ref})

print("\n--- closure fraction: derivada do automático × do reference standard ---")
print(f"  observações pareadas : {len(comp)}")
if comp:
    d = [c["dif"] for c in comp]
    ad = [abs(v) for v in d]
    # THE MEAN IS NO USE HERE. When the automatic WHST gets the baseline wrong, the
    # closure explodes (the smaller a0 goes into the denominator) and a handful of cases
    # domina qualquer media. Reportar mediana, e contar as catastrofes a parte —
    # they are the finding, not noise to be smoothed away.
    catas = [c for c in comp if abs(c["dif"]) > 1.0]
    print(f"  |erro| mediano ..... {st.median(ad):.3f}   (a média é inútil aqui, ver abaixo)")
    print(f"  fora de escala ..... {len(catas)} de {len(comp)} com |Δ| > 1,0 "
          f"— baseline automática errada faz a razão explodir")
    print(f"  |erro| > 0,10 ...... {sum(1 for v in ad if v > .10)} de {len(ad)} "
          f"({100*sum(1 for v in ad if v > .10)/len(ad):.0f}%)")
    print(f"  |erro| > 0,25 ...... {sum(1 for v in ad if v > .25)} de {len(ad)} "
          f"({100*sum(1 for v in ad if v > .25)/len(ad):.0f}%)")
    sinal = sum(1 for c in comp if c["auto"] * c["ref"] < 0)
    print(f"  discordância de sinal {sinal} de {len(comp)} "
          f"({100*sinal/len(comp):.0f}%)")
    piores = sorted(comp, key=lambda c: -abs(c["dif"]))[:5]
    print("\n  piores casos:")
    for c in piores:
        print(f"    {c['sk'][:34]:<34} campo {c['campo']:<4} tp {c['tp']:>3}h  "
              f"auto {c['auto']:>+7.3f}  ref {c['ref']:>+7.3f}  Δ {c['dif']:>+7.3f}")

# ---------------------------------------------------------------- saida
with open("stage3/benchmark_classical_long.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["series_key", "campo", "timepoint_h",
                                      "closure_whst_auto", "closure_reference", "diferenca"])
    w.writeheader()
    for c in comp:
        w.writerow({"series_key": c["sk"], "campo": c["campo"], "timepoint_h": c["tp"],
                    "closure_whst_auto": round(c["auto"], 6),
                    "closure_reference": round(c["ref"], 6),
                    "diferenca": round(c["dif"], 6)})
print(f"\nescrito: stage3/benchmark_classical_long.csv ({len(comp)} linhas)")

print("\n" + "=" * 74)
print("INCORPORATION BIAS — declare this, do not hide it")
print("=" * 74)
inalt_ok = sum(1 for p in pares if p["cat"] == "OK" and abs(p["dif"]) < 1e-9)
n_ok = sum(1 for p in pares if p["cat"] == "OK")
print(f"""
The reference standard was built by CORRECTING the automatic output, not by
measuring from scratch. Where the automatic result was judged acceptable, the
standard is identical to it by construction — {inalt_ok} of the {n_ok} images
triaged OK have a difference of exactly zero.

Consequence: this comparison is not independent and **understates** the error of
the classical tool. It is not a defect to be fixed with statistics; it is a
property of the design, and it has to be declared: it is the same
incorporation-bias objection that STARD raises.

What saves the argument is the direction of the bias: even with the ruler tilted in
favour of the classical tool, it fails on {sum(1 for p in pares if p['cat']=='SEG_RUIM')} of {len(pares)} images
({100*sum(1 for p in pares if p['cat']=='SEG_RUIM')/len(pares):.0f}%). The real number is worse than this. Report it as a
**lower bound** on the failure rate, and the sentence gets stronger, not weaker.

The AI arm does not have this problem: the models never saw the reference standard,
nem no treino nem na construção dele.""")
print("\nWhen the grid finishes, the AI column enters this same file and")
print("the head-to-head against the classical arm comes from a merge, not a new measurement campaign.")
