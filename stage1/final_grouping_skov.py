# -*- coding: utf-8 -*-
"""
stage1/final_grouping_skov.py — Chave de agrupamento final do SKOV.
Defines a normalised field_id within (experiment, treatment) so that the same
campo fisico receba o mesmo id entre timepoints, apesar do deslocamento de faixa.
Uses stage1/mapping_with_treatment.csv (without touching the external bank).
"""
import csv, re
from collections import defaultdict, Counter

rows = list(csv.DictReader(open("stage1/mapping_with_treatment.csv", encoding="utf-8")))
skov = [r for r in rows if r["linha_celular"] == "SKOV"]


def snap_num(arq):
    m = re.match(r"snap[-_ ]?(\d+)", arq.lower())
    return int(m.group(1)) if m else None


def named_field(arq):
    # ct1_2 -> rep1 field2 -> field_id (rep-1)*5+field
    stem = re.sub(r"\.(tiff?|png|jpe?g)$", "", arq.strip(), flags=re.IGNORECASE)
    m = re.match(r"^(75ug\+ptx|75geo|ptx|ct|carbo_geo|geo_carbo|carbo|geo)(\d+)_(\d+)$",
                 stem.lower())
    if m:
        return (int(m.group(2)) - 1) * 5 + int(m.group(3))
    return None


# 1) collect snaps per (exp, treat, tp) to find the block minimum
block_min = {}
tmp = defaultdict(list)
for r in skov:
    key = (r["experimento"], r["tratamento"], r["timepoint_h"])
    sn = snap_num(r["arquivo_a"])
    if sn is not None:
        tmp[key].append(sn)
for key, lst in tmp.items():
    block_min[key] = min(lst)

# 2) atribui field_id
for r in skov:
    key = (r["experimento"], r["tratamento"], r["timepoint_h"])
    sn = snap_num(r["arquivo_a"])
    if sn is not None:
        r["field_id"] = sn - block_min[key] + 1
    else:
        nf = named_field(r["arquivo_a"])
        r["field_id"] = nf if nf is not None else "?"
    r["grp_field"] = f'{r["experimento"]}|{r["tratamento"]}|F{r["field_id"]}'
    r["grp_treat"] = f'{r["experimento"]}|{r["tratamento"]}'


def viability(ng, label):
    nt = max(1, round(ng * 0.15)); nv = max(1, round(ng * 0.15)); ntr = ng - nt - nv
    verd = "OK" if (ntr >= 3 and nv >= 2 and nt >= 2) else "FRACO"
    print(f"  [{label:<28}] {ng:>4} groups -> train={ntr} val={nv} test={nt}  {verd}")


print("=== SKOV: count by candidate key ===")
viability(len(set(r["grp_treat"] for r in skov)), "experimento x tratamento")
viability(len(set(r["grp_field"] for r in skov)), "exp x tratamento x campo")

# distribution of images per field group
gsz = Counter(r["grp_field"] for r in skov)
szv = sorted(gsz.values())
print(f"\nImages per field group: min={szv[0]} median={szv[len(szv)//2]} max={szv[-1]}")
print(f"Total field groups: {len(gsz)}")

# quantos timepoints por campo (idealmente ate 4 em P1, 3 em P2)
tp_per_field = defaultdict(set)
for r in skov:
    tp_per_field[r["grp_field"]].add(r["timepoint_h"])
print(f"Distribution of #timepoints per field: "
      f"{dict(Counter(len(v) for v in tp_per_field.values()))}")

# consistency check: does each field_id 1..15 exist in each (exp,treat)?
print("\n=== field coverage by (exp, treatment) ===")
cov = defaultdict(lambda: defaultdict(set))
for r in skov:
    cov[(r["experimento"], r["tratamento"])][r["field_id"]].add(r["timepoint_h"])
for k in sorted(cov):
    fields = sorted(cov[k], key=lambda x: (isinstance(x, str), x))
    print(f"  {k[0]}/{k[1]:<10}: {len(fields)} fields, ids {fields[:16]}")

# grava o csv final com as chaves
with open("data/mapping_final_skov.csv", "w", encoding="utf-8", newline="") as f:
    cols = list(skov[0].keys())
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(skov)
print("\nSalvo: data/mapping_final_skov.csv (colunas field_id, grp_field, grp_treat)")
