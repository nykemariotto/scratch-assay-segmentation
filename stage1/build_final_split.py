# -*- coding: utf-8 -*-
"""
stage1/build_final_split.py — M2. Split deterministico por split_key (ja com fallback),
estratificado por linha celular, SEED=42, 70/15/15 por GRUPO.
Produz stage1/mapping_dataset_final.csv e reporta a cobertura resultante.
"""
import csv
import numpy as np
from collections import Counter, defaultdict

SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15
SRC = "stage1/mapping_with_splitkey.csv"
OUT = "stage1/mapping_dataset_final.csv"

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
act = [r for r in rows if not r["excluida"]]

# ---------------- split por grupo, estratificado por linha celular ----------
assign = {}
for cl in sorted({r["linha_celular"] for r in act}):
    keys = sorted({r["split_key"] for r in act if r["linha_celular"] == cl})
    rng = np.random.RandomState(SEED)
    keys = list(keys)
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * TEST_FRAC)))
    n_val = max(1, int(round(len(keys) * VAL_FRAC)))
    for i, k in enumerate(keys):
        assign[k] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    print(f"{cl}: {len(keys)} grupos -> train={len(keys)-n_test-n_val} "
          f"val={n_val} test={n_test}")

for r in rows:
    r["partition"] = "EXCLUIDA" if r["excluida"] else assign[r["split_key"]]

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---------------- relatorio ----------------
print("\n=== Images by partition × cell line ===")
t = defaultdict(Counter)
for r in act:
    t[r["partition"]][r["linha_celular"]] += 1
tot = Counter(r["partition"] for r in act)
for p in ("train", "val", "test"):
    print(f"  {p:<6} HUVEC={t[p]['HUVEC']:>4}  SKOV={t[p]['SKOV']:>4}  "
          f"total={tot[p]:>4} ({100*tot[p]/len(act):.1f}%)")
print(f"  TOTAL  {len(act)} active images (+3 excluded)")

print("\n=== Groups by partition × cell line ===")
g = defaultdict(lambda: defaultdict(set))
for r in act:
    g[r["partition"]][r["linha_celular"]].add(r["split_key"])
for p in ("train", "val", "test"):
    print(f"  {p:<6} HUVEC={len(g[p]['HUVEC']):>4}  SKOV={len(g[p]['SKOV']):>4}")

# ---------------- diagnostico de cobertura ----------------
print("\n=== COVERAGE: timepoint × partition ===")
for cl in ("HUVEC", "SKOV"):
    tp = defaultdict(Counter)
    for r in act:
        if r["linha_celular"] == cl:
            tp[r["partition"]][r["timepoint_h"]] += 1
    alltp = sorted({r["timepoint_h"] for r in act if r["linha_celular"] == cl},
                   key=lambda x: int(x))
    print(f"  {cl}:")
    for p in ("train", "val", "test"):
        miss = [x for x in alltp if tp[p][x] == 0]
        print(f"    {p:<6} {{{', '.join(f'{x}h:{tp[p][x]}' for x in alltp)}}}"
              f"{'   AUSENTE: ' + str(miss) if miss else ''}")

print("\n=== COVERAGE: stratum (treatment) × partition ===")
faltas = []
for cl in ("HUVEC", "SKOV"):
    estr = sorted({r["estrato"] for r in act if r["linha_celular"] == cl})
    print(f"  {cl}: {len(estr)} estratos")
    for e in estr:
        c = Counter(r["partition"] for r in act if r["estrato"] == e)
        aus = [p for p in ("train", "val", "test") if c[p] == 0]
        flag = f"   <-- AUSENTE de {aus}" if aus else ""
        if aus:
            faltas.append((e, aus))
        print(f"    {e:<28} train={c['train']:>4} val={c['val']:>4} test={c['test']:>4}{flag}")

print(f"\n>>> strata absent from some partition: {len(faltas)}")
print(f"\nSalvo: {OUT}")
