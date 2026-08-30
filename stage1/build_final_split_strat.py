# -*- coding: utf-8 -*-
"""
stage1/build_final_split_strat.py — ALTERNATIVA ao M2.
Identical to M2 (same split_key with fallback, SEED=42, 70/15/15 by group), but it
stratifies by STRATUM (treatment) as well as cell line, so that no
tratamento fique ausente de val/test.
It does not overwrite stage1/mapping_dataset_final.csv — it writes to a separate file.

⚠️ RUN ONCE, NOT REGENERABLE — IT DRAWS. This is the script that builds the
leakage-free partition the whole revision rests on.
"""
import csv
import numpy as np
from collections import Counter, defaultdict

SEED = 42
SRC = "stage1/mapping_with_splitkey.csv"
OUT = "data/mapping_dataset_final_strat.csv"

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
act = [r for r in rows if not r["excluida"]]

# stratum of each split_key (SHARED super-keys have no treatment -> 'HUVEC|None')
key_estrato = {}
key_size = Counter()
for r in act:
    key_estrato.setdefault(r["split_key"], r["estrato"])
    key_size[r["split_key"]] += 1

by_estrato = defaultdict(list)
for k, e in key_estrato.items():
    by_estrato[e].append(k)

assign = {}
for e in sorted(by_estrato):
    keys = sorted(by_estrato[e], key=lambda k: (-key_size[k], k))
    rng = np.random.RandomState(SEED)
    keys = list(keys)
    rng.shuffle(keys)
    n = len(keys)
    if n >= 3:
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        n_test = min(n_test, n - 2)
        n_val = min(n_val, n - 1 - n_test)
    elif n == 2:
        n_test, n_val = 1, 0
    else:
        n_test, n_val = 0, 0
    for i, k in enumerate(keys):
        assign[k] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

for r in rows:
    r["partition"] = "EXCLUIDA" if r["excluida"] else assign[r["split_key"]]

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

tot = Counter(r["partition"] for r in act)
print("=== Split stratified by treatment ===")
for p in ("train", "val", "test"):
    print(f"  {p:<6} {tot[p]:>4} images ({100*tot[p]/len(act):.1f}%)  "
          f"grupos={len({r['split_key'] for r in act if r['partition']==p})}")

print("\n=== Coverage by stratum ===")
faltas = 0
for cl in ("HUVEC", "SKOV"):
    for e in sorted({r["estrato"] for r in act if r["linha_celular"] == cl}):
        c = Counter(r["partition"] for r in act if r["estrato"] == e)
        aus = [p for p in ("train", "val", "test") if c[p] == 0]
        faltas += bool(aus)
        flag = f"   <-- AUSENTE de {aus}" if aus else ""
        print(f"  {e:<28} train={c['train']:>4} val={c['val']:>4} test={c['test']:>4}{flag}")
print(f"\n>>> strata absent from some partition: {faltas}")

# zero-overlap
g = defaultdict(set)
for r in act:
    g[r["group_key"]].add(r["partition"])
print(f">>> group keys in >1 partition: {sum(1 for v in g.values() if len(v)>1)}")
k = defaultdict(set)
for r in act:
    k[r["split_key"]].add(r["partition"])
print(f">>> split keys in >1 partition: {sum(1 for v in k.values() if len(v)>1)}")

print("\n=== Timepoints ===")
for cl in ("HUVEC", "SKOV"):
    tp = defaultdict(Counter)
    for r in act:
        if r["linha_celular"] == cl:
            tp[r["partition"]][r["timepoint_h"]] += 1
    alltp = sorted({r["timepoint_h"] for r in act if r["linha_celular"] == cl}, key=int)
    print(f"  {cl}: " + " | ".join(
        f"{p}: {{{', '.join(f'{x}h:{tp[p][x]}' for x in alltp)}}}" for p in ("train", "val", "test")))
print(f"\nSalvo: {OUT}")
