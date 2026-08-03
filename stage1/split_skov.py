# -*- coding: utf-8 -*-
"""
stage1/split_skov.py — Materializa o split do SKOV sobre os 118 grp_field,
estratificando por grp_treat (8 coortes experimento x tratamento).
Target 70/15/15 by GROUP, keeping every field whole in a single partition.

⚠️ RUN ONCE, NOT REGENERABLE — IT DRAWS.
"""
import csv, json, random
from collections import defaultdict, Counter

SRC = "data/mapping_final_skov.csv"
OUT = "data/mapping_final_skov.csv"   # reescreve com a coluna split_skov

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))

# grupo -> (estrato, n_imagens)
grp_rows = defaultdict(list)
for r in rows:
    grp_rows[r["grp_field"]].append(r)
grp_stratum = {g: rs[0]["grp_treat"] for g, rs in grp_rows.items()}


def build(seed):
    rnd = random.Random(seed)
    assign = {}
    by_stratum = defaultdict(list)
    for g in grp_rows:
        by_stratum[grp_stratum[g]].append(g)
    for st, gs in by_stratum.items():
        gs = sorted(gs, key=lambda g: (-len(grp_rows[g]), g))
        rnd.shuffle(gs)
        n = len(gs)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        # guarantees at least 1 group in each partition per stratum
        n_val = min(n_val, max(1, n - 2))
        n_test = min(n_test, max(1, n - 1 - n_val))
        for i, g in enumerate(gs):
            if i < n_test:
                assign[g] = "test"
            elif i < n_test + n_val:
                assign[g] = "val"
            else:
                assign[g] = "train"
    return assign


def score(assign):
    """Lower is better: image-proportion error + coverage penalty."""
    img = Counter()
    tp_cov = defaultdict(set)
    tr_cov = defaultdict(set)
    for g, sp in assign.items():
        for r in grp_rows[g]:
            img[sp] += 1
            tp_cov[sp].add(r["timepoint_h"])
            tr_cov[sp].add(r["grp_treat"])
    tot = sum(img.values())
    err = (abs(img["train"] / tot - .70) + abs(img["val"] / tot - .15)
           + abs(img["test"] / tot - .15))
    all_tp = {r["timepoint_h"] for r in rows}
    all_tr = {r["grp_treat"] for r in rows}
    pen = 0
    for sp in ("train", "val", "test"):
        pen += len(all_tp - tp_cov[sp]) * 1.0
        pen += len(all_tr - tr_cov[sp]) * 0.5
    return pen, err


best = None
for seed in range(400):
    a = build(seed)
    s = score(a)
    if best is None or s < best[0]:
        best = (s, seed, a)

(pen, err), seed, assign = best
print(f"melhor seed={seed}  penalidade_cobertura={pen}  erro_proporcao={err:.4f}")

for r in rows:
    r["split_skov"] = assign[r["grp_field"]]

cols = list(rows[0].keys())
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

# ---- relatorio + verificacao ----
img = Counter(r["split_skov"] for r in rows)
grp = Counter(assign.values())
print(f"\nImagens: {dict(img)}  total={sum(img.values())}")
print(f"Grupos : {dict(grp)}  total={sum(grp.values())}")

chk = defaultdict(set)
for r in rows:
    chk[r["grp_field"]].add(r["split_skov"])
print(f">>> field groups in >1 partition: {sum(1 for v in chk.values() if len(v)>1)}")

print("\nTimepoints by partition:")
tp = defaultdict(Counter)
for r in rows:
    tp[r["split_skov"]][r["timepoint_h"]] += 1
for sp in ("train", "val", "test"):
    print(f"  {sp}: {dict(sorted(tp[sp].items(), key=lambda x: int(x[0])))}")

print("\nCohorts (experiment × treatment) by partition:")
tr = defaultdict(Counter)
for r in rows:
    tr[r["grp_treat"]][r["split_skov"]] += 1
for k in sorted(tr):
    print(f"  {k:<18} train={tr[k]['train']:>3} val={tr[k]['val']:>3} test={tr[k]['test']:>3}")

json.dump({"seed": seed, "imagens": dict(img), "grupos": dict(grp),
           "grupos_partidos": sum(1 for v in chk.values() if len(v) > 1)},
          open("stage1/split_skov.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nSalvos: data/mapping_final_skov.csv (col. split_skov), stage1/split_skov.json")
