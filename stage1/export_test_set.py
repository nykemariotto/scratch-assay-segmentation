# -*- coding: utf-8 -*-
"""
stage1/export_test_set.py — exports the TEST partition for manual measurement (Fiji), ordered
por grupo e timepoint, com o caminho da imagem crua no banco A.
"""
import csv, os
from collections import Counter, defaultdict

BANCO_A = os.environ.get("BANCO_A", "<banco_a>")
TEST_IMG = os.path.abspath(os.path.join("dataset", "images", "test"))
SRC = "data/mapping_dataset_final_strat.csv"
OUT = "stage1/test_set_for_whst.csv"

rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["partition"] == "test"]

# expected timepoints per series (to define a "complete group")
EXP_TP = {"HUVEC": {0, 8, 12, 24}, "P1": {0, 24, 48, 72}, "P2": {0, 24, 48}}


def exp_set(r):
    if r["linha_celular"] == "HUVEC":
        return EXP_TP["HUVEC"]
    return EXP_TP.get(r["lote"], set())


def raw_path(r):
    if r["arquivo_a"].strip():
        return os.path.join(BANCO_A, r["pasta_a"].replace("/", os.sep), r["arquivo_a"])
    return ""  # 19 HUVEC with no 1:1 physical source (case-insensitive collision)


out = []
for r in rows:
    rp = raw_path(r)
    out.append({
        "filename": r["arquivo_b"],
        "cell_line": "SKOV-3" if r["linha_celular"] == "SKOV" else r["linha_celular"],
        "group_key": r["group_key"],
        "well_or_field": r["well"],
        "treatment": r["tratamento"],
        "experiment_or_batch": r["lote"],
        "timepoint_h": int(r["timepoint_h"]),
        "raw_path_banco_a": rp,
        "raw_status": "ok_1a1" if rp else "sem_fonte_colisao_case",
        "test_image_path": os.path.join(TEST_IMG, r["arquivo_b"]),  # fallback sempre resolve
    })

# ordena por grupo, depois timepoint
out.sort(key=lambda x: (x["cell_line"], x["group_key"], x["timepoint_h"]))

cols = ["filename", "cell_line", "group_key", "well_or_field", "treatment",
        "experiment_or_batch", "timepoint_h", "raw_path_banco_a", "raw_status", "test_image_path"]
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(out)
print(f"Saved: {OUT}  ({len(out)} images)")

# =============================== DIMENSIONAMENTO ===============================
print("\n" + "=" * 60)
print("DIMENSIONAMENTO DA MEDICAO")
print("=" * 60)

by_cell = Counter(r["linha_celular"] for r in rows)
print(f"\nImagens no teste por linha celular:")
print(f"  HUVEC : {by_cell['HUVEC']}")
print(f"  SKOV-3: {by_cell['SKOV']}")
print(f"  TOTAL : {len(rows)}")

# grupos (series) por linha
grp = defaultdict(lambda: defaultdict(set))   # cell -> group -> set(tp)
grp_exp = {}
for r in rows:
    grp[r["linha_celular"]][r["group_key"]].add(int(r["timepoint_h"]))
    grp_exp[r["group_key"]] = exp_set(r)

print(f"\nGrupos (series) no teste:")
for cl in ("HUVEC", "SKOV"):
    n = len(grp[cl])
    print(f"  {cl}: {n} grupos")

print(f"\nSeries completeness (timepoints present vs expected):")
tot_complete = 0
for cl in ("HUVEC", "SKOV"):
    comp = incomp = 0
    dist = Counter()
    for g, tps in grp[cl].items():
        dist[len(tps)] += 1
        if tps >= grp_exp[g] and grp_exp[g]:
            comp += 1
        else:
            incomp += 1
    tot_complete += comp
    print(f"  {cl}: {comp} completas, {incomp} incompletas "
          f"| distrib. #timepoints/grupo: {dict(sorted(dist.items()))}")

print(f"\n  Series COMPLETAS no total: {tot_complete}")
print(f"  Images with no 1:1 raw path (use test_image_path): "
      f"{sum(1 for r in out if not r['raw_path_banco_a'])} (todas HUVEC)")

# mean series size
sizes = [len(tps) for cl in grp for tps in grp[cl].values()]
print(f"\n  Images per series: min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")
