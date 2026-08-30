# -*- coding: utf-8 -*-
"""
stage1/remove_duplicates.py — Remove as 3 redundancias byte-identicas do dataset.
Mantem 1 copia por par, deterministicamente (alfabetica por (arquivo_a, arquivo_b)),
marks the other as EXCLUDED. It re-hashes only the candidates to confirm the pair.
It records keep/remove + the annotation difference in stage1/duplicates_removed.csv.
"""
import csv, os, json, hashlib, shutil
from collections import defaultdict

BA = os.environ.get("BANCO_A", "<banco_a>")
COCO = os.environ.get("ROBOFLOW_EXPORT", "<roboflow_export>") + r"\Pre Eclampsia.coco-segmentation\train\_annotations.coco.json"
CSV = "data/mapping_dataset_final_strat.csv"

# raws duplicados (do stage1/leakage_md5_check.json), por (pasta, arquivo_a)
dup_json = json.load(open("stage1/leakage_md5_check.json", encoding="utf-8"))
dup_files = set()
for d in dup_json["duplicados_intra"]:
    for a in d["arquivos"]:
        dup_files.add(a)
print("raw files involved in a duplicate:", sorted(dup_files))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
active = [r for r in rows if r["partition"] != "EXCLUIDA" and r["arquivo_a"].strip()]

# hash so os candidatos (arquivo_a em dup_files)
cand = [r for r in active if r["arquivo_a"] in dup_files]
by_md5 = defaultdict(list)
for r in cand:
    p = os.path.join(BA, r["pasta_a"].replace("/", os.sep), r["arquivo_a"])
    by_md5[md5(p)].append(r)
pairs = {h: sorted(v, key=lambda r: (r["arquivo_a"], r["arquivo_b"])) for h, v in by_md5.items() if len(v) > 1}
print(f"pairs confirmed by MD5: {len(pairs)}")

# anotacoes p/ registrar diferenca
coco = json.load(open(COCO, encoding="utf-8"))
imgid = {im["file_name"]: im["id"] for im in coco["images"]}
ann_by = defaultdict(list)
for a in coco["annotations"]:
    ann_by[a["image_id"]].append(a)


def ann_stats(fb):
    aa = ann_by.get(imgid.get(fb), [])
    nv = sum(len(s) // 2 for a in aa for s in a.get("segmentation", []))
    area = sum(a.get("area", 0) for a in aa)
    return len(aa), nv, round(area, 1)


remove_b = set()
log = []
for h, v in pairs.items():
    keep, *rem = v
    kn, kv, ka = ann_stats(keep["arquivo_b"])
    for r in rem:
        remove_b.add(r["arquivo_b"])
        rn, rv, ra = ann_stats(r["arquivo_b"])
        log.append({"md5": h, "particao": keep["partition"], "group_key": keep["group_key"],
                    "kept_arquivo_b": keep["arquivo_b"], "kept_arquivo_a": keep["arquivo_a"],
                    "kept_anotacoes": kn, "kept_vertices": kv, "kept_area": ka,
                    "removed_arquivo_b": r["arquivo_b"], "removed_arquivo_a": r["arquivo_a"],
                    "removed_anotacoes": rn, "removed_vertices": rv, "removed_area": ra})

print(f"images to REMOVE: {len(remove_b)}")

# backup + aplicar
shutil.copy2(CSV, "stage1/mapping_dataset_final_strat.prededup.csv")
n_marked = 0
for r in rows:
    if r["arquivo_b"] in remove_b:
        r["partition"] = "EXCLUIDA"
        r["excluida"] = "1"
        r["resolucao"] = "removida_duplicata_md5"
        n_marked += 1
with open(CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"marked EXCLUIDA: {n_marked}  (backup: stage1/mapping_dataset_final_strat.prededup.csv)")

with open("stage1/duplicates_removed.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
    w.writeheader(); w.writerows(log)

print("\n=== record (kept vs removed) ===")
for r in log:
    print(f"  [{r['particao']}] {r['group_key']}")
    print(f"    KEEP   {r['kept_arquivo_b'][:38]}  area={r['kept_area']} verts={r['kept_vertices']}")
    print(f"    REMOVE {r['removed_arquivo_b'][:38]}  area={r['removed_area']} verts={r['removed_vertices']}")

# novos totais
act = [r for r in rows if r["partition"] != "EXCLUIDA"]
from collections import Counter
print(f"\nactive now: {len(act)}  (was 1366)  | excluded: {len(rows)-len(act)}")
print(f"by partition: {dict(Counter(r['partition'] for r in act))}")
print("\nSaved: stage1/duplicates_removed.csv. Next: re-export COCO -> YOLO -> verify_final.")
