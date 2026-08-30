# -*- coding: utf-8 -*-
"""
stage1/check_640_population.py — explica a diferenca 118 (inventario) x 115 (dataset).

stage1/check_coco_dims.py has already verified that there is no divergence between the dimension
registrada no COCO e a do arquivo em disco (0/1363), portanto os labels estao
intact and the question is one of POPULATION, not of scale.

Here we identify exactly which 640x640 images are in the base export but not in the
materialised dataset, and why.
"""
import csv, json, os
from collections import Counter
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
BASE = (os.environ.get("ROBOFLOW_EXPORT", "<roboflow_export>")
        + r"\Pre Eclampsia.coco-segmentation\train")
ANN = os.path.join(BASE, "_annotations.coco.json")

if not os.path.isfile(ANN):
    raise SystemExit(f"export base inacessivel: {ANN}")

coco = json.load(open(ANN, encoding="utf-8"))
base640 = {im["file_name"] for im in coco["images"]
           if (im["width"], im["height"]) == (640, 640)}
print(f"base export: {len(coco['images'])} images | 640x640: {len(base640)}")

M = list(csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8")))
ativas = {r["arquivo_b"] for r in M
          if r["partition"] != "EXCLUIDA" and r["excluida"] not in ("True", "1", "sim")}
inativas = [r for r in M
            if r["partition"] == "EXCLUIDA" or r["excluida"] in ("True", "1", "sim")]
print(f"mapping: {len(M)} rows | active {len(ativas)} | inactive {len(inativas)}")

fora = sorted(base640 - ativas)
print(f"\n=== 640x640 in the base export but NOT in the materialised dataset: {len(fora)} ===")
byname = {r["arquivo_b"]: r for r in M}
for f in fora:
    r = byname.get(f)
    if r:
        print(f"  {f[:60]}")
        print(f"      excluida={r['excluida']!r}  partition={r['partition']!r}  "
              f"group={r['group_key'][:34]}")
    else:
        print(f"  {f[:60]}\n      [absent from the mapping]")

print(f"\n=== the {len(inativas)} inactive mapping rows: dimension in the base export ===")
dims = {im["file_name"]: (im["width"], im["height"]) for im in coco["images"]}
c = Counter()
for r in inativas:
    d = dims.get(r["arquivo_b"], "(ausente do export base)")
    c[str(d)] += 1
    print(f"  {r['arquivo_b'][:56]:<58} {d}")
print(f"\n  summary of the inactive ones by dimension: {dict(c)}")

n640_ativas = len(base640 & ativas)
print("\n=== ACCOUNTING ===")
print(f"  640 in the base export  : {len(base640)}")
print(f"  640 inactive (excluded) : {len(fora)}")
print(f"  640 active (dataset)    : {n640_ativas}")
print(f"  {len(base640)} - {len(fora)} = {len(base640)-len(fora)}  "
      f"{'MATCHES' if len(base640)-len(fora) == n640_ativas else 'DOES NOT ADD UP'}")
