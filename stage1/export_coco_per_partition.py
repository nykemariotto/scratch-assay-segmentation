# -*- coding: utf-8 -*-
"""
stage1/export_coco_per_partition.py — M4. Divide o COCO BASE em train/val/test conforme
stage1/mapping_dataset_final.csv.

TRAVA CRITICA: so o export base (Pre Eclampsia.coco-segmentation), nunca v23-v37.
Join on file_name (== arquivo_b), which is unique per image in the base export.
"""
import json, csv, os, sys, re
from collections import Counter, defaultdict

COCO_BASE = (os.environ.get("ROBOFLOW_EXPORT", "<roboflow_export>")
             + r"\Pre Eclampsia.coco-segmentation\train\_annotations.coco.json")
# D6: versao estratificada por cell_line x tratamento (0 estratos ausentes)
MAPPING_FINAL = "data/mapping_dataset_final_strat.csv"
OUT_DIR = "coco_partitions"

FORBIDDEN = [f"v{n}" for n in range(23, 38)]

# ---------------------------------------------------------------- TRAVA
low = COCO_BASE.lower()
hit = [f for f in FORBIDDEN if re.search(rf"\.{f}-", low) or f"{f}-" in low]
if hit:
    sys.exit(f"ABORTADO: caminho parece versao proibida {hit}: {COCO_BASE}")
if "pre eclampsia.coco-segmentation" not in low:
    sys.exit(f"ABORTED: this is not the expected base export: {COCO_BASE}")
print("TRAVA OK — export base confirmado:")
print(f"   {COCO_BASE}\n")

coco = json.load(open(COCO_BASE, encoding="utf-8"))
rows = list(csv.DictReader(open(MAPPING_FINAL, encoding="utf-8")))
name_to_part = {r["arquivo_b"]: r["partition"] for r in rows}

parts = {"train": [], "val": [], "test": []}
excluded, unmatched = [], []
for img in coco["images"]:
    fn = img.get("file_name")
    p = name_to_part.get(fn)
    if p is None:
        # fallback: tentar pelo nome original preservado pelo Roboflow
        orig = (img.get("extra") or {}).get("name")
        p = name_to_part.get(orig)
    if p is None:
        unmatched.append(fn)
    elif p == "EXCLUIDA":
        excluded.append(fn)
    else:
        parts[p].append(img)

print(f"Images in the base COCO : {len(coco['images'])}")
print(f"  casadas e atribuidas: {sum(len(v) for v in parts.values())}")
print(f"  excluidas (1 ambigua + 2 escala): {len(excluded)} -> {excluded}")
print(f"  UNMATCHED: {len(unmatched)}")
if unmatched:
    print("   amostra:", unmatched[:10])

os.makedirs(OUT_DIR, exist_ok=True)
tot_ann = 0
print("\nExportando:")
for part, imgs in parts.items():
    ids = {im["id"] for im in imgs}
    anns = [a for a in coco["annotations"] if a["image_id"] in ids]
    tot_ann += len(anns)
    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": imgs,
        "annotations": anns,
    }
    path = os.path.join(OUT_DIR, f"instances_{part}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    vazias = len(ids) - len({a["image_id"] for a in anns})
    print(f"  {part:<6} {len(imgs):>4} imgs  {len(anns):>4} anotacoes  "
          f"({vazias} images without annotation)  -> {path}")

# conservation of the annotations
ann_excl = [a for a in coco["annotations"]
            if a["image_id"] in {im["id"] for im in coco["images"]
                                 if im.get("file_name") in excluded}]
print(f"\nAnotacoes: {len(coco['annotations'])} no base = {tot_ann} exportadas "
      f"+ {len(ann_excl)} nas excluidas")
assert tot_ann + len(ann_excl) == len(coco["annotations"]), "ANNOTATION LOSS"
print("Conservacao de anotacoes OK")

# resolution of the exported images (warning about the 640x640 remapping)
dims = Counter((im["width"], im["height"]) for im in coco["images"])
print(f"\nResolucoes no base: {dict(dims)}")
print("WARNING: the 640x640 images have coordinates in that space (Roboflow stretched them).")
print("  To use them against the raw 2452x2056: x*2452/640, y*2056/640 (anisotropic).")
print("\nPer-partition COCO exported from the BASE EXPORT.")
