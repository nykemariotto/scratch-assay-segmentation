# -*- coding: utf-8 -*-
"""
QC (a) — distribution of annotation area over the whole dataset (train+val+test).
Area = soma das areas dos POLIGONOS (shoelace), NAO o campo 'area' do COCO (que e
area da bbox e superestima). Fracao = area_poligono / (W*H) da propria imagem
(normalises 640 vs native). Groups by cell_line × timepoint. It removes nothing.
"""
import json, csv, os
from collections import defaultdict
import numpy as np

MAP = "data/mapping_dataset_final_strat.csv"
CD = "coco_partitions"

# cell_line / timepoint por imagem
meta = {}
for x in csv.DictReader(open(MAP, encoding="utf-8")):
    meta[x["arquivo_b"]] = ("SKOV-3" if x["linha_celular"] == "SKOV" else x["linha_celular"],
                            x["timepoint_h"])


def shoelace(seg):
    x = np.array(seg[0::2]); y = np.array(seg[1::2])
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


recs = []
for part in ("train", "val", "test"):
    d = json.load(open(os.path.join(CD, f"instances_{part}.json"), encoding="utf-8"))
    ann = defaultdict(list)
    for a in d["annotations"]:
        ann[a["image_id"]].append(a)
    for im in d["images"]:
        W, H = im["width"], im["height"]
        aa = ann.get(im["id"], [])
        poly_area = sum(shoelace(s) for a in aa for s in a.get("segmentation", []) if len(s) >= 6)
        cl, tp = meta.get(im["file_name"], ("?", "?"))
        recs.append({"file": im["file_name"], "cell": cl, "tp": tp, "part": part,
                     "res": "640" if (W, H) == (640, 640) else "nativo",
                     "n_ann": len(aa), "frac": 100.0 * poly_area / (W * H)})

with_ann = [r for r in recs if r["n_ann"] > 0]
empty = [r for r in recs if r["n_ann"] == 0]
print(f"images: {len(recs)} | with annotation: {len(with_ann)} | empty (0 ann): {len(empty)}")

# ---- distribution by cell_line × timepoint (annotated images only) ----
print("\n=== ANNOTATED AREA FRACTION (%) by cell_line × timepoint — annotated images ===")
print(f"{'cell':<7}{'tp':>4}  {'n':>4}  {'med':>7}{'Q1':>7}{'Q3':>7}{'min':>7}{'max':>7}")
grp = defaultdict(list)
for r in with_ann:
    grp[(r["cell"], r["tp"])].append(r["frac"])
for k in sorted(grp, key=lambda k: (k[0], int(k[1]))):
    v = np.array(grp[k])
    print(f"{k[0]:<7}{k[1]:>4}  {len(v):>4}  {np.median(v):>7.2f}{np.percentile(v,25):>7.2f}"
          f"{np.percentile(v,75):>7.2f}{v.min():>7.2f}{v.max():>7.2f}")

# ---- distribuicao geral p/ calibrar limiar ----
allf = np.array([r["frac"] for r in with_ann])
print(f"\noverall fraction (annotated): p5={np.percentile(allf,5):.2f} p10={np.percentile(allf,10):.2f} "
      f"p25={np.percentile(allf,25):.2f} mediana={np.median(allf):.2f}")

# ---- cauda suspeita: limiar por timepoint ----
def thr(tp):
    return 3.0 if tp == "0" else 2.0

print("\n=== CAUDA SUSPEITA (fragmentos): <3% em 0h, <2% nos demais ===")
susp = [r for r in with_ann if r["frac"] < thr(r["tp"])]
from collections import Counter
print(f"images with an annotation below the threshold: {len(susp)}")
print(f"  by partition: {dict(Counter(r['part'] for r in susp))}")
print(f"  por cell x tp: {dict(Counter((r['cell'],r['tp']) for r in susp))}")
print(f"  no TEST (critico p/ mAP): {sum(1 for r in susp if r['part']=='test')}")

# ---- empty images (0 ann) by timepoint/partition (negatives, or missing?) ----
print("\n=== IMAGES WITHOUT ANNOTATION (0 polygons) ===")
print(f"  total: {len(empty)} | by partition: {dict(Counter(r['part'] for r in empty))}")
print(f"  por cell x tp: {dict(Counter((r['cell'],r['tp']) for r in empty))}")
print(f"  0h without annotation (suspicious — 0h should have a wound): "
      f"{sum(1 for r in empty if r['tp']=='0')}")

# ---- salvar detalhe ----
recs.sort(key=lambda r: (r["part"], r["frac"]))
with open("stage1/qc_annotation_area.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "cell", "tp", "part", "res", "n_ann", "frac"])
    w.writeheader()
    for r in recs:
        r2 = dict(r); r2["frac"] = round(r["frac"], 3); w.writerow(r2)

# lista da cauda suspeita separada
with open("data/qc_suspeitas.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "cell", "tp", "part", "res", "n_ann", "frac_pct", "motivo"])
    for r in sorted(susp, key=lambda r: r["frac"]):
        w.writerow([r["file"], r["cell"], r["tp"], r["part"], r["res"], r["n_ann"],
                    round(r["frac"], 3), f"<{thr(r['tp'])}% (fragmento provavel)"])
    for r in sorted(empty, key=lambda r: (r["cell"], r["tp"])):
        w.writerow([r["file"], r["cell"], r["tp"], r["part"], r["res"], 0, 0.0, "sem anotacao (0 poligonos)"])
print("\nSalvos: stage1/qc_annotation_area.csv (todas), data/qc_suspeitas.csv (cauda + vazias)")
