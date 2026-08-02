# -*- coding: utf-8 -*-
"""
stage1/check_coco_dims.py — CRITICAL CHECK of label integrity.

stage1/coco_to_yolo_seg.py normalises the polygon coordinates by dividing by
im["width"]/im["height"] of the COCO RECORD, not by the real dimension of the file.
If the two diverge for any image, the YOLO label of that image
esta com coordenadas normalizadas pelo denominador errado -> mascara deslocada
ou escalada, corrompendo treino e avaliacao.

For each image of each partition it checks:
  1. dimensao registrada no COCO  x  dimensao real do arquivo em disco
  2. if they diverge, quantifies the scale error that caused in the label
  3. checa tambem se algum label YOLO tem coordenada fora de [0,1]
"""
import csv, json, os
from collections import Counter, defaultdict
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
COCO_DIR = "coco_partitions"
IMG_ROOT = os.path.join("dataset", "images")
LAB_ROOT = os.path.join("dataset", "labels")

tot = div = 0
divergentes = []
dims_coco = Counter()
dims_real = Counter()

for part in ("train", "val", "test"):
    p = os.path.join(COCO_DIR, f"instances_{part}.json")
    if not os.path.isfile(p):
        print(f"[FALTA] {p}")
        continue
    coco = json.load(open(p, encoding="utf-8"))
    for im in coco["images"]:
        fn = im["file_name"]
        W, H = im["width"], im["height"]
        dims_coco[(W, H)] += 1
        f = os.path.join(IMG_ROOT, part, fn)
        if not os.path.exists(f):
            divergentes.append((part, fn, (W, H), None, "arquivo ausente"))
            continue
        with Image.open(f) as img:
            rw, rh = img.size
        dims_real[(rw, rh)] += 1
        tot += 1
        if (W, H) != (rw, rh):
            div += 1
            divergentes.append((part, fn, (W, H), (rw, rh),
                                f"escala x={rw/W:.4f} y={rh/H:.4f}"))

print("=== DIMENSOES: registro COCO x arquivo real ===")
print(f"  images checked: {tot}")
print(f"  DIVERGENTES: {div}")
print()
print(f"  distribuicao no COCO  : {dict(dims_coco)}")
print(f"  distribuicao no disco : {dict(dims_real)}")
if divergentes:
    print("\n  casos divergentes (ate 20):")
    for part, fn, c, r, obs in divergentes[:20]:
        print(f"    [{part}] {fn[:56]}")
        print(f"        COCO={c}  real={r}  {obs}")
else:
    print("\n  -> TODAS as dimensoes conferem: a normalizacao usou o denominador")
    print("     correto, portanto os labels YOLO estao integros quanto a escala.")

# ---- checagem independente: coordenadas fora de [0,1] nos labels ----
print("\n=== LABELS YOLO: coordenadas fora de [0,1] ===")
fora = 0
vazios = 0
nlab = 0
piores = []
for part in ("train", "val", "test"):
    d = os.path.join(LAB_ROOT, part)
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if not f.endswith(".txt"):
            continue
        nlab += 1
        pth = os.path.join(d, f)
        if os.path.getsize(pth) == 0:
            vazios += 1
            continue
        mx = 0.0
        for ln in open(pth):
            v = ln.split()
            for x in v[1:]:
                val = float(x)
                mx = max(mx, abs(val - 0.5))
                if val < -1e-9 or val > 1 + 1e-9:
                    fora += 1
        if mx > 0.5 + 1e-6:
            piores.append((part, f, mx))
print(f"  labels: {nlab} ({vazios} vazios)")
print(f"  coordenadas fora de [0,1]: {fora}")
if piores:
    print(f"  arquivos com coordenada fora: {len(piores)} (ate 10)")
    for p, f, m in piores[:10]:
        print(f"    [{p}] {f[:60]}  desvio_max={m:.4f}")
else:
    print("  -> nenhuma coordenada fora do intervalo")

# ---- populacao 640: inventario x dataset ----
print("\n=== POPULACAO 640x640: inventario x dataset materializado ===")
n640_disco = sum(v for k, v in dims_real.items() if k == (640, 640))
n640_coco = sum(v for k, v in dims_coco.items() if k == (640, 640))
print(f"  640x640 in the per-partition COCO : {n640_coco}")
print(f"  640x640 on disk               : {n640_disco}")
if os.path.isfile("data/image_quality_metrics.csv"):
    q = list(csv.DictReader(open("data/image_quality_metrics.csv", encoding="utf-8-sig")))
    print(f"  640x640 in image_quality_metrics: {sum(1 for r in q if r['res'] == '640')}")
