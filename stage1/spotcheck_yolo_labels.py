# -*- coding: utf-8 -*-
"""
stage1/spotcheck_yolo_labels.py — Gate 2.0: valida a conversao lendo os LABELS YOLO
(entrada real do treino) e desenhando-os sobre as imagens do dataset.
Amostra 5 imagens 640x640 e 5 nativas 2452x2056, todas com anotacao.
"""
import os, random
from PIL import Image, ImageDraw

ROOT = "dataset"
PART = "train"
OUT = "spotcheck_labels.png"
TILE = 460

img_dir = os.path.join(ROOT, "images", PART)
lab_dir = os.path.join(ROOT, "labels", PART)

files = sorted(os.listdir(img_dir))
small, native = [], []
for fn in files:
    lab = os.path.join(lab_dir, os.path.splitext(fn)[0] + ".txt")
    if not os.path.exists(lab) or os.path.getsize(lab) == 0:
        continue
    with Image.open(os.path.join(img_dir, fn)) as im:
        wh = im.size
    (small if wh == (640, 640) else native).append(fn)
    if len(small) >= 40 and len(native) >= 40:
        break

random.Random(7).shuffle(small)
random.Random(7).shuffle(native)
pick = [("640", f) for f in small[:5]] + [("nativo", f) for f in native[:5]]
print(f"640x640 com anotacao disponiveis: {len(small)} | nativas: {len(native)}")

tiles = []
for kind, fn in pick:
    im = Image.open(os.path.join(img_dir, fn)).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lab = os.path.join(lab_dir, os.path.splitext(fn)[0] + ".txt")
    n = 0
    for line in open(lab, encoding="utf-8"):
        p = line.split()
        if len(p) < 7:
            continue
        c = list(map(float, p[1:]))
        pts = [(c[i] * W, c[i + 1] * H) for i in range(0, len(c) - 1, 2)]
        d.line(pts + [pts[0]], fill=(0, 255, 255), width=max(3, W // 200))
        n += 1
    im.thumbnail((TILE, TILE), Image.BILINEAR)
    tiles.append((kind, fn, n, im))
    print(f"  [{kind:<6}] {fn[:42]:<42} {W}x{H}  poligonos={n}")

cols, rows = 5, 2
cw = max(t[3].width for t in tiles) + 8
ch = max(t[3].height for t in tiles) + 8
pan = Image.new("RGB", (cols * cw, rows * ch), (18, 18, 18))
for i, (kind, fn, n, im) in enumerate(tiles):
    x, y = (i % cols) * cw, (i // cols) * ch
    pan.paste(im, (x + 4, y + 4))
pan.save(OUT)
print(f"\nSalvo: {OUT}  (linha 1 = 640x640, linha 2 = nativas; ciano = label YOLO)")
