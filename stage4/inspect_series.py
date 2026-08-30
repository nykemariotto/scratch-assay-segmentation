# -*- coding: utf-8 -*-
"""
stage4/inspect_series.py — Paineis de inspecao visual de validade do ensaio.
For each suspect unit: all timepoints × both fields, showing the RAW image without
a contour next to the RAW image with the P0 contour overlaid.
Includes a clean control series from the same batch for comparison.
Pipeline P0 identico ao stage4/whst_batch.ijm (reimplementacao ja validada, dif 0,1%).
"""
import csv, os
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

OUT = "whst_input"
RAD, THR, SAT, MIN_AREA = 20, 100, 0.001, 100


def to_8bit(path):
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., :3].astype(np.float64).mean(axis=2)
    else:
        a = a.astype(np.float64)
        if a.max() > 255:
            a = (a - a.min()) / max(1e-9, a.max() - a.min()) * 255.0
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def enhance(img8, sat):
    h = np.bincount(img8.ravel(), minlength=256).astype(np.float64)
    thr = img8.size * sat / 200.0
    c = 0.0; hmin = 0
    for i in range(256):
        c += h[i]
        if c > thr: hmin = i; break
    c = 0.0; hmax = 255
    for i in range(255, -1, -1):
        c += h[i]
        if c > thr: hmax = i; break
    if hmax <= hmin: return img8.copy()
    return np.clip(np.round((img8.astype(np.float64) - hmin) * (255.0 / (hmax - hmin))), 0, 255).astype(np.uint8)


def seg_p0(path):
    img8 = to_8bit(path); H, W = img8.shape
    e = enhance(img8, SAT)
    r = int(np.ceil(RAD)); y, x = np.mgrid[-r:r+1, -r:r+1]
    k = ((x*x + y*y) <= RAD*RAD + 1).astype(np.float64); k /= k.sum()
    f = e.astype(np.float64)
    m = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.filter2D(f*f, -1, k, borderType=cv2.BORDER_REFLECT)
    var = np.clip(np.round(np.maximum(m2 - m*m, 0)), 0, 255).astype(np.uint8)
    mask = binary_fill_holes((var <= THR)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bi, ba = 0, 0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a >= MIN_AREA and a > ba: ba, bi = a, i
    mk = (lab == bi).astype(np.uint8) if bi else np.zeros_like(mask)
    return img8, mk, (100.0 * ba / (W * H) if bi else 0.0)


qc = list(csv.DictReader(open("data/whst_pass1_qc.csv", encoding="utf-8-sig")))
units = sorted({r["analysis_unit"] for r in qc})


def find(sub):
    for u in units:
        if all(s in u for s in sub):
            return u
    return None


SUSP = [find(["n1", "PEP + NEB 5uM", "D4"]),
        find(["n2", "PET + NEB 5uM", "D3"]),
        find(["n3", "D4"])]
# controle: unidade Migracao com TODAS as medicoes OK e >=3 timepoints
byu = defaultdict(list)
for r in qc:
    byu[r["analysis_unit"]].append(r)
# Nenhuma serie do lote Migracao passa 100% limpa -> dois controles:
#  A) melhor serie do PROPRIO lote Migracao (mesmas condicoes de imagem)
#  B) serie 100% OK com fechamento textbook (referencia de monocamada saudavel)
ctrl_lote = find(["n2", "PEP + NEB 1uM", "B1"])
ctrl_sadio = find(["originais", "D1"]) or find(["Originais", "D1"])
print("suspeitas:", SUSP)
print("control, same batch:", ctrl_lote)
print("controle saudavel  :", ctrl_sadio)

TARGETS = ([(u, "SUSPEITA") for u in SUSP if u]
           + ([(ctrl_lote, "CONTROLE mesmo lote Migracao")] if ctrl_lote else [])
           + ([(ctrl_sadio, "CONTROLE saudavel (fechamento normal)")] if ctrl_sadio else []))
TH = 250   # height of each sub-image


def cell(path, label):
    img8, mk, pct = seg_p0(path)
    raw = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    ov = raw.copy()
    cont, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cont, -1, (0, 255, 255), 14)
    a = Image.fromarray(raw); a.thumbnail((TH, TH))
    b = Image.fromarray(ov);  b.thumbnail((TH, TH))
    tile = Image.new("RGB", (a.width + b.width + 6, a.height + 18), (18, 18, 18))
    tile.paste(a, (0, 18)); tile.paste(b, (a.width + 6, 18))
    ImageDraw.Draw(tile).text((3, 3), f"{label}  P0={pct:.1f}%", fill=(255, 235, 120))
    return tile


for u, tag in TARGETS:
    rs = byu[u]
    tps = sorted({int(r["timepoint_h"]) for r in rs})
    campos = sorted({r["campo"] for r in rs})
    grid = {}
    for r in rs:
        grid[(r["campo"], int(r["timepoint_h"]))] = r
    tiles = {}
    for c in campos:
        for t in tps:
            r = grid.get((c, t))
            if not r:
                continue
            p = os.path.join(OUT, r["whst_input_file"])
            tiles[(c, t)] = cell(p, f"c{c} tp{t}h")
    if not tiles:
        continue
    cw = max(x.width for x in tiles.values()) + 6
    ch = max(x.height for x in tiles.values()) + 6
    pan = Image.new("RGB", (len(tps) * cw, len(campos) * ch + 22), (30, 30, 30))
    ImageDraw.Draw(pan).text((6, 5), f"[{tag}] {u}   (esq=crua  dir=contorno P0)", fill=(140, 220, 255))
    for i, c in enumerate(campos):
        for j, t in enumerate(tps):
            if (c, t) in tiles:
                pan.paste(tiles[(c, t)], (j * cw + 3, 22 + i * ch + 3))
    safe = "".join(ch2 if ch2.isalnum() else "_" for ch2 in u)[:52]
    fn = f"inspect_{safe}.png"
    pan.save(fn)
    print(f"  panel: {fn}  ({len(tiles)} images, {len(campos)} fields × {len(tps)} timepoints)")
print("\nconcluido")
