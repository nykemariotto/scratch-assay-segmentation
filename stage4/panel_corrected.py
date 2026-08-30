# -*- coding: utf-8 -*-
"""
stage4/panel_corrected.py — paineis com os contornos FINAIS (corrigidos onde houve
correction, automatic where the frame was judged OK).

  --implausiveis : the series with an implausible closure after the correction.
                   Se o contorno estiver certo e a area crescer, a leitura e
                   biology (retraction/detachment), not a residual failure.
  --serie SK     : one specific series (all timepoints × both fields).

Cor do contorno indica a procedencia: VERDE = corrigido manualmente,
AMARELO = automatico (frame julgado OK), MAGENTA = fechada (area 0),
VERMELHO = frame invalido.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw

AREAS = "data/whst_areas_final.csv"
CLOS = "stage4/closure_final_por_serie.csv"
INP = "whst_input"
M_AUTO = "whst_output/masks"
M_CORR = "whst_output/rois_corrected/masks"
TH = 250

COR = {"corrigida": (60, 230, 60), "corrigida_fechada": (255, 60, 255),
       "automatica": (255, 210, 0), "automatica_PENDENTE": (255, 140, 0),
       "invalida": (255, 50, 50)}


def base(f):
    for e in (".tiff", ".tif"):
        if f.lower().endswith(e):
            return f[: -len(e)]
    return os.path.splitext(f)[0]


def to8(p):
    a = np.asarray(Image.open(p))
    if a.ndim == 3:
        a = a[..., :3].astype(np.float64).mean(axis=2)
    else:
        a = a.astype(np.float64)
        if a.max() > 255:
            a = (a - a.min()) / max(1e-9, a.max() - a.min()) * 255
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def mask_of(r):
    b = base(r["whst_input_file"])
    if r["fonte_area"].startswith("corrigida"):
        p = os.path.join(M_CORR, b + "_mask.png")
        if os.path.exists(p):
            return np.asarray(Image.open(p).convert("L")) > 127
    p = os.path.join(M_AUTO, b + "_mask.png")
    if os.path.exists(p):
        return np.asarray(Image.open(p).convert("L")) > 127
    return None


def tile(r, closure=None):
    p = os.path.join(INP, r["whst_input_file"])
    if not os.path.exists(p):
        return None
    img = to8(p)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mk = mask_of(r)
    col = COR.get(r["fonte_area"], (180, 180, 180))
    if mk is not None and mk.any():
        cont, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cont, -1, col, 14)
    im = Image.fromarray(rgb); im.thumbnail((TH, TH))
    t = Image.new("RGB", (im.width + 6, im.height + 32), (20, 20, 20))
    t.paste(im, (3, 29))
    d = ImageDraw.Draw(t)
    a = r["area_pct_final"]
    lab = f"tp{r['timepoint_h']}h c{r['campo']}  A={a if a!='' else 'NA'}%"
    d.text((4, 2), lab, fill=col)
    if closure is not None:
        d.text((4, 15), f"closure={closure:+.3f}", fill=(200, 220, 255))
    else:
        d.text((4, 15), r["fonte_area"][:22], fill=(160, 160, 160))
    return t


A = list(csv.DictReader(open(AREAS, encoding="utf-8-sig")))
C = {r["series_key"]: r for r in csv.DictReader(open(CLOS, encoding="utf-8-sig"))}
bys = defaultdict(list)
for r in A:
    bys[r["series_key"]].append(r)


def painel(chaves, fname, titulo):
    blocos = []
    for sk in chaves:
        rs = sorted(bys[sk], key=lambda r: (str(r["campo"]), int(r["timepoint_h"])))
        if not rs:
            continue
        meta = C.get(sk, {})
        cl = {}
        if meta.get("closure_seq"):
            for part in meta["closure_seq"].split(";"):
                tp, v = part.split("h:")
                cl[int(tp)] = float(v)
        campos = sorted({r["campo"] for r in rs})
        tps = sorted({int(r["timepoint_h"]) for r in rs})
        grid = {(r["campo"], int(r["timepoint_h"])): r for r in rs}
        tiles = {}
        for c in campos:
            for t in tps:
                r = grid.get((c, t))
                if r:
                    x = tile(r, cl.get(t))
                    if x:
                        tiles[(c, t)] = x
        if not tiles:
            continue
        cw = max(x.width for x in tiles.values()) + 6
        ch = max(x.height for x in tiles.values()) + 6
        head = 30
        b = Image.new("RGB", (max(len(tps) * cw, 780), head + len(campos) * ch), (12, 12, 12))
        d = ImageDraw.Draw(b)
        d.text((6, 4), f"{rs[0]['analysis_unit']}  |  campo {'/'.join(campos)}", fill=(130, 210, 255))
        d.text((6, 17), f"motivo: {meta.get('motivo','')}   closure_final={meta.get('closure_final','')}"
                        f"   procedencia={meta.get('procedencia_area','')}", fill=(200, 200, 140))
        for i, c in enumerate(campos):
            for j, t in enumerate(tps):
                if (c, t) in tiles:
                    b.paste(tiles[(c, t)], (j * cw + 3, head + i * ch + 3))
        blocos.append(b)
    if not blocos:
        print("nada a desenhar"); return
    W = max(b.width for b in blocos)
    H = sum(b.height for b in blocos) + 40 + 8 * len(blocos)
    pan = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(pan)
    d.text((8, 6), titulo, fill=(255, 255, 255))
    d.text((8, 22), "contorno: VERDE=corrigido  AMARELO=automatico(OK)  "
                    "MAGENTA=fechada(area 0)  LARANJA=pendente", fill=(170, 170, 170))
    y = 40
    for b in blocos:
        pan.paste(b, (0, y)); y += b.height + 8
    pan.save(fname)
    print(f"saved {fname}  ({len(blocos)} series, {pan.width}x{pan.height})")


if "--implausiveis" in sys.argv:
    ks = [k for k, v in C.items() if v["analisavel"] == "nao" and v["motivo"] != "sem_baseline"]
    ks.sort(key=lambda k: C[k]["analysis_unit"])
    painel(ks, "painel_implausiveis.png",
           f"SERIES WITH AN IMPLAUSIBLE CLOSURE AFTER CORRECTION (n={len(ks)}) — "
           f"right contour + growing area = biology, not failure")
elif "--serie" in sys.argv:
    alvo = sys.argv[sys.argv.index("--serie") + 1]
    ks = [k for k in bys if alvo.lower() in k.lower()]
    if not ks:
        sys.exit(f"no series matches {alvo!r}")
    painel(sorted(ks), f"painel_serie.png", f"SERIE: {alvo}")
else:
    print(__doc__)
