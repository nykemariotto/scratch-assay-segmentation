# -*- coding: utf-8 -*-
"""
stage1/inspect_ambiguous.py — adjudication panel for the 12 AMBIGUO cases.
For each analysis unit containing >=1 AMBIGUO, it builds the grid
field (row) × timepoint (column) of the P0 overlays (contour already drawn),
destacando em VERMELHO o(s) frame(s) AMBIGUO. Assim o campo-irmao e o contexto
temporal context sits side by side for the judgement. It reuses whst_output/overlays/.
"""
import csv, os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

OVDIR = "whst_output/overlays"
OUT = "inspect_ambiguous.png"
TH = 240   # lado do thumbnail

hum = {r["whst_input_file"]: r for r in csv.DictReader(open("data/visual_triage.csv", encoding="utf-8-sig"))}
auto = {r["whst_input_file"]: r for r in csv.DictReader(open("data/whst_pass1_qc.csv", encoding="utf-8-sig"))}
mp = {r["whst_input_file"]: r for r in csv.DictReader(open("whst_output/overlays_sorted_map.csv", encoding="utf-8-sig"))}


def hcat(r):
    return r["categoria"] if r["categoria"] != "SEG_RUIM" else "SEG_" + r["subtipo"]


# units containing an AMBIGUO
amb_units = sorted({auto[k]["analysis_unit"] for k in hum if hcat(hum[k]) == "AMBIGUO"})

# index images by unit
by_unit = defaultdict(list)
for k in auto:
    by_unit[auto[k]["analysis_unit"]].append(k)

COLOR = {"AMBIGUO": (255, 60, 60), "OK": (80, 220, 80), "SEG_super": (255, 190, 0),
         "SEG_sub": (0, 190, 255), "IMG_INVALIDA": (200, 0, 200)}


def tile(k):
    src = mp[k]["overlay_source"]
    p = os.path.join(OVDIR, src)
    im = Image.open(p).convert("RGB"); im.thumbnail((TH, TH))
    cat = hcat(hum[k])
    col = COLOR.get(cat, (150, 150, 150))
    bw = 6 if cat == "AMBIGUO" else 3
    t = Image.new("RGB", (im.width + 8, im.height + 34), (25, 25, 25))
    t.paste(im, (4, 30))
    d = ImageDraw.Draw(t)
    d.rectangle([4, 30, 4 + im.width - 1, 30 + im.height - 1], outline=col, width=bw)
    a = auto[k]
    lab = f"tp{a['timepoint_h']}h c{a['campo']}  {cat}"
    d.text((5, 3), lab, fill=col)
    d.text((5, 16), f"area={a['area_pct']}%", fill=(200, 200, 200))
    return t


blocks = []
for u in amb_units:
    ks = by_unit[u]
    campos = sorted({auto[k]["campo"] for k in ks})
    tps = sorted({int(auto[k]["timepoint_h"]) for k in ks})
    grid = {(auto[k]["campo"], int(auto[k]["timepoint_h"])): k for k in ks}
    cw, ch = TH + 8, TH + 34
    head = 24
    blk = Image.new("RGB", (max(len(tps) * cw, 700), head + len(campos) * ch), (15, 15, 15))
    d = ImageDraw.Draw(blk)
    namb = sum(1 for k in ks if hcat(hum[k]) == "AMBIGUO")
    d.text((6, 5), f"[{u}]   AMBIGUO={namb}  (vermelho=AMBIGUO a adjudicar; verde=OK; amarelo=super)",
           fill=(120, 210, 255))
    for i, c in enumerate(campos):
        for j, t in enumerate(tps):
            k = grid.get((c, t))
            if k:
                blk.paste(tile(k), (j * cw, head + i * ch))
    blocks.append(blk)

W = max(b.width for b in blocks)
gap = 10
H = sum(b.height for b in blocks) + gap * (len(blocks) + 1)
panel = Image.new("RGB", (W, H), (0, 0, 0))
y = gap
for b in blocks:
    panel.paste(b, (0, y)); y += b.height + gap
panel.save(OUT)
print(f"saved {OUT}  ({len(blocks)} units, {panel.width}x{panel.height})")
