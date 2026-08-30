# -*- coding: utf-8 -*-
"""
stage4/whst_param_sweep.py — Teste de viabilidade de recalibracao do WHST.

Replica o pipeline de stage4/whst_batch.ijm em Python:
  8-bit -> Enhance Contrast(saturated, normalize) -> Variance(radius)
  -> setThreshold(0, threshold) -> Mask -> Fill Holes
  -> Analyze Particles(size>=100) -> MAIOR componente -> area_pct

VALIDACAO OBRIGATORIA: roda P0 (parametros congelados) e compara com o area_pct
already measured in ImageJ. If it does not reproduce, the sweep is not trustworthy
and the script says so.

Saidas: stage4/whst_param_sweep.csv + paineis visuais (contorno sobreposto) 12 x 4.
"""
import csv, os, re
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw

OUT = "whst_input"
PARAMS = [("P0", 20, 100, 0.001), ("P1", 10, 100, 0.001),
          ("P2", 20, 60, 0.001), ("P3", 20, 100, 0.01)]
MIN_AREA = 100


# ---------------- pipeline fiel ao ImageJ ----------------
def to_8bit(path):
    """ImageJ run('8-bit'): the UNWEIGHTED mean of the RGB channels."""
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float64).mean(axis=2)
    else:
        a = a.astype(np.float64)
        if a.max() > 255:                      # 16-bit -> reescala como o ImageJ
            a = (a - a.min()) / max(1e-9, (a.max() - a.min())) * 255.0
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def enhance_contrast(img8, saturated):
    """ImageJ ContrastEnhancer.stretchHistogram + normalize."""
    hist = np.bincount(img8.ravel(), minlength=256).astype(np.float64)
    n = img8.size
    thr = n * saturated / 200.0
    c = 0.0; hmin = 0
    for i in range(256):
        c += hist[i]
        if c > thr:
            hmin = i; break
    c = 0.0; hmax = 255
    for i in range(255, -1, -1):
        c += hist[i]
        if c > thr:
            hmax = i; break
    if hmax <= hmin:
        return img8.copy()
    out = (img8.astype(np.float64) - hmin) * (255.0 / (hmax - hmin))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def disk_kernel(radius):
    """kernel circular do ImageJ RankFilters: dx^2+dy^2 <= r^2+1."""
    r = int(np.ceil(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((x * x + y * y) <= radius * radius + 1).astype(np.float64)
    return k / k.sum()


def variance_filter(img8, radius):
    """Variance do ImageJ; saida 8-bit (clampada 0-255)."""
    k = disk_kernel(radius)
    f = img8.astype(np.float64)
    m = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.filter2D(f * f, -1, k, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(m2 - m * m, 0.0)
    return np.clip(np.round(var), 0, 255).astype(np.uint8)


def segment(path, radius, threshold, saturated):
    """retorna (area_px, area_pct, mask_do_maior_componente, shape)"""
    img8 = to_8bit(path)
    H, W = img8.shape
    enh = enhance_contrast(img8, saturated)
    var = variance_filter(enh, radius)
    mask = (var <= threshold).astype(np.uint8)            # setThreshold(0, T)
    # Fill Holes (fills holes not connected to the border)
    from scipy.ndimage import binary_fill_holes
    mask = binary_fill_holes(mask.astype(bool)).astype(np.uint8)
    # Analyze Particles: componentes 8-conectados, area >= MIN_AREA, maior
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_i, best_a = 0, 0
    for i in range(1, nlab):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= MIN_AREA and a > best_a:
            best_a, best_i = a, i
    if best_i == 0:
        return 0, 0.0, np.zeros_like(mask), (H, W)
    return int(best_a), 100.0 * best_a / (W * H), (lab == best_i).astype(np.uint8), (H, W)


# ---------------- selection of the 12 images ----------------
qc = list(csv.DictReader(open("data/whst_pass1_qc.csv", encoding="utf-8-sig")))
hu = [r for r in qc if r["cell_line"] == "HUVEC"]
base = {}
for r in hu:
    if int(r["timepoint_h"]) == 0:
        base.setdefault(r["series_key"], []).append(float(r["area_pct"]))
basemax = {k: max(v) for k, v in base.items()}


def lote_of(u):
    return u.split("||")[0]


falhas = []
for r in hu:
    if not int(r["needs_correction"]):
        continue
    tp = int(r["timepoint_h"]); a = float(r["area_pct"])
    ratio = a / basemax[r["series_key"]] if (tp > 0 and r["series_key"] in basemax and basemax[r["series_key"]] > 0) else None
    implaus0 = (tp == 0 and (a < 3 or a > 85))
    if (ratio and ratio > 1.5) or implaus0:
        falhas.append((abs((ratio or 99)), r, ratio, implaus0))
# piores: maior razao / 0h implausivel, diversificando lote
falhas.sort(key=lambda t: -(t[2] or 99))
sel_f, seen = [], set()
for _, r, ratio, imp in falhas:
    L = lote_of(r["analysis_unit"])
    if len(sel_f) < 6 and (L not in seen or len(sel_f) >= 4):
        sel_f.append((r, f"FALHA r={ratio:.2f}" if ratio else "FALHA 0h implausivel"))
        seen.add(L)
oks = [r for r in hu if r["categoria"] == "OK"]
sel_o, seeno = [], set()
for r in sorted(oks, key=lambda r: r["analysis_unit"]):
    L = lote_of(r["analysis_unit"])
    if len(sel_o) < 6 and L not in seeno:
        sel_o.append((r, "OK")); seeno.add(L)
for r in oks:                                            # completa se faltar
    if len(sel_o) < 6 and all(r["whst_input_file"] != s[0]["whst_input_file"] for s in sel_o):
        sel_o.append((r, "OK"))
SEL = sel_f + sel_o
print(f"selecionadas: {len(sel_f)} falhas + {len(sel_o)} OK")

# ---------------- sweep ----------------
rows = []
masks = {}
print("\nrunning 12 images × 4 parameter sets...")
for r, tag in SEL:
    p = os.path.join(OUT, r["whst_input_file"])
    for name, rad, thr, sat in PARAMS:
        apx, apct, mk, shape = segment(p, rad, thr, sat)
        masks[(r["whst_input_file"], name)] = mk
        rows.append({"grupo": tag, "analysis_unit": r["analysis_unit"], "tp": int(r["timepoint_h"]),
                     "arquivo": r["whst_input_file"], "param": name,
                     "radius": rad, "threshold": thr, "sat": sat,
                     "area_pct": round(apct, 3),
                     "area_pct_imagej_P0": float(r["area_pct"]),
                     "categoria_v2": r["categoria"]})
    print(f"  ok {r['whst_input_file'][:44]}")

with open("stage4/whst_param_sweep.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------------- VALIDACAO do P0 ----------------
print("\n=== VALIDACAO: minha reimplementacao P0 vs ImageJ ===")
d = [(x["arquivo"], x["area_pct"], x["area_pct_imagej_P0"]) for x in rows if x["param"] == "P0"]
difs = [abs(a - b) for _, a, b in d]
rel = [abs(a - b) / max(b, 1e-9) * 100 for _, a, b in d]
print(f"  n={len(d)}  dif absoluta media={np.mean(difs):.2f} pp  mediana={np.median(difs):.2f} pp")
print(f"  dif relativa mediana={np.median(rel):.1f}%")
for fn, a, b in d[:12]:
    print(f"    {fn[:40]:<40} py={a:>7.2f}  imagej={b:>7.2f}  d={a-b:+.2f}")
FIEL = np.median(rel) < 15
print(f"  -> reimplementation {'FAITHFUL (continue)' if FIEL else 'DIVERGENT (sweep not trustworthy!)'}")

# ---------------- tabela por imagem ----------------
print("\n=== area_pct per image x set ===")
print(f"{'grupo':<24}{'tp':>4}{'P0':>9}{'P1':>9}{'P2':>9}{'P3':>9}   unidade")
by = defaultdict(dict)
for x in rows:
    by[x["arquivo"]][x["param"]] = x["area_pct"]
for r, tag in SEL:
    fn = r["whst_input_file"]; v = by[fn]
    print(f"{tag:<24}{r['timepoint_h']:>4}{v['P0']:>9.2f}{v['P1']:>9.2f}{v['P2']:>9.2f}{v['P3']:>9.2f}   "
          f"{r['analysis_unit'][:34]}")

# ---------------- paineis visuais ----------------
def contour_overlay(path, mk, title):
    img8 = to_8bit(path)
    rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    cont, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cont, -1, (0, 255, 255), 14)
    im = Image.fromarray(rgb); im.thumbnail((300, 300))
    tile = Image.new("RGB", (im.width, im.height + 18), (18, 18, 18))
    tile.paste(im, (0, 18))
    ImageDraw.Draw(tile).text((3, 3), title, fill=(255, 235, 120))
    return tile

for grupo, sel, fname in (("FALHAS", sel_f, "sweep_falhas.png"), ("OK", sel_o, "sweep_ok.png")):
    tiles = []
    for r, tag in sel:
        p = os.path.join(OUT, r["whst_input_file"])
        for name, *_ in PARAMS:
            a = by[r["whst_input_file"]][name]
            tiles.append(contour_overlay(p, masks[(r["whst_input_file"], name)],
                                         f"{name} {a:.1f}%  tp{r['timepoint_h']}h"))
    cw = max(t.width for t in tiles) + 4; ch = max(t.height for t in tiles) + 4
    pan = Image.new("RGB", (4 * cw, len(sel) * ch), (30, 30, 30))
    for i, t in enumerate(tiles):
        pan.paste(t, ((i % 4) * cw + 2, (i // 4) * ch + 2))
    pan.save(fname)
    print(f"painel {grupo}: {fname}")

print("\nSalvo: stage4/whst_param_sweep.csv, sweep_falhas.png, sweep_ok.png")
