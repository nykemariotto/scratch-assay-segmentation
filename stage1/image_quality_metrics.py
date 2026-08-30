# -*- coding: utf-8 -*-
"""
stage1/image_quality_metrics.py — Tarefa 1: metricas objetivas de qualidade de imagem
for the 1363 images of the dataset (train+val+test). It excludes nothing.

Metricas por imagem:
  1. otsu_eta  : bimodality of the VARIANCE MAP (the same thing WHST thresholds).
                 eta = sigma2_entre_classes(Otsu) / sigma2_total  em [0,1].
                 alto = bimodal (monocamada vs ferida discernivel)
                 baixo = unimodal (sem ferida no campo, ou tudo borrado)
  2. lap_var   : nitidez = variancia do Laplaciano
  3. vignette  : mean intensity of the 4 corners / centre  (<1 = dark corners)

COMPARABILITY: the 118 640x640 images are a centre crop (a different physical scale
from the native 2452x2056). All are resized to a longest side of 1024
e o raio do filtro e escalado (20 * 1024/2452 ~ 8), tornando a vizinhanca fisica
comparavel entre os dois grupos.
"""
import csv, os
import numpy as np
import cv2

LONG = 1024
RADIUS = 8            # 20 * 1024/2452
MAP = "data/mapping_dataset_final_strat.csv"

rows = [r for r in csv.DictReader(open(MAP, encoding="utf-8")) if r["partition"] != "EXCLUIDA"]
print(f"images to process: {len(rows)}")

k = None
def disk(radius):
    r = int(np.ceil(radius)); y, x = np.mgrid[-r:r+1, -r:r+1]
    kk = ((x*x + y*y) <= radius*radius + 1).astype(np.float64)
    return kk / kk.sum()
k = disk(RADIUS)


def metrics(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    H, W = im.shape
    s = LONG / max(H, W)
    im = cv2.resize(im, (max(1, int(round(W*s))), max(1, int(round(H*s)))),
                    interpolation=cv2.INTER_AREA)
    f = im.astype(np.float64)

    # ---- mapa de variancia (como o WHST) ----
    m = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.filter2D(f*f, -1, k, borderType=cv2.BORDER_REFLECT)
    var = np.clip(np.round(np.maximum(m2 - m*m, 0)), 0, 255).astype(np.uint8)

    # ---- Otsu no mapa de variancia: eta = sigma2_B / sigma2_T ----
    hist = np.bincount(var.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    idx = np.arange(256)
    mu_T = (p * idx).sum()
    sigma2_T = (p * (idx - mu_T) ** 2).sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_T * w - mu) ** 2 / (w * (1 - w))
    sb = np.nan_to_num(sb, nan=0.0, posinf=0.0, neginf=0.0)
    sigma2_B = sb.max()
    eta = float(sigma2_B / sigma2_T) if sigma2_T > 1e-9 else 0.0

    # ---- nitidez ----
    lap_var = float(cv2.Laplacian(im, cv2.CV_64F).var())

    # ---- vinhetagem: 4 cantos vs centro ----
    h, wd = im.shape
    ch, cw = int(h*0.15), int(wd*0.15)
    corners = np.concatenate([f[:ch, :cw].ravel(), f[:ch, -cw:].ravel(),
                              f[-ch:, :cw].ravel(), f[-ch:, -cw:].ravel()])
    c0, c1 = int(h*0.35), int(h*0.65)
    d0, d1 = int(wd*0.35), int(wd*0.65)
    center = f[c0:c1, d0:d1]
    vign = float(corners.mean() / max(center.mean(), 1e-9))

    return eta, lap_var, vign, sigma2_T, (H, W)


out = []
for i, r in enumerate(rows, 1):
    p = os.path.join("dataset", "images", r["partition"], r["arquivo_b"])
    mt = metrics(p)
    if mt is None:
        print("  falhou ler:", p); continue
    eta, lv, vg, s2t, (H, W) = mt
    out.append({"arquivo_b": r["arquivo_b"], "partition": r["partition"],
                "cell_line": r["linha_celular"], "timepoint_h": r["timepoint_h"],
                "group_key": r["group_key"], "lote": r["lote"],
                "res": "640" if (W, H) == (640, 640) else "nativo",
                "otsu_eta": round(eta, 5), "lap_var": round(lv, 2),
                "vignette": round(vg, 4), "var_map_sigma2": round(s2t, 2)})
    if i % 200 == 0:
        print(f"  {i}/{len(rows)}")

with open("data/image_quality_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print(f"\nSaved: data/image_quality_metrics.csv ({len(out)} rows)")

# ---- distribuicoes ----
import statistics as st
def pct(v, q): return float(np.percentile(v, q))
eta = np.array([x["otsu_eta"] for x in out])
lap = np.array([x["lap_var"] for x in out])
vig = np.array([x["vignette"] for x in out])
print("\n=== DISTRIBUTIONS (1363 images) ===")
for nm, v in (("otsu_eta", eta), ("lap_var", lap), ("vignette", vig)):
    print(f"  {nm:<10} p5={pct(v,5):>9.4f}  p10={pct(v,10):>9.4f}  p25={pct(v,25):>9.4f}  "
          f"mediana={pct(v,50):>9.4f}  p75={pct(v,75):>9.4f}  p95={pct(v,95):>9.4f}")

print("\n=== otsu_eta by partition ===")
for p in ("train", "val", "test"):
    v = np.array([x["otsu_eta"] for x in out if x["partition"] == p])
    print(f"  {p:<6} n={len(v):>4}  mediana={np.median(v):.4f}  p10={np.percentile(v,10):.4f}")

print("\n=== otsu_eta by resolution (comparability check) ===")
for rr in ("nativo", "640"):
    v = np.array([x["otsu_eta"] for x in out if x["res"] == rr])
    if len(v):
        print(f"  {rr:<7} n={len(v):>4}  mediana={np.median(v):.4f}  p10={np.percentile(v,10):.4f}")
