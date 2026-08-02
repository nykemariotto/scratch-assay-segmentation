# -*- coding: utf-8 -*-
"""
stage4/quality_triangulate.py — tasks 2 and 3.
T2: the overlap between 3 independent signals (monotonic series, missing or
    fragmentary annotation, the lower tail of bimodality) + enrichment over chance.
T3: metrics of the images to be inspected visually (3 suspects + 2 controls), on
    the SAME scale as all 1363, to calibrate the cut.
It EXCLUDES nothing.
"""
import csv, os, json
from collections import defaultdict
import numpy as np
import cv2

Q = {r["arquivo_b"]: r for r in csv.DictReader(open("data/image_quality_metrics.csv", encoding="utf-8-sig"))}
qc = list(csv.DictReader(open("data/whst_pass1_qc.csv", encoding="utf-8-sig")))
corr = {r["whst_input_file"]: r for r in csv.DictReader(open("whst_input/correspondencia.csv", encoding="utf-8-sig"))}

# ---------- (a) images from the 14 series with monotonic growth ----------
ser = defaultdict(list)
for r in qc:
    ser[r["series_key"]].append((int(r["timepoint_h"]), float(r["area_pct"]), r))
def longest_inc(v):
    b = c = 1
    for i in range(1, len(v)):
        if v[i] > v[i-1]: c += 1; b = max(b, c)
        else: c = 1
    return b
A = set()
mono_series = []
for sk, v in ser.items():
    bytp = {}
    for tp, a, r in sorted(v): bytp[tp] = max(bytp.get(tp, -1), a)
    tps = sorted(bytp)
    if len(tps) >= 3 and longest_inc([bytp[t] for t in tps]) >= 3:
        mono_series.append(sk)
        for tp, a, r in v:
            ti = r["test_image"]
            if ti in Q: A.add(ti)

# ---------- (b) missing annotation (150) or fragment (25) ----------
B = set()
if os.path.exists("data/qc_suspeitas.csv"):
    for r in csv.DictReader(open("data/qc_suspeitas.csv", encoding="utf-8-sig")):
        if r["file"] in Q: B.add(r["file"])

# ---------- (c) lower tail of bimodality ----------
eta = {k: float(v["otsu_eta"]) for k, v in Q.items()}
vals = np.array(list(eta.values()))
UNIV = set(Q)

def tail(pctl):
    thr = np.percentile(vals, pctl)
    return {k for k, v in eta.items() if v <= thr}, thr

print(f"universe: {len(UNIV)} images")
print(f"(a) monotonic series: {len(mono_series)} series -> {len(A)} images (test only)")
print(f"(b) missing/fragmentary annotation: {len(B)} images")

def rep(X, Y, nx, ny, univ):
    inter = len(X & Y)
    exp = len(X) * len(Y) / max(len(univ), 1)
    enr = inter / exp if exp > 0 else float("nan")
    print(f"  {nx:<28} n {ny:<28} obs={inter:>4}  expected={exp:>6.1f}  enrichment={enr:>5.2f}x")

for pctl in (5, 10, 20):
    C, thr = tail(pctl)
    print(f"\n=== (c) lower tail of otsu_eta: p{pctl} (eta <= {thr:.4f}) -> {len(C)} images ===")
    rep(B, C, "(b) annot. missing/frag", "(c) low bimodality", UNIV)
    # (a) is test-only -> the test universe
    TEST = {k for k, v in Q.items() if v["partition"] == "test"}
    rep(A, C & TEST, "(a) monotonic series", "(c) low bimod (test)", TEST)
    rep(A, B & TEST, "(a) monotonic series", "(b) annot. missing/frag (test)", TEST)
    print(f"  triple (a&b&c, in test): {len(A & B & C & TEST)}")

# ---------- fraction of the TRAINING set in the tail ----------
print("\n=== FRACTION OF THE TRAINING SET IN THE BAD TAIL (a delicate decision) ===")
tr = {k for k, v in Q.items() if v["partition"] == "train"}
for pctl in (5, 10, 20):
    C, thr = tail(pctl)
    n = len(tr & C)
    print(f"  eta <= p{pctl} ({thr:.4f}): {n}/{len(tr)} of the training set = {100*n/len(tr):.1f}%")
lap = {k: float(v["lap_var"]) for k, v in Q.items()}
for pctl in (5, 10):
    thr = np.percentile(list(lap.values()), pctl)
    n = sum(1 for k in tr if lap[k] <= thr)
    print(f"  lap_var <= p{pctl} ({thr:.1f}): {n}/{len(tr)} of the training set = {100*n/len(tr):.1f}%")

# ---------- T3: metrics of the images to inspect ----------
LONG, RADIUS = 1024, 8
def disk(r):
    rr = int(np.ceil(r)); y, x = np.mgrid[-rr:rr+1, -rr:rr+1]
    k = ((x*x + y*y) <= r*r + 1).astype(np.float64); return k / k.sum()
K = disk(RADIUS)

def metrics_path(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    H, W = im.shape; s = LONG / max(H, W)
    im = cv2.resize(im, (int(round(W*s)), int(round(H*s))), interpolation=cv2.INTER_AREA)
    f = im.astype(np.float64)
    m = cv2.filter2D(f, -1, K, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.filter2D(f*f, -1, K, borderType=cv2.BORDER_REFLECT)
    var = np.clip(np.round(np.maximum(m2-m*m, 0)), 0, 255).astype(np.uint8)
    h = np.bincount(var.ravel(), minlength=256).astype(np.float64); p = h/h.sum()
    idx = np.arange(256); muT = (p*idx).sum(); s2T = (p*(idx-muT)**2).sum()
    w = np.cumsum(p); mu = np.cumsum(p*idx)
    den = w * (1 - w)
    valid = den > 1e-12                      # avoids dividing at the extremes (w=0 or 1)
    sb = np.zeros_like(den)
    sb[valid] = (muT*w[valid] - mu[valid])**2 / den[valid]
    sb = np.nan_to_num(sb, nan=0.0, posinf=0.0, neginf=0.0)
    eta_ = float(sb.max()/s2T) if s2T > 1e-9 else 0.0
    lv = float(cv2.Laplacian(im, cv2.CV_64F).var())
    hh, ww = im.shape; ch, cw = int(hh*.15), int(ww*.15)
    cor = np.concatenate([f[:ch,:cw].ravel(), f[:ch,-cw:].ravel(), f[-ch:,:cw].ravel(), f[-ch:,-cw:].ravel()])
    cen = f[int(hh*.35):int(hh*.65), int(ww*.35):int(ww*.65)]
    return eta_, lv, float(cor.mean()/max(cen.mean(),1e-9))

units = sorted({r["analysis_unit"] for r in qc})
def find(sub):
    for u in units:
        if all(s in u for s in sub): return u
INSPECT = [(find(["n1","PEP + NEB 5uM","D4"]), "SUSPEITA"), (find(["n2","PET + NEB 5uM","D3"]), "SUSPEITA"),
           (find(["n3","D4"]), "SUSPEITA"), (find(["n2","PEP + NEB 1uM","B1"]), "CTRL same batch"),
           (find(["originais","D1"]), "CTRL healthy")]
p10_eta = np.percentile(vals, 10); p10_lap = np.percentile(list(lap.values()), 10)
print(f"\n=== T3: METRICS OF THE IMAGES TO INSPECT (reference: p10 eta={p10_eta:.4f}, p10 lap={p10_lap:.1f}) ===")
print(f"{'unit':<44}{'tp':>4}{'c':>2}  {'eta':>7}{'lap_var':>9}{'vign':>7}")
t3 = []
for u, tag in INSPECT:
    if not u: continue
    rs = sorted([r for r in qc if r["analysis_unit"] == u], key=lambda r: (r["campo"], int(r["timepoint_h"])))
    print(f"  --- [{tag}] {u}")
    for r in rs:
        mt = metrics_path(os.path.join("whst_input", r["whst_input_file"]))
        if not mt: continue
        e, lv, vg = mt
        flag = " <-- low eta" if e <= p10_eta else ""
        print(f"    {'':<42}{r['timepoint_h']:>4}{r['campo']:>2}  {e:>7.4f}{lv:>9.1f}{vg:>7.3f}{flag}")
        t3.append({"unidade": u, "tag": tag, "tp": r["timepoint_h"], "campo": r["campo"],
                   "arquivo": r["whst_input_file"], "otsu_eta": round(e,5),
                   "lap_var": round(lv,2), "vignette": round(vg,4), "area_pct": r["area_pct"]})
with open("stage4/inspection_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(t3[0].keys())); w.writeheader(); w.writerows(t3)
print("\nSaved: stage4/inspection_metrics.csv")
