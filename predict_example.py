# -*- coding: utf-8 -*-
"""
predict_example.py — runs the pipeline over the example set, downloading
nothing, so the step-by-step guide is executable rather than described.

Two modes, chosen automatically:

  A) If .pt weights are present (Zenodo, DOI 10.5281/zenodo.20298129): runs the
     deep-learning segmentation and measures the wound fraction.

  B) If NO weights are present: runs the CLASSICAL segmentation (Wound Healing
     Size Tool, frozen P0 parameters — radius=20, threshold=100,
     saturated=0.001), reimplemented in Python and validated against ImageJ
     (median relative difference 0.1%). Useful both to try the pipeline out and
     to compare the classical method against the deep-learning one.

Either way: writes a contour overlay and prints the measured area.
Output: examples/output/
"""
import csv, os, sys
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
EX = "examples"
OUT = os.path.join(EX, "output")
RAD, THR, SAT, MIN_AREA = 20, 100, 0.001, 100


# ----------------------------------------------------------------- classical
def to8(path):
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., :3].astype(np.float64).mean(axis=2)
    else:
        a = a.astype(np.float64)
        if a.max() > 255:
            a = (a - a.min()) / max(1e-9, a.max() - a.min()) * 255
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def enhance(img, sat):
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    thr = img.size * sat / 200.0
    c = 0.0; lo = 0
    for i in range(256):
        c += h[i]
        if c > thr:
            lo = i; break
    c = 0.0; hi = 255
    for i in range(255, -1, -1):
        c += h[i]
        if c > thr:
            hi = i; break
    if hi <= lo:
        return img.copy()
    return np.clip(np.round((img.astype(np.float64) - lo) * (255.0 / (hi - lo))), 0, 255).astype(np.uint8)


def segment_classical(path):
    """WHST P0 — the same logic as stage4/whst_batch.ijm."""
    from scipy.ndimage import binary_fill_holes
    img = to8(path)
    H, W = img.shape
    e = enhance(img, SAT)
    r = int(np.ceil(RAD)); y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((x * x + y * y) <= RAD * RAD + 1).astype(np.float64); k /= k.sum()
    f = e.astype(np.float64)
    m = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.filter2D(f * f, -1, k, borderType=cv2.BORDER_REFLECT)
    var = np.clip(np.round(np.maximum(m2 - m * m, 0)), 0, 255).astype(np.uint8)
    mask = binary_fill_holes(var <= THR).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bi, ba = 0, 0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a >= MIN_AREA and a > ba:
            ba, bi = a, i
    mk = (lab == bi).astype(np.uint8) if bi else np.zeros_like(mask)
    return img, mk, 100.0 * ba / (W * H)


def save_overlay(img, mk, dst, colour=(0, 255, 255)):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if mk.any():
        cont, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cont, -1, colour, max(3, img.shape[1] // 200))
    Image.fromarray(rgb).save(dst)


# ------------------------------------------------------------------ main
manp = os.path.join(EX, "MANIFEST.csv")
if not os.path.isfile(manp):
    sys.exit("example set missing — run: python stage1/prepare_examples.py")
man = list(csv.DictReader(open(manp, encoding="utf-8-sig")))
os.makedirs(OUT, exist_ok=True)

# THIS PROJECT's weights. Exclude the pretrained checkpoints Ultralytics
# downloads on its own (yolo11s-seg.pt, yolo11m-seg.pt, ...): they are COCO, they
# have no notion of "wound", and using them here would return 0% everywhere — a
# meaningless result that would look like a bug.
import re as _re
BASE_ULTRA = _re.compile(r"^yolo\d+[nsmlx]?-?(seg|cls|pose|obb)?\.pt$", _re.I)
weights = [os.path.join(r, f) for r in ("models", ".") if os.path.isdir(r)
         for f in sorted(os.listdir(r))
         if f.endswith(".pt") and not BASE_ULTRA.match(f)]

if weights:
    print(f"=== MODE A · deep learning ({os.path.basename(weights[0])}) ===\n")
    from ultralytics import YOLO
    model = YOLO(weights[0])
    for r in man:
        p = os.path.join(EX, "images", r["arquivo"])
        res = model.predict(p, conf=0.8, imgsz=640, verbose=False)[0]
        img = to8(p)
        H, W = img.shape
        mk = np.zeros((H, W), np.uint8)
        if res.masks is not None:
            for m in res.masks.data.cpu().numpy():
                mk |= (cv2.resize(m, (W, H)) > 0.5).astype(np.uint8)
        pct = 100.0 * mk.sum() / (W * H)
        save_overlay(img, mk, os.path.join(OUT, f"dl_{r['arquivo']}"), (60, 230, 60))
        print(f"  {r['papel']:<38} area={pct:>6.2f}%  (annotated: {r['n_instancias']} inst)")
else:
    print("=== MODE B · classical segmentation (WHST P0) ===")
    print("    No .pt weights found. For the deep-learning mode, download from")
    print("    https://doi.org/10.5281/zenodo.20298129 and place them in models/.\n")
    for r in man:
        p = os.path.join(EX, "images", r["arquivo"])
        img, mk, pct = segment_classical(p)
        save_overlay(img, mk, os.path.join(OUT, f"whst_{r['arquivo']}"))
        print(f"  {r['papel']:<38} area={pct:>6.2f}%  (annotated: {r['n_instancias']} inst)")
    print("\n  NOTE: the classical method over-segments in 55% of the valid test-set")
    print("  images (see PROTOCOLO_CORRECAO_MANUAL.md §10) — if the area above looks")
    print("  too large on a negative, that is exactly the reported effect.")

print(f"\noverlays in {OUT}/")
