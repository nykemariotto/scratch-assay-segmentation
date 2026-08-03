# -*- coding: utf-8 -*-
"""
stage3/iou_por_imagem.py — per-image mask IoU, for YOLO and U-Net, on the SAME basis.

WHY. The U-Net threshold sweep (`unet_comparator/varredura_limiar.py`) showed that
at a threshold of 0.999 it reaches recall 0.873 and precision 1.000 — practically
identical to YOLO M at conf 0.80 (0.875 and 1.000). That is the matched operating
point. But the YOLO IoU was missing: the records from `stage3/eval_test.py` store
matching for mAP, and `stage3/predict_areas.py` stored area — neither stored mask IoU.

METHOD DECISION. Everything is measured at the ORIGINAL RESOLUTION (2452 x 2056),
with the reference standard rasterised right there by `unet_data.rasteriza`. Not at
640 x 640.

The reason is to avoid a silent trap: Ultralytics does its own letterbox, and
`unet_data.letterbox` does its own. If the two padding conventions differ by a pixel
or in rounding, the IoU of the two arms would be measured in slightly different
spaces — and the whole comparison would be biased by an implementation detail rather
than by a difference between models. At the original resolution there is no
convention to reconcile: it is the image space as acquired.

A consequence to declare: the IoU here is NOT numerically equal to the `iou` of the
sweep, which was computed at 640 x 640. This script recomputes both arms, so the
numbers being compared come from the same place.

    python stage3/iou_por_imagem.py
"""
import argparse
import csv
import glob
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "unet_comparator"))
sys.path.insert(0, os.path.join(RAIZ, "stage2"))   # padding_patch lives here
os.chdir(RAIZ)

import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

from unet_data import (carrega_splits, desfaz_letterbox, letterbox,   # noqa: E402
                       rasteriza, rotulo_de)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

YOLO_ROOT = os.path.join("runs", "segment", "runs_revision")
UNET_ROOT = os.path.join("runs", "segment", "unet_comparator")
# the REFERENCE configuration of the single-variable design
YOLO_CFG = "yolo11m-seg_black_coco"
LIMIARES_UNET = [0.50, 0.999]     # native, and the one that matches YOLO's recall


def iou_de(pred, gt):
    """IoU between two boolean masks. An image with no wound and no prediction = 1.0,
    the same convention as avalia() and the sweep."""
    inter = int(np.logical_and(pred, gt).sum())
    uni = int(np.logical_or(pred, gt).sum())
    return 1.0 if uni == 0 else inter / uni


def gt_original(f):
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    return rasteriza(rotulo_de(f), w, h) > 0, w, h


def yolo_masks(pesos, imagens, conf, imgsz, padding):
    import padding_patch
    padding_patch.apply(padding)
    from ultralytics import YOLO
    m = YOLO(pesos)
    out = []
    for f in imagens:
        r = m.predict(f, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]
        h, w = r.orig_shape
        if r.masks is None or len(r.masks) == 0:
            uni = np.zeros((h, w), bool)
        else:
            md = r.masks.data.cpu().numpy() > 0.5
            uni = np.any(md, axis=0)
            if uni.shape != (h, w):
                uni = cv2.resize(uni.astype(np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(uni)
    del m
    return out


def unet_masks(pesos, imagens, imgsz, limiares):
    from unet_model import UNet
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(pesos, map_location=dev, weights_only=False)
    fill = 0 if ck.get("padding", "black") == "black" else 255
    m = UNet().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    out = {t: [] for t in limiares}
    with torch.no_grad():
        for f in imagens:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            lb_img, _, lb = letterbox(img, np.zeros((h, w), np.uint8), imgsz, fill)
            x = torch.from_numpy(lb_img.transpose(2, 0, 1).copy()).float().div_(255)
            prob = torch.sigmoid(m(x.unsqueeze(0).to(dev)).float())[0, 0]
            for t in limiares:
                p = (prob > t).byte().cpu().numpy()
                out[t].append(desfaz_letterbox(p, lb) > 0)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from eval_test import grade_ocupada
    ocup = grade_ocupada()
    if ocup:
        sys.exit(f"ABORTED: the grid looks active ({ocup}).")

    d = carrega_splits("data.yaml")["test"]
    imagens = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                     for f in glob.glob(os.path.join(d, e)))
    print(f"test: {len(imagens)} images · YOLO conf={args.conf} · "
          f"U-Net limiares {LIMIARES_UNET}\n")

    print("rasterising the reference standard at the original resolution…")
    gts, dims = [], []
    for f in imagens:
        g, w, h = gt_original(f)
        gts.append(g)
        dims.append((w, h))
    n_pos = sum(1 for g in gts if g.any())
    print(f"  {n_pos} images with a wound · {len(gts)-n_pos} negatives\n")

    linhas = []
    yolos = sorted(glob.glob(os.path.join(YOLO_ROOT, f"{YOLO_CFG}_seed*")))
    unets = sorted(glob.glob(os.path.join(UNET_ROOT, "unet_black_seed*")))
    for d_ in yolos + unets:
        nome = os.path.basename(d_)
        p = os.path.join(d_, "weights", "best.pt")
        if not os.path.isfile(p):
            p = os.path.join(d_, "best.pt")
        if not os.path.isfile(p):
            print(f"  [SKIPPED] {nome}: no best.pt")
            continue
        t0 = time.time()
        if nome.startswith("unet"):
            for t, masks in unet_masks(p, imagens, args.imgsz, LIMIARES_UNET).items():
                for f, pred, gt in zip(imagens, masks, gts):
                    linhas.append({"run": nome, "braco": "U-Net", "ponto": f"sigmoid={t}",
                                   "arquivo": os.path.basename(f),
                                   "iou": round(iou_de(pred, gt), 6),
                                   "pred_vazia": int(not pred.any()),
                                   "tem_ferida": int(gt.any())})
        else:
            from eval_test import padding_do_run
            masks = yolo_masks(p, imagens, args.conf, args.imgsz, padding_do_run(d_))
            for f, pred, gt in zip(imagens, masks, gts):
                linhas.append({"run": nome, "braco": "YOLO M", "ponto": f"conf={args.conf}",
                               "arquivo": os.path.basename(f),
                               "iou": round(iou_de(pred, gt), 6),
                               "pred_vazia": int(not pred.any()),
                               "tem_ferida": int(gt.any())})
        print(f"  {nome}  ({time.time()-t0:.0f}s)")

    saida = os.path.join("stage3", "iou_por_imagem.csv")
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"\nwrote {saida}")

    # ─────────────────────────────── summary per arm and operating point
    import statistics as st
    print(f"\n{'arm':<8} {'point':<15} {'recall':>7} {'precision':>9} "
          f"{'IoU (pos)':>10} {'IoU (all)':>12}")
    print("-" * 66)
    chaves = []
    for L in linhas:
        k = (L["braco"], L["ponto"])
        if k not in chaves:
            chaves.append(k)
    for braco, ponto in chaves:
        por_run = {"rec": [], "prec": [], "iou_pos": [], "iou_all": []}
        for run in sorted({L["run"] for L in linhas
                           if L["braco"] == braco and L["ponto"] == ponto}):
            sub = [L for L in linhas if L["run"] == run and L["ponto"] == ponto
                   and L["braco"] == braco]
            pos = [L for L in sub if L["tem_ferida"]]
            neg = [L for L in sub if not L["tem_ferida"]]
            tp = sum(1 for L in pos if not L["pred_vazia"])
            fp = sum(1 for L in neg if not L["pred_vazia"])
            por_run["rec"].append(tp / len(pos))
            por_run["prec"].append(tp / (tp + fp) if (tp + fp) else float("nan"))
            por_run["iou_pos"].append(st.mean(L["iou"] for L in pos))
            por_run["iou_all"].append(st.mean(L["iou"] for L in sub))
        print(f"{braco:<8} {ponto:<15} {st.mean(por_run['rec']):>7.4f} "
              f"{st.mean(por_run['prec']):>9.4f} {st.mean(por_run['iou_pos']):>10.4f} "
              f"{st.mean(por_run['iou_all']):>12.4f}")
    print("""
IoU (pos) = mean over the images WITH an annotated wound
IoU (all) = includes the negatives, which score 1.0 when the prediction is empty too

Means over the 5 seeds of each arm. The `sigmoid=0.999` point is the one that matches
YOLO's recall; `sigmoid=0.5` is the U-Net's native regime.""")


if __name__ == "__main__":
    main()
