# -*- coding: utf-8 -*-
"""
stage3/varredura_conf.py — what the confidence slider actually does, measured.

WHY THIS EXISTS. The deployed interface lets the user move the detection confidence
between 20% and 100%, and every number the paper reports was measured at a single
point on that range, 0.80. The U-Net arm has the equivalent knob characterised end to
end by unet_comparator/varredura_limiar.py; the YOLO arm had nothing. So the tool
ships a control whose effect on this task was never measured, while its comparator's
was. That asymmetry is a fair thing for a reviewer to ask about, and this closes it.

THIS DESCRIBES. IT DOES NOT RE-CHOOSE.
0.80 is the operating point at which the reported precision and recall were measured
and it stays. Picking a different default now, after seeing the sweep, would be
selecting a cutpoint from the data — the exact practice the analysis plan rules out
(Altman 1994; Royston 2006). The output here belongs in the Discussion as a
description of the control, not in the Methods as a new setting.

TWO QUESTIONS IT ANSWERS

  1. The ordinary one: how recall, precision and IoU move with the threshold, at the
     image level, which is the level that applies to both arms.

  2. The one specific to this assay, and the reason the sweep is worth running at
     all: WHETHER THE LOSS IS SIZE-DEPENDENT. The endpoint is the closure fraction
     (a0 - at)/a0. A constant multiplicative bias cancels in that ratio; a threshold
     does not act that way. If raising it preferentially drops small, faint wounds,
     those are the late timepoints — and an undetected wound is an area of zero,
     which reads as a closure of 1.0, a false complete closure. Recall is therefore
     also reported stratified by wound size, over bands fixed here in the source
     before the sweep is run.

     THIS IS NOT HYPOTHETICAL. It was observed while building Figure 1, on the SKOV-3
     series P1|CT|F8: at 24 h the reference standard annotates an open wound and the
     model at conf 0.80 returns nothing at all. Zero area over the baseline computes
     to "100.0% closure", so the panel would have asserted a complete closure over a
     wound that is plainly open. One frame is an anecdote, not a rate — which is
     exactly what this sweep exists to supply.

ONE FORWARD PASS PER IMAGE. Predicting at the lowest threshold and filtering the
detections afterwards by their own scores gives the same result as re-predicting at
each threshold, because Ultralytics applies the confidence cut before NMS and NMS
only ever suppresses the lower-scoring box of an overlapping pair — so no detection
that survives at a high threshold can be removed by running at a low one. That is an
argument, not a measurement, so --verifica re-predicts directly at two thresholds and
asserts the masks match.

    python stage3/varredura_conf.py
    python stage3/varredura_conf.py --todas-configs
    python stage3/varredura_conf.py --verifica
"""
import argparse
import csv
import glob
import os
import statistics as st
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

from unet_data import rasteriza, rotulo_de          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

YOLO_ROOT = os.path.join("runs", "segment", "runs_revision")
# the reference configuration of the single-variable design
CFG_REF = "yolo11m-seg_black_coco"
# the range the interface exposes, walked in steps of 0.05. 0.80 is in the grid
# because it is the reported point and has to be readable straight off the table.
LIMIARES = [round(x, 2) for x in np.arange(0.20, 1.00, 0.05)]
CONF_PISO = min(LIMIARES)
PONTO_REPORTADO = 0.80
# wound size as a percentage of the field, fixed HERE, before the sweep runs, so the
# bands cannot be drawn around whatever the result turns out to be
FAIXAS = [(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 100.0)]


def faixa_de(pct):
    for lo, hi in FAIXAS:
        if lo <= pct < hi:
            return f"{lo:g}-{hi:g}%"
    return f"{FAIXAS[-1][0]:g}-{FAIXAS[-1][1]:g}%"


def gt_original(f):
    """Reference mask at the acquired resolution, not at 640x640.

    Same choice as stage3/iou_por_imagem.py, and for the same reason: Ultralytics
    letterboxes one way and unet_data another, so comparing in either letterbox space
    would fold a padding convention into the number. The acquired resolution has no
    convention to reconcile.
    """
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    return rasteriza(rotulo_de(f), w, h) > 0, w, h


def uniao(md, idx, hw):
    """Union of the selected masks, at the image's own size."""
    h, w = hw
    if not len(idx):
        return np.zeros((h, w), bool)
    u = np.any(md[idx], axis=0)
    if u.shape != (h, w):
        u = cv2.resize(u.astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return u


def iou_de(pred, gt):
    inter = int(np.logical_and(pred, gt).sum())
    uni = int(np.logical_or(pred, gt).sum())
    return 1.0 if uni == 0 else inter / uni


def varre_run(pesos, nome, imagens, gts, dims, imgsz, padding, verifica=False):
    import padding_patch
    padding_patch.apply(padding)
    from ultralytics import YOLO
    m = YOLO(pesos)
    linhas = []
    for f, gt, (w, h) in zip(imagens, gts, dims):
        r = m.predict(f, conf=CONF_PISO, imgsz=imgsz, retina_masks=True,
                      verbose=False)[0]
        if r.masks is None or len(r.masks) == 0:
            md = np.zeros((0, h, w), bool)
            escores = np.zeros(0)
        else:
            md = r.masks.data.cpu().numpy() > 0.5
            escores = r.boxes.conf.cpu().numpy()
        gt_pct = 100.0 * float(gt.sum()) / (w * h)
        for t in LIMIARES:
            sel = np.where(escores >= t)[0]
            p = uniao(md, sel, (h, w))
            linhas.append({"run": nome, "conf": t,
                           "arquivo": os.path.basename(f),
                           "area_px": int(p.sum()),
                           "area_pct": round(100.0 * float(p.sum()) / (w * h), 6),
                           "iou": round(iou_de(p, gt), 6),
                           "vazia": int(not p.any()),
                           "tem_ferida": int(bool(gt.any())),
                           "gt_area_pct": round(gt_pct, 6),
                           "faixa_gt": faixa_de(gt_pct),
                           "n_deteccoes": int(len(sel))})
        if verifica:
            for t in (PONTO_REPORTADO, 0.50):
                d = m.predict(f, conf=t, imgsz=imgsz, retina_masks=True,
                              verbose=False)[0]
                direto = (np.zeros((h, w), bool) if d.masks is None or not len(d.masks)
                          else uniao(d.masks.data.cpu().numpy() > 0.5,
                                     np.arange(len(d.masks)), (h, w)))
                posthoc = uniao(md, np.where(escores >= t)[0], (h, w))
                if not np.array_equal(direto, posthoc):
                    sys.exit(f"EQUIVALENCE BROKEN at conf={t} on {f}: predicting "
                             f"directly differs from filtering afterwards. The "
                             f"one-pass shortcut is invalid; re-predict per threshold.")
    del m
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--todas-configs", action="store_true",
                    help="all 25 runs; by default only the five seeds of the "
                         "reference configuration")
    ap.add_argument("--verifica", action="store_true",
                    help="re-predict directly at 0.80 and 0.50 and assert the one-pass "
                         "filtering matches. Slow; run it once.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from eval_test import grade_ocupada, padding_do_run
    ocup = grade_ocupada()
    if ocup and not args.force:
        sys.exit(f"ABORTED: the grid looks active ({ocup}). Wait, or pass --force.")

    from unet_data import carrega_splits
    d = carrega_splits("data.yaml")["test"]
    imagens = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                     for f in glob.glob(os.path.join(d, e)))
    print(f"test: {len(imagens)} images · {len(LIMIARES)} thresholds "
          f"{LIMIARES[0]:.2f}–{LIMIARES[-1]:.2f}")

    print("rasterising the reference standard at the acquired resolution…")
    gts, dims = [], []
    for f in imagens:
        g, w, h = gt_original(f)
        gts.append(g)
        dims.append((w, h))
    n_pos = sum(1 for g in gts if g.any())
    print(f"  {n_pos} with a wound · {len(gts) - n_pos} negatives")
    dist = {}
    for g, (w, h) in zip(gts, dims):
        if g.any():
            dist[faixa_de(100.0 * float(g.sum()) / (w * h))] = \
                dist.get(faixa_de(100.0 * float(g.sum()) / (w * h)), 0) + 1
    print("  by wound size: " + " · ".join(f"{k} n={v}" for k, v in sorted(dist.items())))

    alvo = "*" if args.todas_configs else f"{CFG_REF}_seed*"
    runs = sorted(glob.glob(os.path.join(YOLO_ROOT, alvo)))
    runs = [r for r in runs if os.path.isdir(r)]
    print(f"\nruns: {len(runs)}\n")

    todas = []
    for i, dr in enumerate(runs, 1):
        nome = os.path.basename(dr)
        p = os.path.join(dr, "weights", "best.pt")
        if not os.path.isfile(p):
            print(f"  [SKIPPED] {nome}: no best.pt")
            continue
        t0 = time.time()
        todas += varre_run(p, nome, imagens, gts, dims, args.imgsz,
                           padding_do_run(dr), verifica=args.verifica)
        print(f"  [{i}/{len(runs)}] {nome}  ({time.time() - t0:.0f}s)")

    if not todas:
        sys.exit("nothing swept")

    p1 = os.path.join("stage3", "yolo_varredura_conf.csv")
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)

    # ── summary per (run, threshold), at the IMAGE level ─────────────────────
    resumo = []
    for nome in sorted({L["run"] for L in todas}):
        for t in LIMIARES:
            sub = [L for L in todas if L["run"] == nome and L["conf"] == t]
            pos = [L for L in sub if L["tem_ferida"]]
            neg = [L for L in sub if not L["tem_ferida"]]
            tp = sum(1 for L in pos if not L["vazia"])
            fp = sum(1 for L in neg if not L["vazia"])
            linha = {"run": nome, "conf": t,
                     "recall": round(tp / len(pos), 6) if pos else float("nan"),
                     "precisao": round(tp / (tp + fp), 6) if (tp + fp) else float("nan"),
                     "iou_medio_pos": round(st.mean(L["iou"] for L in pos), 6) if pos else float("nan"),
                     "fn": len(pos) - tp, "fp": fp}
            for lo, hi in FAIXAS:
                k = f"{lo:g}-{hi:g}%"
                f_ = [L for L in pos if L["faixa_gt"] == k]
                linha[f"recall_{k}"] = (round(sum(1 for L in f_ if not L["vazia"]) / len(f_), 6)
                                        if f_ else float("nan"))
            resumo.append(linha)

    p2 = os.path.join("stage3", "yolo_varredura_conf_resumo.csv")
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumo[0].keys()))
        w.writeheader()
        w.writerows(resumo)

    print(f"\nwrote {p1} and {p2}")

    faixas_rot = [f"{lo:g}-{hi:g}%" for lo, hi in FAIXAS]
    print("\nmean over the seeds, per threshold (image level):")
    print(f"  {'conf':>6} {'recall':>8} {'precision':>10} {'IoU pos':>8} {'FN':>5}"
          + "".join(f"{('rec ' + r):>13}" for r in faixas_rot))
    for t in LIMIARES:
        sub = [r for r in resumo if r["conf"] == t]
        cel = []
        for r_ in faixas_rot:
            v = [x[f"recall_{r_}"] for x in sub if x[f"recall_{r_}"] == x[f"recall_{r_}"]]
            cel.append(f"{st.mean(v):>13.4f}" if v else f"{'—':>13}")
        marca = "  <- reported" if abs(t - PONTO_REPORTADO) < 1e-9 else ""
        print(f"  {t:>6.2f} {st.mean(r['recall'] for r in sub):>8.4f} "
              f"{st.mean(r['precisao'] for r in sub):>10.4f} "
              f"{st.mean(r['iou_medio_pos'] for r in sub):>8.4f} "
              f"{st.mean(r['fn'] for r in sub):>5.1f}" + "".join(cel) + marca)

    print(f"""
HOW TO READ THIS. The `rec` columns are recall within a band of wound size. If they
fall together as the threshold rises, the control costs sensitivity evenly and the
closure fraction is not distorted. If the small-wound band falls faster, the loss is
size-dependent: small wounds are the late timepoints, an undetected wound is an area
of zero, and a zero reads as complete closure. That is the failure this sweep exists
to detect, and it is a directional bias in the endpoint rather than added noise.

{PONTO_REPORTADO:.2f} is the point every reported figure was measured at, and it stays.
This table describes the control the interface exposes; it is not a search for a
better default. A threshold chosen from this table would be a cutpoint chosen from
the data.""")


if __name__ == "__main__":
    main()
