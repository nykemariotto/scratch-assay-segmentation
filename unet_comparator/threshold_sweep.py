# -*- coding: utf-8 -*-
"""
threshold_sweep.py — the U-Net across the threshold, to compare against YOLO at the
SAME operating point.

THE PROBLEM. The YOLO arm was evaluated at `conf = 0.80`, the default of the
published interface; the U-Net has no confidence, only `sigmoid > 0.50`. Comparing
0.80 with 0.50 compares two arbitrary choices, not two architectures. The native
regimes are qualitatively distinct:

    YOLO M @0.80    recall  87.5%   precision 100.0%  (never invents, stays silent on 12%)
    U-Net  @0.50    recall  99.8%   precision  97.1%  (almost never misses, but invents on 21% of the empties)

WHAT THIS SCRIPT DOES. A single forward pass per image; from the probability map it
applies every threshold in the grid. The sweep therefore costs one GPU pass, not one
per threshold.

For each (run, threshold) it records, per image:
  · area_px at the original resolution (undoes the letterbox, as stage3/predict_areas.py does)
  · IoU against the reference mask, in 640x640 space — the same convention as
    `avalia()` in train_unet.py
  · whether the mask came out empty

RECALL/PRECISION DEFINITION. At the IMAGE level, which is what applies to both arms:
the U-Net does not produce scored instances, so the detection recall of Table 2 does
not apply to it.
  recall    = images with an annotated wound where a non-empty mask came out
  precision = images with a non-empty mask that do have a wound

Output: `stage3/unet_sweep.csv` (per image) and `stage3/unet_sweep_summary.csv`
(per run and threshold).

    python unet_comparator/threshold_sweep.py
"""
import argparse
import csv
import glob
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
os.chdir(os.path.dirname(AQUI))

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402

from unet_data import WoundDataset, carrega_splits, desfaz_letterbox   # noqa: E402
from unet_model import UNet                                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUNS = os.path.join("runs", "segment", "unet_comparator")
LIMIARES = [round(x, 3) for x in
            (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
             0.90, 0.95, 0.98, 0.99, 0.995, 0.999)]


def grade_ocupada():
    """does not run alongside a training job: 234 images × 5 runs would contend for the GPU."""
    import time
    for p in glob.glob(os.path.join(RUNS, "*", "results.csv")):
        d = os.path.dirname(p)
        if not os.path.isfile(os.path.join(d, "COMPLETED.json")) and \
           time.time() - os.path.getmtime(p) < 900:
            return os.path.basename(d)
    return None


@torch.no_grad()
def varre(pesos, dl, dev):
    ck = torch.load(pesos, map_location=dev, weights_only=False)
    m = UNet().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    linhas = []
    for x, y, nomes, lbs in dl:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        prob = torch.sigmoid(m(x).float())            # ONE forward per image
        for i, nome in enumerate(nomes):
            gt = y[i, 0] > 0.5
            gt_soma = float(gt.sum())
            for t in LIMIARES:
                p = prob[i, 0] > t
                inter = float((p & gt).sum())
                uni = float((p | gt).sum())
                # IoU in 640x640 space, same convention as avalia()
                iou = 1.0 if uni == 0 else inter / uni
                # area at the original resolution, the same convention as stage3/predict_areas.py
                mm = desfaz_letterbox(p.byte().cpu().numpy(),
                                      {k: (v[i].item() if torch.is_tensor(v) else v[i])
                                       for k, v in lbs.items()})
                linhas.append({"arquivo": nome, "limiar": t,
                               "area_px": int(mm.sum()),
                               "iou": round(iou, 6),
                               "vazia": int(p.sum() == 0),
                               "tem_ferida": int(gt_soma > 0)})
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ocup = grade_ocupada()
    if ocup and not args.force:
        sys.exit(f"ABORTED: the grid looks active ({ocup}). Wait, or pass --force.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    splits = carrega_splits("data.yaml")
    ds = WoundDataset(splits[args.split], 640, treino=False, fill=0)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2,
                    pin_memory=(dev == "cuda"))

    alvos = sorted(d for d in glob.glob(os.path.join(RUNS, "*"))
                   if os.path.isfile(os.path.join(d, "COMPLETED.json")))
    if not alvos:
        sys.exit("no finished U-Net run")
    print(f"{len(alvos)} run(s) · {len(ds)} images from {args.split} · "
          f"{len(LIMIARES)} limiares · device {dev}\n")

    todas = []
    for i, d in enumerate(alvos, 1):
        nome = os.path.basename(d)
        import time
        t = time.time()
        for L in varre(os.path.join(d, "best.pt"), dl, dev):
            L["run"] = nome
            todas.append(L)
        print(f"  [{i}/{len(alvos)}] {nome}  ({time.time()-t:.0f}s)")

    os.makedirs("stage3", exist_ok=True)
    p1 = os.path.join("stage3", "unet_sweep.csv")
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "limiar", "arquivo", "area_px",
                                          "iou", "vazia", "tem_ferida"])
        w.writeheader()
        for L in todas:
            w.writerow({k: L[k] for k in w.fieldnames})

    # summary per (run, threshold) — recall and precision at the IMAGE level
    resumo = []
    for nome in sorted({L["run"] for L in todas}):
        for t in LIMIARES:
            sub = [L for L in todas if L["run"] == nome and L["limiar"] == t]
            pos = [L for L in sub if L["tem_ferida"]]
            neg = [L for L in sub if not L["tem_ferida"]]
            tp = sum(1 for L in pos if not L["vazia"])
            fp = sum(1 for L in neg if not L["vazia"])
            fn = len(pos) - tp
            rec = tp / len(pos) if pos else float("nan")
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            ious = [L["iou"] for L in pos]
            resumo.append({"run": nome, "limiar": t, "n_pos": len(pos),
                           "n_neg": len(neg), "tp": tp, "fp": fp, "fn": fn,
                           "recall": round(rec, 6), "precisao": round(prec, 6),
                           "iou_medio_pos": round(float(np.mean(ious)), 6)})
    p2 = os.path.join("stage3", "unet_sweep_summary.csv")
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumo[0].keys()))
        w.writeheader()
        w.writerows(resumo)

    print(f"\nwrote {p1} and {p2}")
    print("\nmean over the 5 seeds, per threshold (image level):")
    print(f"  {'thresh':>7} {'recall':>8} {'precision':>9} {'IoU pos':>8} "
          f"{'FN':>5} {'FP':>5}")
    for t in LIMIARES:
        sub = [r for r in resumo if r["limiar"] == t]
        print(f"  {t:>7.3f} {np.mean([r['recall'] for r in sub]):>8.4f} "
              f"{np.mean([r['precisao'] for r in sub]):>9.4f} "
              f"{np.mean([r['iou_medio_pos'] for r in sub]):>8.4f} "
              f"{np.mean([r['fn'] for r in sub]):>5.1f} "
              f"{np.mean([r['fp'] for r in sub]):>5.1f}")
    print("""
To match the operating point: find the threshold whose `recall` equals that of the YOLO
reference at the image level (M @ conf 0.80 = 0.875) and compare precision and IoU
there. The native regimes (0.50 for the U-Net, 0.80 for YOLO) are still reported —
matching the point answers "which is better at equal sensitivity", it does not replace
the description of how each operates by default.""")


if __name__ == "__main__":
    main()
