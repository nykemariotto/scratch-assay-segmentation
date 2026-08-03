# -*- coding: utf-8 -*-
"""
predict_unet.py — predicted areas on the test set, in the format the benchmark consumes.

The delicate point is the AREA. The network predicts on a 640x640 letterbox; the area
the paper reports is a fraction of the ORIGINAL image. Predicting and counting pixels
in letterbox space would give a number inflated by the padding and distorted by the
rescaling — and area_pct is exactly the quantity that enters the closure fraction.
So the mask is returned to the original space (`desfaz_letterbox`) BEFORE measuring.

Output: unet_test_areas.csv with one row per test-set image
       (file, area_px, area_pct, n_components), ready for the merge with
       stage3/benchmark_classico_longo.csv.

DO NOT RUN WHILE THE YOLO GRID IS OCCUPYING THE GPU.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from unet_data import WoundDataset, carrega_splits, desfaz_letterbox
from unet_model import UNet


def colar(lote):
    """collate that keeps the letterbox dicts instead of trying to stack them."""
    xs = torch.stack([b[0] for b in lote])
    ys = torch.stack([b[1] for b in lote])
    return xs, ys, [b[2] for b in lote], [b[3] for b in lote]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best.pt of a U-Net run")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limiar", type=float, default=0.5)
    ap.add_argument("--min-area-px", type=int, default=0,
                    help="drop components smaller than this (0 = keep everything)")
    ap.add_argument("--out", default="unet_test_areas.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.weights, map_location=dev, weights_only=False)
    fill = 0 if ck.get("padding", "black") == "black" else 255
    modelo = UNet().to(dev)
    modelo.load_state_dict(ck["model"])
    modelo.eval()
    print(f"weights: {args.weights} · seed {ck.get('seed')} · epoch {ck.get('epoch')} · device {dev}")

    pasta = carrega_splits(args.data)[args.split]
    ds = WoundDataset(pasta, args.imgsz, treino=False, fill=fill)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=colar)
    print(f"{args.split}: {len(ds)} images")

    linhas = []
    with torch.no_grad():
        for x, _, nomes, lbs in dl:
            p = (torch.sigmoid(modelo(x.to(dev))) > args.limiar).byte().cpu().numpy()[:, 0]
            for i, nome in enumerate(nomes):
                m = desfaz_letterbox(p[i], lbs[i])
                if args.min_area_px > 0:
                    n, rot, stats, _ = cv2.connectedComponentsWithStats(m, 8)
                    keep = np.zeros_like(m)
                    for c in range(1, n):
                        if stats[c, cv2.CC_STAT_AREA] >= args.min_area_px:
                            keep[rot == c] = 1
                    m = keep
                n_comp = cv2.connectedComponentsWithStats(m, 8)[0] - 1
                area = int(m.sum())
                total = lbs[i]["orig_h"] * lbs[i]["orig_w"]
                linhas.append({"arquivo": nome, "area_px": area,
                               "area_pct": round(100.0 * area / total, 6),
                               "n_componentes": n_comp,
                               "orig_w": lbs[i]["orig_w"], "orig_h": lbs[i]["orig_h"]})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    vazias = sum(1 for r in linhas if r["area_px"] == 0)
    print(f"wrote: {args.out} ({len(linhas)} rows · {vazias} empty predictions)")


if __name__ == "__main__":
    main()
