
# -*- coding: utf-8 -*-
"""
stage3/latencia_inferencia.py — what a prediction costs, per configuration.

WHY THIS MATTERS. The five configurations are statistically indistinguishable in
mAP, and the U-Net differs from them in failure mode rather than in average quality.
When accuracy ties, the criterion for what to deploy becomes **latency** — and
offering the user a fast option with a declared loss of precision is a feature, not
a concession.

IT MEASURES ON CPU AND ON GPU, and the CPU is what decides: a free Hugging Face
Space runs on CPU. The GPU latency serves the computational-cost paragraph of the
manuscript; the CPU latency decides what goes live.

METHODOLOGY
  · 3 warm-up images discarded (they allocate buffers and settle autotuning);
  · the median over N images, not the mean — one OS pause ruins the mean;
  · the same conf, imgsz and padding as each run's training;
  · the U-Net enters through the same pipeline as predict_areas (letterbox + sigmoid).

TWO LIMITS TO DECLARE, both measured by re-running this script on the same machine
on a different day:

  · These are wall-clock medians on ONE machine, and they move with how loaded it
    is: the same configurations came out 2-20% faster on a second run. Ratios
    BETWEEN configurations are stable (S stayed at ~50% of M in both); the absolute
    milliseconds are not portable.

  · M, M-white and M-scratch are the SAME architecture — 22.4 M parameters, 45.2 MB
    of weights — differing only in the padding colour and the initialisation used to
    TRAIN them, neither of which touches the cost of a forward pass. Their true
    latency is identical, and the order they come out in is measurement noise. The
    second run reordered exactly those three and left every other position intact.
    Do not read a ranking among them.

    python stage3/latencia_inferencia.py --n 40
"""
import argparse
import csv
import glob
import json
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
import torch                                        # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# one representative seed per configuration — the five seeds are architecturally
# equivalent, and latency does not depend on the seed
ALVOS = [
    ("S",         "runs/segment/runs_revision/yolo11s-seg_black_coco_seed42", "yolo", 10.1),
    ("M",         "runs/segment/runs_revision/yolo11m-seg_black_coco_seed42", "yolo", 22.4),
    ("X",         "runs/segment/runs_revision/yolo11x-seg_black_coco_seed42", "yolo", 62.1),
    ("M-white",   "runs/segment/runs_revision/yolo11m-seg_white_coco_seed42", "yolo", 22.4),
    ("M-scratch", "runs/segment/runs_revision/yolo11m-seg_black_scratch_seed42", "yolo", 22.4),
    ("U-Net",     "runs/segment/unet_comparator/unet_black_seed42", "unet", 31.0),
]


def pesos_de(d):
    for p in (os.path.join(d, "weights", "best.pt"), os.path.join(d, "best.pt")):
        if os.path.isfile(p):
            return p
    return None


def mede_yolo(pesos, imgs, dev, conf, imgsz, padding, aquec=3):
    import padding_patch
    padding_patch.apply(padding)
    from ultralytics import YOLO
    m = YOLO(pesos)
    m.to(dev)
    ts = []
    for i, f in enumerate(imgs):
        t = time.perf_counter()
        m.predict(f, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False,
                  device=dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= aquec:
            ts.append(time.perf_counter() - t)
    del m
    return ts


def mede_unet(pesos, imgs, dev, imgsz, aquec=3):
    from unet_data import desfaz_letterbox, letterbox
    from unet_model import UNet
    ck = torch.load(pesos, map_location=dev, weights_only=False)
    fill = 0 if ck.get("padding", "black") == "black" else 255
    m = UNet().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    ts = []
    with torch.no_grad():
        for i, f in enumerate(imgs):
            t = time.perf_counter()
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            lb, _, par = letterbox(img, np.zeros((h, w), np.uint8), imgsz, fill)
            x = torch.from_numpy(lb.transpose(2, 0, 1).copy()).float().div_(255)
            p = (torch.sigmoid(m(x.unsqueeze(0).to(dev))) > 0.5).byte().cpu().numpy()[0, 0]
            desfaz_letterbox(p, par)
            if dev == "cuda":
                torch.cuda.synchronize()
            if i >= aquec:
                ts.append(time.perf_counter() - t)
    del m
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from unet_data import carrega_splits
    d = carrega_splits("data.yaml")["test"]
    imgs = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                  for f in glob.glob(os.path.join(d, e)))[:args.n + 3]
    print(f"{len(imgs)-3} images measured (+3 warm-up) · conf={args.conf} · "
          f"imgsz={args.imgsz}")
    print(f"CPU: {os.cpu_count()} cores · "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'absent'}\n")

    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    linhas = []
    for dev in devs:
        print(f"── {dev.upper()} ──")
        print(f"  {'config':<11} {'params':>8} {'median':>10} {'p90':>9} "
              f"{'img/s':>8} {'tam. peso':>10}")
        for rot, dd, tipo, par in ALVOS:
            p = pesos_de(dd)
            if not p:
                print(f"  {rot:<11} [no best.pt]")
                continue
            from eval_test import padding_do_run
            pad = padding_do_run(dd) if tipo == "yolo" else "black"
            ts = (mede_yolo(p, imgs, dev, args.conf, args.imgsz, pad) if tipo == "yolo"
                  else mede_unet(p, imgs, dev, args.imgsz))
            med = st.median(ts)
            p90 = sorted(ts)[int(0.9 * len(ts))]
            mb = os.path.getsize(p) / 1e6
            print(f"  {rot:<11} {par:>6.1f} M {1000*med:>8.0f} ms {1000*p90:>7.0f} ms "
                  f"{1/med:>8.2f} {mb:>8.1f} MB")
            linhas.append({"config": rot, "device": dev, "params_M": par,
                           "mediana_ms": round(1000 * med, 2),
                           "p90_ms": round(1000 * p90, 2),
                           "img_por_s": round(1 / med, 3),
                           "peso_MB": round(mb, 1), "n": len(ts)})
            if dev == "cuda":
                torch.cuda.empty_cache()
        print()

    dest = os.path.join("stage3", "latencia_inferencia.csv")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"wrote {dest}")

    cpu = {l["config"]: l for l in linhas if l["device"] == "cpu"}
    # The ratio that answers "which of the five do we deploy" has to be computed over
    # the five ALONE. Taking the min and the max over every CPU row put the U-Net at
    # the top and printed a spread of ~9x as though it were the spread among the
    # detectors, in the same paragraph that calls the U-Net a separate case. The
    # honest number is ~4x, S against X.
    yolos = {k: v for k, v in cpu.items() if k != "U-Net"}
    if len(yolos) > 1:
        rap = min(yolos.values(), key=lambda l: l["mediana_ms"])
        len_ = max(yolos.values(), key=lambda l: l["mediana_ms"])
        print(f"""
READING. On CPU — which is where a free Space runs — the fastest of the detectors is
{rap['config']} ({rap['mediana_ms']:.0f} ms) and the slowest {len_['config']}
({len_['mediana_ms']:.0f} ms), a ratio of {len_['mediana_ms']/rap['mediana_ms']:.1f}x.

Since the five YOLO configurations are indistinguishable in mAP, that ratio is the
only objective criterion left between them — bearing in mind that M, M-white and
M-scratch share an architecture, so only S, M and X are genuinely different points.""")
        u = cpu.get("U-Net")
        if u:
            print(f"""The U-Net is OUTSIDE that ratio, on purpose: at {u['mediana_ms']:.0f} ms it is \
{u['mediana_ms']/rap['mediana_ms']:.1f}x the
fastest detector, but it differs in operating mode, so the choice between it and the
detector is not one of speed.""")


if __name__ == "__main__":
    main()
