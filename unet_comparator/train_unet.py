# -*- coding: utf-8 -*-
"""
train_unet.py — trains ONE U-Net with an explicit seed, mirroring
stage2/train_config.py.

The same contract as the YOLO grid: same partition, 640x640, black padding, 100
epochs, batch 4, no early stopping, forced determinism, and the same provenance
artefacts (`provenance.json`, `pip_freeze.txt`, `COMPLETED.json`) so that the
downstream treats both arms alike.

Usage:
    python unet_comparator/train_unet.py --seed 42
    python unet_comparator/run_unet_grid.py          # all 5 seeds

DO NOT RUN WHILE THE YOLO GRID IS OCCUPYING THE GPU.
"""
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from unet_data import WoundDataset, carrega_splits
from unet_model import UNet, BCEDiceLoss


def set_deterministic(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def avalia(modelo, loader, dev, limiar=0.5):
    """Mean per-image IoU and Dice. An image with no wound (a negative) counts 1.0 if the
    prediction is empty too — the convention stage4/correction_agreement.py uses."""
    modelo.eval()
    ious, dices = [], []
    for x, y, _, _ in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        p = (torch.sigmoid(modelo(x)) > limiar).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        uni = ((p + y) > 0).float().sum(dim=(1, 2, 3))
        soma = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        for i in range(x.shape[0]):
            ious.append(1.0 if uni[i] == 0 else (inter[i] / uni[i]).item())
            dices.append(1.0 if soma[i] == 0 else (2 * inter[i] / soma[i]).item())
    return float(np.mean(ious)), float(np.mean(dices))


def diagnostica_csv(rcsv, epochs_esperadas, min_seq=3):
    """Reads results.csv and decides whether the run is valid.

    The old sentinel only COUNTED rows. That is how seed42 came out with
    `"status": "ok"` while carrying 60 epochs of train_loss = NaN — 100 rows
    present, 100 epochs declared. Counting rows is not checking results.

    This function lives SEPARATE from the training loop on purpose: a checker
    that can only be exercised by running 3 h of GPU is not testable, and an
    untested checker is exactly what let the padding mismatch and the fp16
    overflow through. See test_sentinela.py, which runs it against the real CSV
    of the destroyed run.

    Returns a diagnostic dict; the `_ruins` and `_degeneradas` keys carry the
    epoch lists and are consumed by the caller.
    """
    with open(rcsv, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    ruins = [int(r["epoch"]) for r in linhas
             if not all(math.isfinite(_f(r[c]))
                        for c in ("train_loss", "val_iou", "val_dice"))]

    # IoU == Dice is the signature of a degenerate prediction (all-zeros or all-ones):
    # Dice = 2*IoU/(1+IoU), so they only coincide at 0 or 1. But the columns are MEANS
    # over images, and two means can coincide by chance — with 6 decimals in the CSV
    # that is a ~1e-6 chance per epoch. Failing a whole run over one coincidence would
    # be a false positive; we require `min_seq` CONSECUTIVE epochs, which is what the
    # real failure produces (on seed42 there were 60 in a row).
    marca = [math.isfinite(_f(r["val_iou"]))
             and abs(_f(r["val_iou"]) - _f(r["val_dice"])) < 1e-9
             and 0.0 < _f(r["val_iou"]) < 1.0 for r in linhas]
    degeneradas, seq = set(), []
    for r, m in zip(linhas, marca):
        seq = seq + [int(r["epoch"])] if m else []
        if len(seq) >= min_seq:
            degeneradas |= set(seq)
    degeneradas = sorted(degeneradas)

    return {"requested_epochs": epochs_esperadas,
            "completed_epochs": len(linhas),
            "stopped_early": len(linhas) < epochs_esperadas,
            "nan_epochs": len(ruins),
            "primeira_epoca_nao_finita": ruins[0] if ruins else None,
            "epocas_iou_igual_dice": len(degeneradas),
            "final_val_iou": round(_f(linhas[-1]["val_iou"]), 6) if linhas else None,
            "_ruins": ruins, "_degeneradas": degeneradas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--padding", default="black", choices=["black", "white"])
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--project", default=os.path.join("runs", "segment", "unet_comparator"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    fill = 0 if args.padding == "black" else 255
    run = args.name or f"unet_{args.padding}_seed{args.seed}"
    outdir = os.path.join(args.project, run)
    os.makedirs(outdir, exist_ok=True)

    set_deterministic(args.seed)
    splits = carrega_splits(args.data)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds_tr = WoundDataset(splits["train"], args.imgsz, treino=True, fill=fill, seed=args.seed)
    ds_va = WoundDataset(splits["val"], args.imgsz, treino=False, fill=fill)
    # The shuffle generator uses a FIXED seed, not the training seed.
    # This mirrors Ultralytics: there, data order does not vary with the seed.
    # If it varied here, the U-Net arm would carry one more source of variance than
    # the YOLO arm, and the Table 2 standard deviations would not be comparable.
    from unet_data import SEED_AUG
    g = torch.Generator()
    g.manual_seed(SEED_AUG)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=(dev == "cuda"), drop_last=False, generator=g)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=(dev == "cuda"))
    print(f"{run} · train {len(ds_tr)} · val {len(ds_va)} · device {dev}")

    modelo = UNet().to(dev)
    crit = BCEDiceLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=5e-4)
    # WARMUP — the YOLO arm has `warmup_epochs: 3.0` (it is in the args.yaml of every
    # run in the grid); the comparator had none. Beyond the design asymmetry, it was the
    # likely source of the instability: CosineAnnealingLR started at lr=1e-3 on the
    # very first epoch, over random weights, with batch 4. Three runs collapsed within
    # the first ~20 epochs (seed43 six times and survived; seed44 three times and died
    # with NaN at epoch 13) — which seeds finish became a lottery, and reporting only
    # the survivors would be survivorship bias.
    #
    # The warmup is per ITERATION, not per epoch: at 233 batches per epoch, 3 epochs
    # give 699 ramp steps, which is smooth. Per epoch it would be 3 steps.
    WARMUP_EPOCAS = 3
    # the cosine only starts counting after the warmup, and starts from args.lr
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - WARMUP_EPOCAS))
    escala = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    rcsv = os.path.join(outdir, "results.csv")
    with open(rcsv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_iou", "val_dice", "lr", "seconds"])

    passos_warmup = WARMUP_EPOCAS * len(dl_tr)
    passo = 0
    print(f"warmup: {WARMUP_EPOCAS} epochs = {passos_warmup} iterations, "
          f"from {args.lr/100:.2e} to {args.lr:.2e}", flush=True)

    melhor, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        ds_tr.set_epoca(ep)          # MANDATORY: without this the augmentation freezes
        modelo.train()
        te, perdas = time.time(), []
        for i_lote, (x, y, _, _) in enumerate(dl_tr):
            # linear lr ramp during the warmup, applied PER ITERATION
            if passo < passos_warmup:
                fator = 0.01 + 0.99 * (passo + 1) / passos_warmup
                for gp in opt.param_groups:
                    gp["lr"] = args.lr * fator
            passo += 1
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                perda = crit(modelo(x), y)

            # ABORT on the first NaN/Inf — skipping the batch and carrying on does
            # not help. The GradScaler protects opt.step(), but not the BatchNorm
            # buffers: running_mean/running_var are updated in the FORWARD pass. The
            # moment one activation turns inf/NaN the buffers absorb NaN and the model
            # is dead, even though the weights stay finite. That is what happened to
            # seed42: the 100 epochs "finished", 60 of them with the model already gone.
            v = float(perda.detach())
            if not math.isfinite(v):
                falha = {"status": "diverged", "run": run, "epoch": ep,
                         "batch": i_lote, "loss": str(v),
                         "epochs_completed_before": ep - 1,
                         "reason": "non-finite loss; training aborted. BatchNorm "
                                   "buffers are poisoned in the forward pass and do "
                                   "not recover — continuing would produce a dead run "
                                   "that would still report 100 epochs."}
                json.dump(falha, open(os.path.join(outdir, "FAILED.json"), "w",
                                      encoding="utf-8"), indent=2, ensure_ascii=False)
                sys.exit(f"\nABORTED: loss {v} at epoch {ep}, batch {i_lote}. "
                         f"FAILED.json written to {outdir}")

            escala.scale(perda).backward()
            # unscale_ before clipping: the clipping needs the gradient at its real
            # scale, otherwise max_norm means nothing.
            escala.unscale_(opt)
            # max_norm=10.0 is the value Ultralytics uses in
            # BaseTrainer.optimizer_step (v8.4.102, engine/trainer.py). Copying the
            # number, rather than picking one, keeps the comparator aligned with the
            # YOLO arm — a different max_norm would be an asymmetry to defend at
            # review time for no gain.
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=10.0)
            escala.step(opt)
            escala.update()
            perdas.append(v)
        # the cosine only acts after the warmup; during it, the lr is the ramp above
        if ep > WARMUP_EPOCAS:
            sched.step()

        # RELEASE THE ALLOCATOR CACHE BEFORE VALIDATING.
        # Without this, validation took 181-267 s against 14.5 s — 12.5x. Training runs
        # in fp16 under autocast and leaves ~7.0 GB of the 8.6 GB reserved in the cache;
        # validation runs in fp32, OUTSIDE autocast, and asks for twice the memory per
        # sample. The allocator had to hand blocks back to the driver and request others,
        # and every cudaMalloc/cudaFree serialises — hence the GPU reading 100%
        # utilisation while drawing only 53 W of 160 W: it was not compute, it was the
        # allocator. The result is BIT-FOR-BIT IDENTICAL (IoU 0.500330 with and without).
        # Measured in diag_validation2.py, which compares the four conditions.
        if dev == "cuda":
            torch.cuda.empty_cache()
        iou, dice = avalia(modelo, dl_va, dev)
        dt = time.time() - te
        with open(rcsv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, round(float(np.mean(perdas)), 6),
                                    round(iou, 6), round(dice, 6),
                                    opt.param_groups[0]["lr"], round(dt, 1)])
        if iou > melhor:
            melhor = iou
            torch.save({"model": modelo.state_dict(), "epoch": ep, "val_iou": iou,
                        "seed": args.seed, "padding": args.padding}, os.path.join(outdir, "best.pt"))
        print(f"  epoch {ep:>3}/{args.epochs}  loss {np.mean(perdas):.4f}  "
              f"IoU {iou:.4f}  Dice {dice:.4f}  ({dt:.0f}s)", flush=True)

    torch.save({"model": modelo.state_dict(), "epoch": args.epochs, "seed": args.seed,
                "padding": args.padding}, os.path.join(outdir, "last.pt"))
    total = time.time() - t0

    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=120).stdout
    except Exception:
        freeze = ""
    open(os.path.join(outdir, "pip_freeze.txt"), "w", encoding="utf-8").write(freeze)
    json.dump({"run": run, "model": "unet-canonical-base64", "padding": args.padding,
               "padding_fill_value": fill, "init": "scratch", "seed": args.seed,
               "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
               "lr": args.lr, "optimizer": "AdamW", "loss": "0.5*BCE + 0.5*Dice",
               "data": args.data, "wall_seconds": round(total, 1),
               "best_val_iou": round(melhor, 6),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "note": "architectural reimplementation (Ronneberger 2015) used as an "
                       "architectural comparator; NOT a run of the tool published by "
                       "Dogru et al."},
              open(os.path.join(outdir, "provenance.json"), "w", encoding="utf-8"), indent=2)

    # ── sentinel ─────────────────────────────────────────────────────────────
    diag = diagnostica_csv(rcsv, args.epochs)
    diag.update({"run": run, "best_val_iou": round(melhor, 6),
                 "wall_seconds": round(total, 1)})
    ruins, degeneradas = diag.pop("_ruins"), diag.pop("_degeneradas")
    feitas = diag["completed_epochs"]

    if ruins or degeneradas or feitas < args.epochs:
        diag["status"] = "diverged"
        diag["reason"] = ("non-finite epochs and/or val_iou == val_dice (degenerate "
                          "prediction). COMPLETED.json NOT written — the downstream "
                          "requires that sentinel and therefore ignores this run.")
        json.dump(diag, open(os.path.join(outdir, "FAILED.json"), "w",
                             encoding="utf-8"), indent=2, ensure_ascii=False)
        sys.exit(f"\nINVALID RUN: {run} — {len(ruins)} non-finite epochs, "
                 f"{len(degeneradas)} degenerate. FAILED.json written.")

    diag["status"] = "ok"
    json.dump(diag, open(os.path.join(outdir, "COMPLETED.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nDone: {run}  ({total/60:.1f} min)  best val IoU {melhor:.4f}")


if __name__ == "__main__":
    main()
