# -*- coding: utf-8 -*-
"""
stage2/train_config.py — trains one configuration with an explicit seed and
determinism.

Usage:
  python stage2/train_config.py --model yolo11m-seg --padding black --init coco --seed 42

Two corrections relative to the draft specification:
  * padding is applied through padding_patch (Ultralytics does not expose it in
    the config; 114 is hard-coded in Mosaic, RandomPerspective and LetterBox, and
    patching only one would contaminate the ablation). Verified at pixel level by
    stage2/verify_padding.py.
  * 'rtdetr-seg' does not exist in Ultralytics (detection only). The script
    refuses it explicitly rather than failing obscurely.
"""
import argparse, os, random, json, subprocess, sys, time


def set_deterministic(seed):
    import numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--padding", default="black", choices=["black", "white", "gray"])
    ap.add_argument("--init", default="coco", choices=["coco", "scratch"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--project", default="runs_revision")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers (2 reduz pico de RAM)")
    ap.add_argument("--name", default=None, help="sobrescreve o nome do run")
    args = ap.parse_args()

    if "rtdetr" in args.model.lower() and "seg" in args.model.lower():
        sys.exit("ABORTED: there is no segmentation RT-DETR in Ultralytics "
                 "(only rtdetr-l/x for detection).")

    set_deterministic(args.seed)

    # padding before importing/instantiating the data pipeline
    import padding_patch
    fill = padding_patch.apply(args.padding)

    from ultralytics import YOLO, RTDETR
    run = args.name or f"{args.model}_{args.padding}_{args.init}_seed{args.seed}"
    weights = f"{args.model}.pt" if args.init == "coco" else f"{args.model}.yaml"
    Model = RTDETR if args.model.lower().startswith("rtdetr") else YOLO
    model = Model(weights)

    # ------------------------------------------------------------------
    # PADDING EVIDENCE RECORDED FROM INSIDE THE REAL TRAINING RUN.
    #
    # On 2026-07-27 we lost 34 h because the padding patch held in the parent process
    # and not in the DataLoader workers (spawn on Windows). stage2/verify_padding.py
    # missed it because it built its OWN dataset in the main process — it measured the
    # right place in the wrong process.
    #
    # This callback reads the FIRST BATCH TRAINING ACTUALLY CONSUMES and records the
    # histogram of the extreme values. If the padding does not arrive, it is recorded
    # in the run itself, and the preflight canary catches it before another 34 h.
    # ------------------------------------------------------------------
    # WHERE TO INTERCEPT. The `on_train_batch_start` callback does not work: at that
    # point `trainer.batch` does not exist yet (BaseTrainer never assigns self.batch),
    # and the probe silently reads None — which is what happened in the first version
    # of this instrument. The right point is `preprocess_batch`, which RECEIVES the
    # batch coming from the workers, still uint8, before the division by 255.
    evidencia = {}

    from ultralytics.models.yolo.detect.train import DetectionTrainer
    if not hasattr(DetectionTrainer, "_orig_preprocess_batch"):
        DetectionTrainer._orig_preprocess_batch = DetectionTrainer.preprocess_batch

    def _pp(self, batch):
        if not evidencia:
            try:
                import numpy as _np
                a = batch["img"].detach().cpu().numpy()
                if a.dtype.kind == "f" and a.max() <= 1.0 + 1e-6:
                    a = a * 255.0
                a = _np.rint(a).astype(_np.int32)
                n = a.size
                evidencia.update({
                    "n_pixels": int(n),
                    "frac_0": round(float((a == 0).sum()) / n, 6),
                    "frac_114": round(float((a == 114).sum()) / n, 6),
                    "frac_255": round(float((a == 255).sum()) / n, 6),
                    "min": int(a.min()), "max": int(a.max()),
                    "padding_esperado": fill,
                    "capturado_em": "preprocess_batch (batch real, pre-normalizacao)",
                })
            except Exception as e:
                evidencia["erro"] = f"{type(e).__name__}: {e}"
        return DetectionTrainer._orig_preprocess_batch(self, batch)

    DetectionTrainer.preprocess_batch = _pp

    t0 = time.time()
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, name=run, project=args.project,
        exist_ok=True, workers=args.workers, val=True, plots=False, patience=args.patience,
    )
    dt = time.time() - t0

    # provenance record IN ULTRALYTICS' REAL DIRECTORY (save_dir),
    # not in args.project (Ultralytics prefixes runs/<task>/).
    outdir = str(getattr(model.trainer, "save_dir", os.path.join(args.project, run)))
    os.makedirs(outdir, exist_ok=True)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=120).stdout
    except Exception:
        freeze = ""
    import torch
    with open(os.path.join(outdir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run": run, "model": args.model, "padding": args.padding,
            "padding_fill_value": fill, "init": args.init, "seed": args.seed,
            "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
            "data": args.data, "wall_seconds": round(dt, 1),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "padding_patch": "Mosaic+RandomPerspective+LetterBox (114 -> %d)" % fill,
            "workers": args.workers,
            # measured on the first real training batch, not on a parallel dataloader
            "evidencia_padding_no_batch": evidencia,
        }, f, indent=2)

    # hard guard: if the real batch does not show the requested padding, the run is
    # SUSPECT. It does not abort (training has already finished), but it records the
    # alert where it will not slip by.
    aviso = None
    if evidencia and "erro" not in evidencia:
        f114 = evidencia.get("frac_114", 0.0)
        alvo = {0: "frac_0", 255: "frac_255", 114: "frac_114"}[fill]
        falvo = evidencia.get(alvo, 0.0)
        if fill != 114 and f114 > 0.5 * falvo:
            aviso = (f"SUSPECT: grey(114) in {100*f114:.2f}% of the pixels against "
                     f"{100*falvo:.2f}% of the requested padding ({fill}). The patch may "
                     f"not have reached the workers.")
        elif falvo < 0.005:
            aviso = (f"SUSPECT: the requested padding ({fill}) appears in only "
                     f"{100*falvo:.3f}% of the pixels of the first batch.")
    if aviso:
        print("\n" + "!" * 70 + f"\n{aviso}\n" + "!" * 70, flush=True)
        with open(os.path.join(outdir, "AVISO_PADDING.txt"), "w", encoding="utf-8") as f:
            f.write(aviso + "\n")
    with open(os.path.join(outdir, "pip_freeze.txt"), "w", encoding="utf-8") as f:
        f.write(freeze)

    # ---- completion sentinel (WRITTEN LAST, only after exit 0) ----
    # Counts the epochs actually finished from results.csv and records whether
    # there was a legitimate early stop. stage2/run_grid.py only skips a run that has
    # THIS file with status=ok — best.pt alone is not enough (Ultralytics writes
    # best.pt continuously, so a crash midway leaves best.pt present).
    import csv as _csv
    rcsv = os.path.join(outdir, "results.csv")
    completed_epochs = 0
    if os.path.exists(rcsv):
        with open(rcsv, encoding="utf-8") as f:
            completed_epochs = sum(1 for _ in _csv.reader(f)) - 1  # menos o header
    stopped_early = completed_epochs < args.epochs
    with open(os.path.join(outdir, "COMPLETED.json"), "w", encoding="utf-8") as f:
        json.dump({
            "status": "ok",
            "run": run,
            "requested_epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
            "patience": args.patience,
            "wall_seconds": round(dt, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nConcluido: {run}  ({dt/60:.1f} min)  "
          f"epocas={completed_epochs}/{args.epochs}"
          f"{'  (early-stop)' if stopped_early else ''}")


if __name__ == "__main__":
    main()
