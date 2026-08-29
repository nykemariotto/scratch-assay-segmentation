# -*- coding: utf-8 -*-
"""
run_unet_grid.py — the 5 U-Net seeds, same mechanics as stage2/run_grid.py.

Each run executes as an isolated process (RAM is returned between runs), and the
COMPLETED.json sentinel plus a cross-check against results.csv decides what has
already finished — best.pt alone is not enough, because it is written continuously.

  python unet_comparator/run_unet_grid.py --dry-run
  python unet_comparator/run_unet_grid.py

GPU GUARD: it refuses to start if the YOLO grid is still running, so as not to
compete for the GPU in the middle of a two-hour training run. Override with
--force.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

PY = sys.executable
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 100
SAVE_ROOT = os.path.join(RAIZ, "runs", "segment", "unet_comparator")
YOLO_ROOT = os.path.join(RAIZ, "runs", "segment", "runs_revision")


def done(run, epochs=EPOCHS):
    d = os.path.join(SAVE_ROOT, run)
    s = os.path.join(d, "COMPLETED.json")
    if not os.path.exists(s):
        return False
    try:
        c = json.load(open(s, encoding="utf-8"))
    except Exception:
        return False
    if c.get("status") != "ok":
        return False
    r = os.path.join(d, "results.csv")
    if not os.path.exists(r):
        return False
    with open(r, encoding="utf-8") as f:
        n = max(0, sum(1 for _ in csv.reader(f)) - 1)
    return n == c.get("completed_epochs", -1) and n >= epochs


def grade_yolo_rodando(tol_seg=900):
    """Heurística: algum results.csv da grade YOLO escrito nos últimos 15 min."""
    for r in glob.glob(os.path.join(YOLO_ROOT, "*", "results.csv")):
        d = os.path.dirname(r)
        if os.path.exists(os.path.join(d, "COMPLETED.json")):
            continue
        if time.time() - os.path.getmtime(r) < tol_seg:
            return os.path.basename(d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--force", action="store_true", help="ignora a guarda de GPU")
    args = ap.parse_args()

    runs = [f"unet_black_seed{s}" for s in SEEDS]
    pend = [r for r in runs if not done(r, args.epochs)]
    print(f"U-Net: {len(runs)} runs | completed: {len(runs)-len(pend)} | "
          f"to run: {len(pend)} | epochs={args.epochs}")
    for r in runs:
        print(f"   {'[OK]' if done(r, args.epochs) else '[  ]'} {r}")

    ocupada = grade_yolo_rodando()
    if ocupada and not args.dry_run and not args.force:
        sys.exit(f"\nABORTED: the YOLO grid looks active ({ocupada} wrote recently).\n"
                 f"Wait for it to finish, or pass --force if you know it has stopped.")
    if args.dry_run:
        if ocupada:
            print(f"\n[guard] YOLO grid active: {ocupada}")
        return

    log = []
    t0 = time.time()
    for i, r in enumerate(pend, 1):
        seed = int(r.split("seed")[1])
        print(f"\n===== [{i}/{len(pend)}] {r} =====", flush=True)
        t = time.time()
        rc = subprocess.run([PY, os.path.join(AQUI, "train_unet.py"),
                             "--seed", str(seed), "--epochs", str(args.epochs),
                             "--project", SAVE_ROOT, "--name", r],
                            cwd=RAIZ).returncode
        ok = rc == 0 and done(r, args.epochs)
        log.append({"run": r, "returncode": rc, "ok": ok, "sec": round(time.time() - t, 1)})
        json.dump(log, open(os.path.join(RAIZ, "stage3/unet_grid_log.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"----- {r}: {'OK' if ok else 'FAILED rc=%d' % rc}")

        # STOPS AT THE FIRST FAILURE. The grid used to move on to the next seed,
        # which only makes sense if the failures are independent, and they are not:
        # the fp16 overflow that killed seed42 is deterministic and would hit all
        # five. Continuing would cost ~3 h per seed to produce more invalid runs.
        if not ok:
            falhou = os.path.join(SAVE_ROOT, r, "FAILED.json")
            detalhe = ""
            if os.path.isfile(falhou):
                try:
                    detalhe = "\n" + open(falhou, encoding="utf-8").read()
                except Exception:
                    pass
            sys.exit(f"\n===== GRID INTERRUPTED at {r} (rc={rc}).\n"
                     f"{len(log)-1}/{len(pend)} runs completed before this failure.\n"
                     f"Diagnostique antes de reiniciar — os runs ja prontos sao "
                     f"preservados e a grade retoma de onde parou.{detalhe}")
    print(f"\n===== U-NET FINISHED: {sum(1 for x in log if x['ok'])}/{len(log)} "
          f"em {(time.time()-t0)/3600:.1f} h")


if __name__ == "__main__":
    main()
