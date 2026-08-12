# -*- coding: utf-8 -*-
"""
stage2/run_grid.py — orchestrates the 25-run training grid of stage 2.

Each run executes as an ISOLATED PROCESS (subprocess), so RAM is reclaimed
between runs, avoiding the accumulation that crashed an earlier attempt. It
continues on failure: a run that crashes is recorded and the grid moves on, and
the failed ones can be re-executed afterwards (runs that already produced a
best.pt are skipped).

Usage:
  python stage2/run_grid.py                 # run everything still missing
  python stage2/run_grid.py --dry-run       # list the plan only
  python stage2/run_grid.py --only s        # filter by substring of the name
"""
import argparse, os, subprocess, sys, time, json, csv

PY = sys.executable
PROJECT = "runs_revision"
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 100
BATCH = 4
WORKERS = 2
SAVE_ROOT = os.path.join("runs", "segment", PROJECT)


def grid():
    cfgs = []
    for m in ("yolo11s-seg", "yolo11m-seg", "yolo11x-seg"):      # tamanho
        for s in SEEDS:
            cfgs.append((m, "black", "coco", s))
    for s in SEEDS:                                               # padding
        cfgs.append(("yolo11m-seg", "white", "coco", s))
    for s in SEEDS:                                              # init
        cfgs.append(("yolo11m-seg", "black", "scratch", s))
    return cfgs


def run_name(m, pad, init, seed):
    return f"{m}_{pad}_{init}_seed{seed}"


def done(name, expected_epochs=EPOCHS):
    """Robustly finished? best.pt alone is not enough (Ultralytics writes best.pt
    continuously, so a crash midway leaves best.pt present). It requires:
      1) sentinela COMPLETED.json com status=ok (escrito so apos exit 0);
      2) a cross-check: results.csv with the number of rows the sentinel declares;
      3) cobertura de epocas: completou o pedido OU parou por early-stop legitimo.
    Any inconsistency -> not finished -> it will be redone."""
    d = os.path.join(SAVE_ROOT, name)
    sentinel = os.path.join(d, "COMPLETED.json")
    if not os.path.exists(sentinel):
        return False
    try:
        c = json.load(open(sentinel, encoding="utf-8"))
    except Exception:
        return False
    if c.get("status") != "ok":
        return False
    ce = c.get("completed_epochs", 0)
    # cross-check contra o results.csv real (defende de sentinela obsoleto/corrompido)
    rcsv = os.path.join(d, "results.csv")
    if not os.path.exists(rcsv):
        return False
    with open(rcsv, encoding="utf-8") as f:
        rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
    if rows != ce:
        return False
    # cobertura: atingiu o esperado, ou parou cedo de forma legitima (patience)
    if ce >= expected_epochs:
        return True
    return bool(c.get("stopped_early")) and ce > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    cfgs = [c for c in grid() if args.only in run_name(*c)]
    log = []
    pending = [c for c in cfgs if not done(run_name(*c), args.epochs)]
    print(f"Grade: {len(cfgs)} treinos | ja concluidos: {len(cfgs)-len(pending)} | "
          f"a rodar: {len(pending)} | epochs={args.epochs}")
    for c in cfgs:
        print(f"   {'[OK]' if done(run_name(*c), args.epochs) else '[  ]'} {run_name(*c)}")
    if args.dry_run:
        return

    t_grid = time.time()
    for i, (m, pad, init, seed) in enumerate(pending, 1):
        name = run_name(m, pad, init, seed)
        print(f"\n===== [{i}/{len(pending)}] {name} =====", flush=True)
        cmd = [PY, "stage2/train_config.py", "--model", m, "--padding", pad,
               "--init", init, "--seed", str(seed), "--epochs", str(args.epochs),
               "--batch", str(BATCH), "--workers", str(WORKERS),
               "--project", PROJECT, "--name", name, "--patience", "1000"]
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = time.time() - t0
        ok = rc == 0 and done(name, args.epochs)
        log.append({"run": name, "returncode": rc, "ok": ok, "sec": round(dt, 1)})
        print(f"----- {name}: {'OK' if ok else 'FALHOU rc=%d' % rc}  ({dt/60:.1f} min)")
        with open("stage2/grid_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in log if r["ok"])
    print(f"\n===== GRADE CONCLUIDA: {ok}/{len(log)} OK em {(time.time()-t_grid)/3600:.1f} h")
    fail = [r["run"] for r in log if not r["ok"]]
    if fail:
        print(f"FAILED (re-run stage2/run_grid.py to resume): {fail}")


if __name__ == "__main__":
    main()
