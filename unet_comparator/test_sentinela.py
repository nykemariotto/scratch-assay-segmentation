# -*- coding: utf-8 -*-
"""
test_sentinela.py — does the sentinel reject the run the old version approved?

This is the question nobody asked about the padding defect or about the fp16
overflow: a checker is only worth having if it is exercised against the failure it
is supposed to catch. Here it runs against the real results.csv of
unet_black_seed42 — the run that recorded `"status": "ok"` with 60 of 100 epochs
in NaN.

Beyond that, the test covers the false positives: a synthetic healthy run, a run
with one isolated coincidence of val_iou == val_dice (which must not be rejected),
and a run with three consecutive coincidences (which must be rejected).

    python unet_comparator/test_sentinela.py
"""
import csv
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_unet import diagnostica_csv  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

falhas = []


def ok(cond, msg):
    print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")
    if not cond:
        falhas.append(msg)


def escreve(linhas):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                    newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "val_iou", "val_dice", "lr", "seconds"])
    w.writerows(linhas)
    f.close()
    return f.name


def reprovado(diag, epochs):
    return bool(diag["_ruins"] or diag["_degeneradas"]
                or diag["completed_epochs"] < epochs)


# ── 1. the REAL CSV of the destroyed run ────────────────────────────────────
print("1. against the real results.csv of unet_black_seed42 (the run that passed)")
# The target is the QUARANTINED run, not just any seed42. The old glob was
# `unet_comparator*` and started matching two files once the new grid produced a
# clean seed42 — the test then asserted over the wrong file and failed. A test
# that changes its own target is worse than a test that does not exist.
alvos = [p for p in glob.glob(os.path.join(
    "runs", "segment", "unet_comparator_INVALIDO_fp16", "unet_black_seed42",
    "results.csv"))]
if not alvos:
    print("     [SKIPPED] the quarantined run is not in")
    print("              runs/segment/unet_comparator_INVALIDO_fp16/unet_black_seed42/")
    print("              The synthetic tests (2-5) still apply.")
else:
    d = diagnostica_csv(alvos[0], 100)
    print(f"     {alvos[0]}")
    print(f"     epochs {d['completed_epochs']} · non-finite {d['nan_epochs']}"
          f" (1st at {d['primeira_epoca_nao_finita']}) · degenerate"
          f" {d['epocas_iou_igual_dice']} · IoU final {d['final_val_iou']}")
    ok(reprovado(d, 100), "REJECTS the run the old sentinel approved")
    ok(d["nan_epochs"] == 52, f"counts 52 non-finite epochs (saw {d['nan_epochs']})")
    ok(d["primeira_epoca_nao_finita"] == 41,
       f"points at epoch 41 as the first (saw {d['primeira_epoca_nao_finita']})")

# ── 2. a healthy run must not be rejected ───────────────────────────────────
print("\n2. false positive: a healthy run has to pass")
saudavel = [[e, round(0.5 - 0.003 * e, 6), round(0.30 + 0.005 * e, 6),
             round(0.45 + 0.004 * e, 6), 1e-3, 100.0] for e in range(1, 101)]
p = escreve(saudavel)
d = diagnostica_csv(p, 100)
ok(not reprovado(d, 100), "a healthy 100-epoch run passes")
os.unlink(p)

# ── 3. ONE isolated coincidence must not reject ─────────────────────────────
print("\n3. false positive: one isolated coincidence of val_iou == val_dice")
uma = [r[:] for r in saudavel]
uma[49][3] = uma[49][2]          # epoch 50: dice == iou, on its own
p = escreve(uma)
d = diagnostica_csv(p, 100)
ok(not reprovado(d, 100),
   f"one isolated coincidence does not reject (degenerate={d['epocas_iou_igual_dice']})")
os.unlink(p)

# ── 4. three in a row must be rejected ──────────────────────────────────────────
print("\n4. true positive: three consecutive epochs with val_iou == val_dice")
tres = [r[:] for r in saudavel]
for i in (49, 50, 51):
    tres[i][3] = tres[i][2]
p = escreve(tres)
d = diagnostica_csv(p, 100)
ok(reprovado(d, 100),
   f"three consecutive reject (degenerate={d['epocas_iou_igual_dice']})")
os.unlink(p)

# ── 5. run interrompido tem de reprovar ──────────────────────────────────────
print("\n5. true positive: a run interrupted before 100 epochs")
p = escreve(saudavel[:37])
d = diagnostica_csv(p, 100)
ok(reprovado(d, 100), f"37/100 epochs rejects (stopped_early={d['stopped_early']})")
os.unlink(p)

print("\n" + "=" * 68)
if falhas:
    print(f"{len(falhas)} FAILURE(S):")
    for f in falhas:
        print("  · " + f)
    sys.exit(1)
print("sentinel verified against the real failure and against false positives")
