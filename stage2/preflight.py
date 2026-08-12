# -*- coding: utf-8 -*-
"""
stage2/preflight.py — the mandatory check before spending 34 h of GPU.

Written after losing 34 h because the padding patch held in the parent process and
not in the workers. `stage2/verify_padding.py` missed it: it built its OWN dataset in
the main process — it measured the right place in the wrong process.

PRINCIPLE. Verification in a parallel pipeline does not count. What counts is:
  (a) proving the invariant INSIDE the process that does the work, and
  (b) comparing the OUTPUTS of short real training runs.

The canary (phase E) actually trains, for 2 epochs, calling the grid's own
`stage2/train_config.py`, and asserts three things the broken grid did not satisfy:

    black != white   -> the padding reaches training   (this is what was failing)
    seed42 != seed43 -> the seed changes something
    black == black   -> same config, same result       (determinism)

It costs ~6 min. The grid costs 34 h.

    python stage2/preflight.py            # everything
    python stage2/preflight.py --rapido   # skips the canary (does not clear the grid)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = sys.executable
# RAIZ is the REPOSITORY root, one level above this file. Everything below runs on
# paths relative to it (data.yaml, the COCO checkpoints, runs/), so the chdir has to
# climb. While preflight.py sat at the root, `dirname(__file__)` was already the root
# and the chdir was a no-op; moving it into stage2/ made it land one level too deep,
# and every relative path silently missed. Same convention as the stage3 scripts.
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
os.chdir(RAIZ)
CANARIO = os.path.join("runs", "segment", "_preflight")

falhas, avisos = [], []


def ok(cond, rot, extra="", fatal=True):
    marca = "ok " if cond else ("FAIL" if fatal else "warn")
    print(f"  [{marca}] {rot}{('  ' + extra) if extra else ''}")
    if not cond:
        (falhas if fatal else avisos).append(rot)
    return cond


def info(rot, extra=""):
    """Informational only. NOT a check.

    It exists because four calls to ok() were decorative — `ok(True, ...)` and
    `ok(cond or True, ...)` printed "[ok ]" without being able to fail. A check that
    cannot fail is worse than no check: it creates false confidence, which is exactly
    how stage2/verify_padding.py let the padding defect through. Marking these as info
    makes the distinction visible in the output.
    """
    print(f"  [info ] {rot}{('  ' + extra) if extra else ''}")


def sec(t):
    print(f"\n{t}\n" + "-" * len(t))


# ══════════════════════════════════════════════════════════ A. ambiente
def fase_ambiente():
    sec("A. AMBIENTE")
    import torch
    ok(torch.cuda.is_available(), "GPU available",
       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    import ultralytics
    info(f"ultralytics {ultralytics.__version__}")
    info(f"torch {torch.__version__} · cuda {torch.version.cuda}")

    livre = shutil.disk_usage(RAIZ).free / 2**30
    # 25 runs x (best+last); yolo11x-seg weighs ~120 MB per run in the worst case
    ok(livre > 20, f"disk space: {livre:.0f} GB free", "(minimum required: 20 GB)")

    for m in ("yolo11s-seg", "yolo11m-seg", "yolo11x-seg"):
        p = f"{m}.pt"
        ok(os.path.isfile(p), f"COCO checkpoint present: {p}",
           f"({os.path.getsize(p)/2**20:.0f} MB)" if os.path.isfile(p) else "")

    # The `scratch` arm instantiates YOLO("<model>.yaml"). If Ultralytics cannot
    # resolve that name, 5 runs fail after hours in the queue. The previous check
    # was `os.path.isfile(...) or True` — it always passed.
    from ultralytics import YOLO
    for m in ("yolo11s-seg", "yolo11m-seg", "yolo11x-seg"):
        try:
            n = sum(x.numel() for x in YOLO(f"{m}.yaml").model.parameters())
            ok(n > 0, f"architecture {m}.yaml builds (scratch arm)",
               f"({n/1e6:.1f} M parameters)")
        except Exception as e:
            ok(False, f"architecture {m}.yaml builds (scratch arm)",
               f"-> {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════ B. dados
def fase_dados():
    sec("B. DATA AND PARTITION")
    import yaml
    d = yaml.safe_load(open("data.yaml", encoding="utf-8"))
    # same rule as unet_data.py: a relative `path` resolves against the YAML
    aqui = os.path.dirname(os.path.abspath("data.yaml"))
    base = d.get("path") or aqui
    if not os.path.isabs(base):
        base = os.path.join(aqui, base)
    cont = {}
    for k, esperado in (("train", 932), ("val", 197), ("test", 234)):
        p = d[k] if os.path.isabs(d[k]) else os.path.join(base, d[k])
        imgs = [f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                for f in glob.glob(os.path.join(p, e))]
        lab = glob.glob(os.path.join(p.replace("images", "labels"), "*.txt"))
        cont[k] = len(imgs)
        ok(len(imgs) == esperado, f"{k}: {len(imgs)} images", f"(expected {esperado})")
        ok(len(lab) == esperado, f"{k}: {len(lab)} labels", f"(expected {esperado})")
    ok(sum(cont.values()) == 1363, f"total active = {sum(cont.values())}", "(expected 1363)")

    M = list(csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8-sig")))
    A = [r for r in M if r["partition"] in ("train", "val", "test")]
    from collections import defaultdict
    for col, n_esp in (("group_key", 265), ("split_key", 246)):
        g = defaultdict(set)
        for r in A:
            g[r[col]].add(r["partition"])
        cruzam = sum(1 for v in g.values() if len(v) > 1)
        ok(cruzam == 0, f"{col}: {len(g)} keys, {cruzam} crossing the partition",
           "(has to be 0 — it is the central requirement of the partition)")

    # Negatives: an EMPTY label (image with no annotated wound) != a MISSING label.
    # The previous check was ok(True, ...) — it reported the number and never failed.
    vaz = sum(1 for f in glob.glob(os.path.join(base, "labels", "train", "*.txt"))
              if os.path.getsize(f) == 0)
    ok(vaz == 103, f"negatives in train (empty label): {vaz}", "(expected 103)")
    ausentes = sum(1 for f in glob.glob(os.path.join(base, "images", "train", "*.*"))
                   if not os.path.isfile(os.path.join(
                       base, "labels", "train",
                       os.path.splitext(os.path.basename(f))[0] + ".txt")))
    ok(ausentes == 0, f"train images with no label file: {ausentes}",
       "(has to be 0 — missing is not the same as negative)")


# ══════════════════════════════════════════════════════════ C. patch
def fase_patch():
    sec("C. PADDING PATCH (parent process + workers)")
    r = subprocess.run([PY, "stage2/test_padding_patch.py"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    linhas = [l for l in (r.stdout or "").splitlines() if "[ok ]" in l or "[FAIL]" in l]
    for l in linhas:
        print("   " + l.strip())
    ok(r.returncode == 0, f"stage2/test_padding_patch.py ({len(linhas)} checks)",
       "" if r.returncode == 0 else "-> see the output above")


# ══════════════════════════════════════════════════════════ D. evaluation
def fase_avaliacao():
    sec("D. EVALUATION PIPELINE (no use training if you cannot measure)")
    for s, rot in ((["stage3/test_ap_core.py"], "AP core (9 cases)"),):
        r = subprocess.run([PY] + s, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        ok(r.returncode == 0, rot)
    ok(os.path.isfile("stage3/eval_test.py") and os.path.isfile("stage3/aggregate.py"),
       "stage3/eval_test.py and stage3/aggregate.py present")

    # guard: the pre-correction runs must be refused by the evaluation
    arq = os.path.join("runs", "segment", "runs_revision_D12_invalido")
    if os.path.isdir(arq):
        import json
        velhos = glob.glob(os.path.join(arq, "*", "provenance.json"))
        com_ev = sum(1 for p in velhos
                     if "evidencia_padding_no_batch" in json.load(open(p, encoding="utf-8")))
        ok(len(velhos) > 0 and com_ev == 0,
           f"archived pre-correction runs still lack padding evidence: {com_ev}/{len(velhos)}",
           "(this is what makes stage3/eval_test.py refuse them)")

    # U-Net comparator: the two tests that closed the augmentation defect
    for s, rot in ((["unet_comparator/smoke_test.py"], "U-Net · smoke test"),
                   (["unet_comparator/test_epoca_workers.py"],
                    "U-Net · augmentation varies by epoch inside the workers")):
        if not os.path.isfile(s[0]):
            ok(False, rot, "-> file missing", fatal=False)
            continue
        r = subprocess.run([PY] + s, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        ok(r.returncode == 0, rot, "" if r.returncode == 0 else "-> failed", fatal=False)


# ══════════════════════════════════════════════════════════ E. canary
def treina_canario(padding, seed, epochs, nome):
    d = os.path.join(CANARIO, nome)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run(
        [PY, "stage2/train_config.py", "--model", "yolo11m-seg", "--padding", padding,
         "--init", "coco", "--seed", str(seed), "--epochs", str(epochs),
         "--batch", "4", "--workers", "2", "--patience", "1000",
         "--project", "_preflight", "--name", nome],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r, d


def pesos(d):
    import torch
    p = os.path.join(d, "weights", "last.pt")
    if not os.path.isfile(p):
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return {k: v.float() for k, v in ck["model"].state_dict().items()
            if v.dtype.is_floating_point}


def iguais(a, b):
    import torch
    if a is None or b is None:
        return None
    ks = [k for k in a if k in b and a[k].shape == b[k].shape]
    return sum(1 for k in ks if torch.equal(a[k], b[k])), len(ks)


def fase_canario(epochs):
    sec(f"E. CANARY — a real {epochs}-epoch training run (the test that was missing)")
    print("   It really trains, through the grid's own stage2/train_config.py, and compares")
    print("   the OUTPUTS. It is the only way to prove the padding reaches training.\n")

    plano = [("black", 42, "cn_black_42"), ("white", 42, "cn_white_42"),
             ("black", 43, "cn_black_43"), ("black", 42, "cn_black_42_bis")]
    saidas = {}
    for pad, seed, nome in plano:
        t = time.time()
        r, d = treina_canario(pad, seed, epochs, nome)
        bom = r.returncode == 0 and os.path.isfile(os.path.join(d, "weights", "last.pt"))
        print(f"   {nome:<18} {pad:<6} seed{seed}  {'ok' if bom else 'FAILED'}  "
              f"({time.time()-t:.0f}s)")
        if not bom:
            print((r.stderr or "")[-1200:])
            falhas.append(f"canary {nome} did not train")
            return
        saidas[nome] = d

    print()
    # ---- evidence recorded from inside training
    for nome in ("cn_black_42", "cn_white_42"):
        p = os.path.join(saidas[nome], "provenance.json")
        ev = json.load(open(p, encoding="utf-8")).get("evidencia_padding_no_batch", {})
        alvo = "frac_0" if "black" in nome else "frac_255"
        ok(ev and "erro" not in ev, f"{nome}: evidence recorded from the real batch",
           f"0={100*ev.get('frac_0',0):.2f}% 114={100*ev.get('frac_114',0):.2f}% "
           f"255={100*ev.get('frac_255',0):.2f}%")
        ok(ev.get(alvo, 0) > 0.005,
           f"{nome}: the requested padding is present in the REAL training batch",
           f"({alvo}={100*ev.get(alvo,0):.2f}%)")
        ok(ev.get("frac_114", 0) < 0.5 * ev.get(alvo, 1e-9),
           f"{nome}: grey 114 does NOT dominate",
           f"(114={100*ev.get('frac_114',0):.2f}%)")
        ok(not os.path.isfile(os.path.join(saidas[nome], "AVISO_PADDING.txt")),
           f"{nome}: no AVISO_PADDING")

    print()
    # ---- the three invariants
    wb42, ww42 = pesos(saidas["cn_black_42"]), pesos(saidas["cn_white_42"])
    wb43 = pesos(saidas["cn_black_43"])
    wb42b = pesos(saidas["cn_black_42_bis"])

    ig, tot = iguais(wb42, ww42)
    ok(ig < tot, "INVARIANT 1 — black != white (the padding reaches training)",
       f"({ig}/{tot} tensors equal; the broken grid gave {tot}/{tot})")

    ig, tot = iguais(wb42, wb43)
    ok(ig < tot, "INVARIANT 2 — seed42 != seed43 (the seed changes something)",
       f"({ig}/{tot} equal)")

    ig, tot = iguais(wb42, wb42b)
    ok(ig == tot, "INVARIANT 3 — same config twice = identical (determinism)",
       f"({ig}/{tot} equal)")

    # ---- the training losses have to diverge, not only the weights
    def loss(d):
        rows = list(csv.DictReader(open(os.path.join(d, "results.csv"), encoding="utf-8")))
        c = next(x for x in rows[0] if "train/box_loss" in x)
        return [float(r[c]) for r in rows]
    lb, lw = loss(saidas["cn_black_42"]), loss(saidas["cn_white_42"])
    idem = sum(1 for x, y in zip(lb, lw) if x == y)
    ok(idem == 0, "INVARIANT 4 — the training loss diverges from the 1st epoch",
       f"({idem}/{len(lb)} identical epochs; the broken grid gave 100/100)")


# ══════════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="skip the canary")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--manter", action="store_true", help="do not delete the canary")
    args = ap.parse_args()

    print("=" * 74)
    print("PREFLIGHT — the check before 34 h of GPU")
    print("=" * 74)

    fase_ambiente()
    fase_dados()
    fase_patch()
    fase_avaliacao()
    if args.rapido:
        print("\n[--rapido] canary SKIPPED — this does NOT clear the grid.")
        avisos.append("canary not executed")
    else:
        fase_canario(args.epochs)
        if not args.manter and os.path.isdir(CANARIO):
            shutil.rmtree(CANARIO, ignore_errors=True)
            print(f"\n   (canary removed: {CANARIO})")

    print("\n" + "=" * 74)
    if falhas:
        print(f"DO NOT TRAIN — {len(falhas)} failure(s):")
        for f in falhas:
            print("   ✗", f)
        sys.exit(1)
    if avisos:
        print(f"{len(avisos)} warning(s):")
        for a in avisos:
            print("   !", a)
    print("PREFLIGHT OK" + (" (with warnings)" if avisos else "") + " — the grid can run.")
    if args.rapido:
        sys.exit(2)


if __name__ == "__main__":
    main()
