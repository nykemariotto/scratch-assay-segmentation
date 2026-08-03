# -*- coding: utf-8 -*-
"""
stage2/test_padding_patch.py — the padding patch gate. Runs BEFORE any training.

It exists because of the 2026-07-27 defect: the patch held in the parent process and not
in the workers, and the entire grid trained with GREY padding on both arms of the
ablation. Nothing in the log gave it away — the runs took 34 h and produced numbers
plausible. Only a bit-by-bit comparison of the weights revealed it.

This test fails if that ever happens again.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

falhas = []


def check(cond, rot, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {rot}{('  ' + extra) if extra else ''}")
    if not cond:
        falhas.append(rot)


class Sonda(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        import numpy as _np
        import sys as _sys
        pp = _sys.modules.get("padding_patch")
        return {"pid": os.getpid(),
                "modo": (pp.applied() if pp else None) or "NONE",
                "full114": int(_np.full((1,), 114, dtype=_np.uint8)[0])}


def colate(b):
    return b


def main():
    import padding_patch

    print("1. patch in the parent process")
    for modo, esperado in (("black", 0), ("white", 255)):
        fill = padding_patch.apply(modo)
        check(fill == esperado, f"apply({modo!r}) -> {fill}", f"(expected {esperado})")
        check(int(np.full((1,), 114, dtype=np.uint8)[0]) == esperado,
              f"np.full(114) in the parent returns {int(np.full((1,), 114, dtype=np.uint8)[0])}")

    print("\n2. the worker_init_fn was installed in Ultralytics")
    import ultralytics.data.build as B
    check(B.seed_worker is padding_patch.worker_init,
          "build.seed_worker replaced by padding_patch.worker_init")

    print("\n3. WHAT MATTERS — the patch exists INSIDE the workers")
    for modo, esperado in (("white", 255), ("black", 0)):
        padding_patch.apply(modo)
        dl = DataLoader(Sonda(), batch_size=4, num_workers=2, collate_fn=colate,
                        worker_init_fn=padding_patch.worker_init)
        vistos = {}
        for lote in dl:
            for r in lote:
                vistos[r["pid"]] = r
        workers = {p: r for p, r in vistos.items() if p != os.getpid()}
        check(len(workers) >= 1, f"[{modo}] workers observed: {len(workers)}")
        for p, r in workers.items():
            check(r["modo"] == modo and r["full114"] == esperado,
                  f"[{modo}] worker {p}: mode={r['modo']} np.full(114)->{r['full114']}",
                  f"(expected {modo}/{esperado})")

    print("\n4. the two modes produce DIFFERENT batches (the test that was missing)")
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG

    def primeiro_lote(modo):
        padding_patch.apply(modo)
        cfg = get_cfg(DEFAULT_CFG)
        cfg.imgsz, cfg.mosaic, cfg.task = 640, 1.0, "segment"
        ds = build_yolo_dataset(cfg, "dataset/images/train", 4,
                                {"names": {0: "wound"}, "channels": 3},
                                mode="train", rect=False)
        dl = build_dataloader(ds, 4, 2, shuffle=False)
        for b in dl:
            return b["img"].numpy().copy()

    a = primeiro_lote("black")
    b = primeiro_lote("white")
    n114_a = int((a == 114).sum())
    n114_b = int((b == 114).sum())
    check(not np.array_equal(a, b), "the same batch differs between black and white")
    # Do NOT require 114 to vanish: it is a NATURAL value in these microscopy images
    # (~1.6% of pixels). What proves no padding is grey is 114 appearing in the SAME
    # quantity in both modes — if one of them still used grey for
    # preencher, a contagem dele seria muito maior.
    dif = abs(n114_a - n114_b) / max(n114_a, n114_b, 1)
    check(dif < 0.05,
          "the count of 114 is the same in both modes (natural occurrence, not padding)",
          f"(black {n114_a}, white {n114_b}, difference {100*dif:.1f}%)")
    check(int((b == 255).sum()) > int((a == 255).sum()),
          f"white has more 255 pixels than black",
          f"({int((b==255).sum())} vs {int((a==255).sum())})")

    print("\n" + "=" * 62)
    if falhas:
        print(f"{len(falhas)} FAILURE(S) — DO NOT TRAIN:")
        for f in falhas:
            print("   -", f)
        sys.exit(1)
    print("Padding patch validated inside the workers. Safe to train.")


if __name__ == "__main__":
    main()
