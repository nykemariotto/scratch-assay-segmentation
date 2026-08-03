# -*- coding: utf-8 -*-
"""
test_epoca_workers.py — does `set_epoca` survive the worker processes?

The lesson of the padding defect: a property that holds in the parent process may
not hold in the worker that actually builds the batch. The per-epoch augmentation
fix depends on the `epoca` attribute travelling in the dataset pickle at every
iterator creation (persistent_workers=False, the default). If it does not travel,
the augmentation freezes again — silently.

It has to be a FILE, not a heredoc: on Windows the spawn re-imports the __main__
module, and without the `if __name__ == "__main__"` guard the process hangs.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
os.chdir(os.path.dirname(AQUI))

import torch
from torch.utils.data import DataLoader, Subset

from unet_data import WoundDataset, carrega_splits


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sp = carrega_splits("data.yaml")
    ds = WoundDataset(sp["train"], 640, treino=True, fill=0, seed=42)
    sub = Subset(ds, list(range(4)))

    def lote(workers, epoca):
        ds.set_epoca(epoca)
        dl = DataLoader(sub, batch_size=4, shuffle=False, num_workers=workers)
        for x, _, _, _ in dl:
            return x.clone()

    falhas = []
    print(f"{'workers':>8}  {'epoch 1 × epoch 2':<22} {'epoch 1 × epoch 1':<20}")
    print("-" * 56)
    for w in (0, 2):
        a = lote(w, 1)
        b = lote(w, 2)
        c = lote(w, 1)
        varia = not torch.equal(a, b)
        estavel = torch.equal(a, c)
        print(f"{w:>8}  {'DIFFER   ok' if varia else 'IDENTICAL  FAIL':<22} "
              f"{'identical ok' if estavel else 'differ  FAIL':<20}")
        if not varia:
            falhas.append(f"workers={w}: augmentation does not vary between epochs")
        if not estavel:
            falhas.append(f"workers={w}: the same epoch does not reproduce")

    print()
    if falhas:
        print("FAILED:")
        for f in falhas:
            print("   ✗", f)
        sys.exit(1)
    print("`set_epoca` survives the spawn: augmentation varies by epoch inside the")
    print("workers too, and stays reproducible given the epoch.")


if __name__ == "__main__":
    main()
