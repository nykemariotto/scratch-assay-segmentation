# -*- coding: utf-8 -*-
"""
diag_spawn.py — how much does creating the DataLoader worker processes cost?

Training (91.7 s) + validation (15.1 s) = 106.8 s, but a real epoch takes 268 s.
About 161 s are missing. With `persistent_workers=False`, every epoch creates 4
processes: 2 for the training loader and 2 for the validation one. On Windows the
start method is `spawn`: each worker brings up a fresh interpreter and RE-IMPORTS
the __main__ module, which drags in torch, numpy and cv2.

The day before, the same gap was 83 s. It doubled, and nothing in the code changed.
This script measures iterator creation in isolation, without touching the model.

    python unet_comparator/diag_spawn.py
"""
import os
import statistics as st
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
os.chdir(os.path.dirname(AQUI))

import torch                                        # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402

from unet_data import WoundDataset, carrega_splits, SEED_AUG   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def mede(ds, workers, persistent, n_ciclos=3, n_lotes=2):
    """time until the FIRST batch comes out — the cost of bringing the workers up."""
    g = torch.Generator()
    g.manual_seed(SEED_AUG)
    kw = {}
    if workers > 0:
        kw["persistent_workers"] = persistent
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=workers,
                    pin_memory=True, generator=g, **kw)
    ts = []
    for _ in range(n_ciclos):
        t = time.perf_counter()
        it = iter(dl)                      # <- the spawn happens here
        next(it)                           # waits for the first batch
        ts.append(time.perf_counter() - t)
        for _ in range(n_lotes - 1):
            try:
                next(it)
            except StopIteration:
                break
        del it
    del dl
    return ts


def main():
    print(f"start method: {torch.multiprocessing.get_start_method()}  ·  "
          f"CPUs: {os.cpu_count()}\n")
    splits = carrega_splits("data.yaml")
    ds_tr = WoundDataset(splits["train"], 640, treino=True, fill=0, seed=42)
    ds_tr.set_epoca(1)
    ds_va = WoundDataset(splits["val"], 640, treino=False, fill=0)

    print("time to the FIRST batch (3 cycles; = the cost of bringing workers up)\n")
    for rot, ds in (("train (932 imgs)", ds_tr), ("val (197 imgs)", ds_va)):
        for workers, persistent, nome in ((0, False, "num_workers=0 (no process)"),
                                          (2, False, "num_workers=2, NOT persistent"),
                                          (2, True, "num_workers=2, persistent")):
            ts = mede(ds, workers, persistent)
            print(f"  {rot:22s} {nome:32s} "
                  + "  ".join(f"{x:6.2f}s" for x in ts)
                  + f"   mediana {st.median(ts):6.2f}s")
        print()

    print("HOW TO READ THIS. With persistent_workers=False, the cost of cycle 1 repeats")
    print("EVERY EPOCH, for both loaders. Multiply by 100 epochs × 5 seeds.")
    print("Cycle 1 of the persistent mode is the spawn; cycles 2-3 show what would be")
    print("saved if the workers survived — but they must NOT survive: it is the pickle")
    print("of the dataset each epoch that carries `set_epoca` into the workers (see")
    print("test_epoca_workers.py). The cost is the price of per-epoch augmentation.")


if __name__ == "__main__":
    main()
