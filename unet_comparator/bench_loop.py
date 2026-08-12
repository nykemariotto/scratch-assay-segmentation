# -*- coding: utf-8 -*-
"""
bench_loop.py — measures the cost of each change made to the training loop.

WHY. After the fp16 patch the epoch went from ~105 s to ~256 s (2.4x). There is a
suspect — the synchronisation point moved from after the backward to before it — but
the arithmetic does not add up: the difference is 0.69 s per batch, and a sync adds no
work. So this measures instead of guessing.

Four variants, same model, same data, same number of batches:

  A  original      sync after the step, no clipping
  B  sync first    sync before the backward, no clipping  <- isolates the sync
  C  clipping      sync after the step, with unscale_+clip <- isolates the clipping
  D  current       sync first + clipping                  <- what is running

If D is close to A, the slowness comes from elsewhere (desktop contention, thermal throttling) and
changing the loop will not fix it.

DO NOT RUN WITH THE GRID ACTIVE — it takes 8 GB of GPU and would contend for memory.

    python unet_comparator/bench_loop.py --lotes 30
"""
import argparse
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
from unet_model import UNet, BCEDiceLoss                       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def roda(variante, dl, dev, lotes, lr=1e-3):
    torch.manual_seed(42)
    modelo = UNet().to(dev)
    crit = BCEDiceLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=5e-4)
    escala = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    sync_antes = variante in ("B", "D")
    clipa = variante in ("C", "D")

    modelo.train()
    tempos, i = [], 0
    for x, y, _, _ in dl:
        if i >= lotes + 3:                # 3 lotes de aquecimento, descartados
            break
        t = time.perf_counter()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            perda = crit(modelo(x), y)
        if sync_antes:
            v = float(perda.detach())
        escala.scale(perda).backward()
        if clipa:
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=10.0)
        escala.step(opt)
        escala.update()
        if not sync_antes:
            v = float(perda.detach())
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= 3:
            tempos.append(time.perf_counter() - t)
        i += 1
    del modelo, opt, escala
    if dev == "cuda":
        torch.cuda.empty_cache()
    return tempos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotes", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        livre, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)} · "
              f"{livre/1e9:.1f} GB free of {total/1e9:.1f} GB")
        if livre < 4e9:
            sys.exit("ABORTED: under 4 GB free — is the grid still running?")

    splits = carrega_splits("data.yaml")
    ds = WoundDataset(splits["train"], args.imgsz, treino=True, fill=0, seed=42)
    ds.set_epoca(1)
    g = torch.Generator()
    g.manual_seed(SEED_AUG)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=(dev == "cuda"), generator=g)

    NOMES = {"A": "original     (sync after,  no clip)",
             "B": "sync first   (sync first,  no clip)",
             "C": "clipping     (sync after,  with clip)",
             "D": "current      (sync first,  with clip)"}
    res = {}
    for v in ("A", "B", "C", "D"):
        t = roda(v, dl, dev, args.lotes)
        res[v] = st.median(t)
        print(f"  {v} · {NOMES[v]:38s} median {res[v]*1000:7.1f} ms/batch "
              f"(n={len(t)})")

    base = res["A"]
    print(f"\nrelativo ao original (A = 1,00x):")
    for v in ("B", "C", "D"):
        print(f"  {v} {NOMES[v][:12]:14s} {res[v]/base:5.2f}x  "
              f"({(res[v]-base)*1000:+7.1f} ms/batch)")

    n_lotes_epoca = (len(ds) + args.batch - 1) // args.batch
    print(f"\nper-epoch projection ({n_lotes_epoca} batches):")
    for v in ("A", "D"):
        print(f"  {v}: {res[v]*n_lotes_epoca:6.1f} s/epoch -> "
              f"{res[v]*n_lotes_epoca*100*5/3600:5.1f} h for the 5 seeds")

    d_sync, d_clip = res["B"] - base, res["C"] - base
    print(f"\nveredito:")
    print(f"  cost of the early sync : {d_sync*1000:+7.1f} ms/batch")
    print(f"  cost of the clipping   : {d_clip*1000:+7.1f} ms/batch")
    if res["D"] / base < 1.25:
        print("  NEITHER of the two explains the slowness — the cause is outside the loop\n"
              "  (desktop contention, thermal throttling, GPU state).\n"
              "  Changing the loop would NOT fix it; reverting would be change for its own sake.")
    else:
        pior = "the early sync" if d_sync > d_clip else "the clipping"
        print(f"  the cost is in: {pior}. Worth fixing before moving on.")


if __name__ == "__main__":
    main()
