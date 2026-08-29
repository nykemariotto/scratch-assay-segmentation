# -*- coding: utf-8 -*-
"""
diag_validation2.py — why does avalia() take 267 s after training and 15 s on its own?

diag_fases showed validation consuming 73% of the epoch (267 s of 367 s), with the
wait for data at 0.1%. But diag_validation measured the same function, on the SAME 197
images, in 15.1 s.

The only difference is the state of the GPU: there, validation ran on its own with a
clean allocator; here it runs right after a training epoch, with the allocator cache
full of large blocks. And it is fp32, OUTSIDE the autocast, so it asks for twice the
memory per sample, in a different shape than training uses. If the allocator has to
return blocks to the driver and request others, every `cudaMalloc`/`cudaFree`
serialises, which matches the GPU reading 100% utilisation while drawing only 53 W
of 160 W.

It measures four conditions, always after a real training epoch:

  A  as it stands today                     (fp32, allocator full)
  B  with torch.cuda.empty_cache() first    (returns the cache once, deliberately)
  C  under fp16 autocast                    (half the activation memory)
  D  empty_cache + autocast                 (both)

If B or C fix it, the epoch drops from ~350 s to near 110 s, and retraining the five
seeds drops from ~48 h to ~15 h.

    python unet_comparator/diag_validation2.py
"""
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
os.chdir(os.path.dirname(AQUI))

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402

from unet_data import WoundDataset, carrega_splits, SEED_AUG   # noqa: E402
from unet_model import UNet, BCEDiceLoss                       # noqa: E402
from train_unet import avalia, set_deterministic               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


@torch.no_grad()
def avalia_ac(modelo, loader, dev, limiar=0.5):
    """avalia() with the forward under fp16 autocast; everything else identical."""
    modelo.eval()
    ious, dices = [], []
    for x, y, _, _ in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            saida = modelo(x)
        p = (torch.sigmoid(saida.float()) > limiar).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        uni = ((p + y) > 0).float().sum(dim=(1, 2, 3))
        soma = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        for i in range(x.shape[0]):
            ious.append(1.0 if uni[i] == 0 else (inter[i] / uni[i]).item())
            dices.append(1.0 if soma[i] == 0 else (2 * inter[i] / soma[i]).item())
    return float(np.mean(ious)), float(np.mean(dices))


def main():
    dev = "cuda"
    if not torch.cuda.is_available():
        sys.exit("ABORTED: no CUDA")
    livre, tot = torch.cuda.mem_get_info()
    if livre < 4e9:
        sys.exit("ABORTED: under 4 GB free — is the grid running?")
    print(f"GPU: {livre/1e9:.1f} GB free of {tot/1e9:.1f}\n")

    set_deterministic(42)
    splits = carrega_splits("data.yaml")
    ds_tr = WoundDataset(splits["train"], 640, treino=True, fill=0, seed=42)
    ds_tr.set_epoca(1)
    ds_va = WoundDataset(splits["val"], 640, treino=False, fill=0)
    g = torch.Generator()
    g.manual_seed(SEED_AUG)
    dl_tr = DataLoader(ds_tr, batch_size=4, shuffle=True, num_workers=2,
                       pin_memory=True, generator=g)
    dl_va = DataLoader(ds_va, batch_size=4, shuffle=False, num_workers=2,
                       pin_memory=True)

    modelo = UNet().to(dev)
    crit = BCEDiceLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=5e-4)
    escala = torch.amp.GradScaler("cuda", enabled=True)

    print("running ONE training epoch to leave the allocator in its real state…")
    modelo.train()
    t = time.perf_counter()
    for x, y, _, _ in dl_tr:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            perda = crit(modelo(x), y)
        escala.scale(perda).backward()
        escala.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=10.0)
        escala.step(opt)
        escala.update()
        float(perda.detach())
    torch.cuda.synchronize()
    print(f"training: {time.perf_counter()-t:.1f}s\n")

    def reservado():
        return torch.cuda.memory_reserved() / 1e9

    CONDS = [("A  as it stands today (fp32, allocator full)", False, False),
             ("B  empty_cache() first", True, False),
             ("C  fp16 autocast on the forward", False, True),
             ("D  empty_cache + autocast", True, True)]
    base = None
    for rot, limpa, ac in CONDS:
        if limpa:
            torch.cuda.empty_cache()
        r0 = reservado()
        t = time.perf_counter()
        iou, dice = (avalia_ac if ac else avalia)(modelo, dl_va, dev)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        if base is None:
            base = dt
        print(f"{rot:42s} {dt:7.1f}s  ({base/dt:4.1f}x)  "
              f"IoU {iou:.6f}  reserved {r0:.2f}->{reservado():.2f} GB")

    print(f"""
HOW TO READ THIS. If B or C bring the time down, the bottleneck is the allocator,
not the computation. The epoch today is ~350 s: 95 s of training + 267 s of
validation. With validation at ~15 s it drops to ~115 s, and the 5 seeds go from
~48 h to ~16 h.

Careful: condition C changes the number. Compare the IoUs above: if they differ, the
choice between B and C stops being only about performance.""")


if __name__ == "__main__":
    main()
