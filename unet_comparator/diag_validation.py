# -*- coding: utf-8 -*-
"""
diag_validation.py — is validation really slow, or was the measurement an artefact?

diag_lentidao.py measured validation at 138 s with determinism and 19 s without. But
the old run used exactly the same set_deterministic() and the same fp32 avalia(), and
spent 105 s on the WHOLE EPOCH — training included. The numbers do not add up, which
suggests an artefact: each call measures the first execution, which carries cuDNN
algorithm selection and DataLoader worker startup.

Here validation is repeated 3x in the same configuration. If the 1st is slow and the
following ones fast, it was warm-up and there is nothing to fix.

Three configurations:
  fp32-det   as it stands today (outside autocast, cudnn.deterministic=True)
  fp32-nodet outside autocast, no determinism
  fp16-det   INSIDE autocast, deterministic=True — as in training, and as
             Ultralytics validates on the YOLO arm

    python unet_comparator/diag_validation.py
"""
import os
import statistics as st
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
os.chdir(os.path.dirname(AQUI))

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402

from unet_data import WoundDataset, carrega_splits  # noqa: E402
from unet_model import UNet                         # noqa: E402
from train_unet import set_deterministic            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


@torch.no_grad()
def avalia_cfg(modelo, loader, dev, usa_autocast, limiar=0.5):
    """faithful copy of avalia(), with the autocast as a parameter."""
    modelo.eval()
    ious, dices = [], []
    for x, y, _, _ in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=usa_autocast):
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
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        sys.exit("ABORTED: no CUDA")
    livre, _ = torch.cuda.mem_get_info()
    if livre < 4e9:
        sys.exit("ABORTED: under 4 GB free — is the grid running?")
    print(f"{torch.cuda.get_device_name(0)} · {livre/1e9:.1f} GB free\n")

    splits = carrega_splits("data.yaml")
    CFGS = [("fp32-det   (as it stands today)", True, False),
            ("fp32-nodet (no determinism)", False, False),
            ("fp16-det   (inside the autocast)", True, True)]

    for rotulo, det, ac in CFGS:
        if det:
            set_deterministic(42)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.manual_seed(42)
        modelo = UNet().to(dev)
        dl = DataLoader(WoundDataset(splits["val"], 640, treino=False, fill=0),
                        batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
        ts, ious = [], []
        for rep in range(3):
            t = time.perf_counter()
            iou, dice = avalia_cfg(modelo, dl, dev, ac)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
            ious.append(iou)
        print(f"{rotulo}")
        print(f"  tempos: " + "  ".join(f"{x:6.1f}s" for x in ts)
              + f"   (median of repeats 2-3: {st.median(ts[1:]):.1f}s)")
        print(f"  IoU   : " + "  ".join(f"{x:.6f}" for x in ious))
        if max(ious) - min(ious) > 1e-9:
            print("  WARNING: IoU varied between repeats of the SAME configuration")
        print()
        del modelo
        torch.cuda.empty_cache()

    print("If the 1st repeat is slow and the rest fast, it was warm-up —")
    print("real training pays this ONCE, not every epoch, and there is nothing to fix.")


if __name__ == "__main__":
    main()
