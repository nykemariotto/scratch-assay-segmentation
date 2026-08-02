# -*- coding: utf-8 -*-
"""
estimate_cost.py — what the U-Net should cost, without occupying the GPU.

WHY NOT EXTRAPOLATE FROM PARAMETERS. The U-Net has 31.0 M and yolo11m-seg ~22 M,
which would suggest a similar cost. But parameters are not work: the U-Net runs its
first two convolutions at FULL 640x640 with 64 channels, while YOLO drops to /2 at
the first stem and never returns to full resolution. The cost lies in the
resolution × channels of each stage, not in the number of weights.

This script measures two things, both on CPU:
  (1) MACs per forward — via thop, which ships with Ultralytics;
  (2) wall-clock time of forward+backward on CPU, as an empirical proxy.

Neither predicts GPU wall-clock exactly (memory bandwidth and kernel efficiency
come into it), but the ratio between the two models is defensible — and it is
anchored on the ~78 min/seed that yolo11m-seg actually took.
"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

from unet_model import UNet

assert not torch.cuda.is_available(), "GPU visible — aborting so as not to compete with the grid"
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

IMGSZ, BATCH = 640, 1
MEDIDO_YOLO_M_MIN = 76.0          # real mean of the 5 yolo11m-seg seeds, provenance.json
N_TREINO = 932                    # training images
EPOCHS = 100
SEEDS = 5


def macs(modelo, entrada):
    try:
        from thop import profile
        m, _ = profile(modelo, inputs=(entrada,), verbose=False)
        return m
    except Exception as e:
        print(f"  (thop unavailable: {e})")
        return None


def _escalar(y):
    """reduce the output to a differentiable scalar, be it tensor, list or dict.

    yolo11m-seg in training mode returns a dict (one tensor per head); the U-Net
    returns a tensor. The value of the loss does not matter here — only that the
    backward traverses the whole graph, which is what is being timed.
    """
    if isinstance(y, dict):
        y = list(y.values())
    if isinstance(y, (list, tuple)):
        partes = [t for t in y if torch.is_tensor(t) and t.requires_grad]
        if not partes:
            partes = [t for t in y if torch.is_tensor(t)]
        return sum(t.float().pow(2).mean() for t in partes)
    return y.float().pow(2).mean()


def cronometra(modelo, entrada, n=3, treino=True):
    modelo.train(treino)
    ts = []
    for _ in range(n):
        t = time.time()
        y = modelo(entrada)
        if treino:
            _escalar(y).backward()
            modelo.zero_grad(set_to_none=True)
        ts.append(time.time() - t)
    return sorted(ts)[len(ts) // 2]


x = torch.zeros(BATCH, 3, IMGSZ, IMGSZ)

print("=" * 70)
print(f"U-NET COST — input {BATCH}x3x{IMGSZ}x{IMGSZ}, CPU, "
      f"{torch.get_num_threads()} threads")
print("=" * 70)

modelos = {}
u = UNet()
modelos["U-Net (base 64)"] = u
try:
    from ultralytics import YOLO
    y = YOLO("yolo11m-seg.pt")
    modelos["yolo11m-seg"] = y.model.float()
except Exception as e:
    print(f"\n[warning] could not load yolo11m-seg: {e}")

res = {}
for nome, m in modelos.items():
    npar = sum(p.numel() for p in m.parameters())
    g = macs(m, x)
    print(f"\n{nome}")
    print(f"  parameters ......... {npar/1e6:>8.1f} M")
    if g:
        print(f"  MACs / forward ..... {g/1e9:>8.1f} G")
    # Ultralytics turns requires_grad off when loading a .pt for inference;
    # without turning it back on, the backward fails with "does not require grad"
    for p in m.parameters():
        p.requires_grad_(True)
    t_fwd = cronometra(m, x, treino=False)
    print(f"  forward (CPU) ...... {t_fwd:>8.2f} s")
    try:
        t_bwd = cronometra(m, x, treino=True)
        print(f"  fwd+bwd (CPU) ...... {t_bwd:>8.2f} s")
    except Exception as e:
        t_bwd = None
        print(f"  fwd+bwd (CPU) ...... unavailable ({type(e).__name__})")
    res[nome] = {"par": npar, "macs": g, "fwd": t_fwd, "step": t_bwd}

print("\n" + "=" * 70)
print("EXTRAPOLATION")
print("=" * 70)
if "yolo11m-seg" in res:
    ru, ry = res["U-Net (base 64)"], res["yolo11m-seg"]
    r_par = ru["par"] / ry["par"]
    r_mac = (ru["macs"] / ry["macs"]) if (ru["macs"] and ry["macs"]) else None
    r_cpu = (ru["step"] / ry["step"]) if (ru["step"] and ry["step"]) else None
    r_fwd = (ru["fwd"] / ry["fwd"]) if (ru["fwd"] and ry["fwd"]) else None
    print(f"\n  razão U-Net / yolo11m-seg")
    print(f"    por parâmetros ... {r_par:>6.2f}x   <- engana, ver cabeçalho")
    if r_mac:
        print(f"    por MACs ......... {r_mac:>6.2f}x")
    if r_fwd:
        print(f"    forward em CPU ... {r_fwd:>6.2f}x")
    if r_cpu:
        print(f"    passo em CPU ..... {r_cpu:>6.2f}x")
    faixa = sorted(v for v in (r_mac, r_fwd, r_cpu) if v)
    lo, hi = faixa[0], faixa[-1]
    print(f"\n  ancorado nos {MEDIDO_YOLO_M_MIN:.0f} min/seed medidos do yolo11m-seg:")
    print(f"    U-Net por seed ... {lo*MEDIDO_YOLO_M_MIN/60:>5.1f} h  a  {hi*MEDIDO_YOLO_M_MIN/60:>5.1f} h")
    print(f"    {SEEDS} seeds ......... {SEEDS*lo*MEDIDO_YOLO_M_MIN/60:>5.1f} h  a  "
          f"{SEEDS*hi*MEDIDO_YOLO_M_MIN/60:>5.1f} h")
    print(f"\n  WARNING: MACs and CPU time overstate the GPU cost for networks with")
    print(f"  heavy high-resolution convolution — the GPU parallelises that pattern")
    print(f"  well. Treat this as a CEILING, and measure the first seed for real")
    print(f"  before assuming the total. `train_unet.py` writes wall_seconds to")
    print(f"  provenance.json, so the first run already gives the true number.")
else:
    print("  no yolo11m-seg to compare against — only the absolute numbers above.")

print(f"\n  (per epoch: {N_TREINO} training images, batch 4 = {N_TREINO//4 + 1} steps)")
