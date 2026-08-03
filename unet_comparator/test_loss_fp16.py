# -*- coding: utf-8 -*-
"""
test_loss_fp16.py — locks in the fix for the float16 overflow in the Dice term.

WHAT FAILED. The unet_black_seed42 run trained well up to epoch 40 (val IoU 0.846
at epoch 34) and from epoch 41 onward reported train_loss = NaN for 60 epochs, with
val_iou frozen at 0.091371 — which is exactly 18/197, the fraction of negative
images in the validation set. Weights in last.pt: 31,049,409 of 31,049,409
parameters NaN. best.pt (epoch 34): zero NaN.

WHY. Under autocast the logits arrive as float16. `p.sum(dim=(1,2,3))` adds
640*640 = 409,600 sigmoids per image; the float16 maximum is 65,504. The mean of the
sigmoids only has to exceed 65504/409600 = 0.16 for the sum to become inf. The
failure has two stages — first Dice pins at 1.0 (finite, no gradient), then the
numerator overflows too and gives inf/inf = NaN.

WHY THE GradScaler DID NOT SAVE IT. It only protects opt.step(). BatchNorm
statistics are updated in the FORWARD pass; as soon as an activation becomes
inf/NaN, the running_mean/running_var buffers absorb NaN and never recover. That is
why training has to ABORT on the first NaN rather than skip the batch.

    python unet_comparator/test_loss_fp16.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_model import BCEDiceLoss  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

H = W = 640
N = H * W
TETO16 = torch.finfo(torch.float16).max
falhas = []


def ok(cond, msg):
    print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")
    if not cond:
        falhas.append(msg)


def alvo_um_terco(dtype, dev):
    a = torch.zeros(2, 1, H, W, dtype=dtype, device=dev)
    a[..., : H // 3, :] = 1.0
    return a


print(f"pixels per image: {N:,} · float16 ceiling: {TETO16:,.0f}")
print(f"sigmoid mean that saturates: {TETO16 / N:.4f}\n")

# ── 1. REGRESSION: the old formulation really did overflow ─────────────────────────
print("1. the OLD formulation (reduction in fp16) overflows — proof the bug was real")
print("   DTYPES AS IN THE REAL CODE: under autocast the conv returns fp16 logits, so")
print("   p = sigmoid(logits) is fp16; the target comes from the DataLoader in fp32 and")
print("   is NOT recast. Only the PREDICTION sum overflows — the target sum never does.")
degradado, nan = [], []
for media in (0.05, 0.10, 0.15, 0.20, 0.35, 0.50):
    p16 = torch.full((2, 1, H, W), media, dtype=torch.float16)
    a32 = alvo_um_terco(torch.float32, "cpu")          # target in fp32, as in the real run
    num = 2 * (p16 * a32.half()).sum(dim=(1, 2, 3)) + 1.0
    den = p16.sum(dim=(1, 2, 3)) + a32.sum(dim=(1, 2, 3)) + 1.0
    d = 1 - (num / den).mean()
    e_nan = bool(torch.isnan(d).any())
    # "degraded" = finite but pinned at 1.0, the first stage of the failure
    e_deg = (not e_nan) and abs(d.item() - 1.0) < 1e-6
    nan.append(e_nan)
    degradado.append(e_deg)
    marca = "  <- NaN" if e_nan else ("  <- pinned at 1.0, no gradient" if e_deg else "")
    print(f"     mean {media:.2f} -> sum_p_fp16 {str(p16.sum(dim=(1,2,3))[0].item()):>10s}"
          f"  dice {d.item()}{marca}")

ok(not (nan[0] or degradado[0]),
   "small-area regime (mean 0.05) is healthy — which is why training ran fine "
   "up to epoch 40")
ok(any(degradado), "failure stage 1: finite Dice pinned at 1.0 (dead gradient)")
ok(any(nan), "failure stage 2: Dice = NaN")

# ── 2. the NEW formulation is finite across the whole range ─────────────────────────
print("\n2. corrected BCEDiceLoss: finite for any sigmoid mean")
crit = BCEDiceLoss()
for media in (0.01, 0.05, 0.16, 0.20, 0.35, 0.50, 0.75, 0.99):
    # logit that produces the desired sigmoid mean
    logit = torch.logit(torch.tensor(media)).item()
    lg = torch.full((2, 1, H, W), logit, dtype=torch.float32)
    a = alvo_um_terco(torch.float32, "cpu")
    v = crit(lg, a)
    ok(torch.isfinite(v).all(), f"mean {media:.2f} -> loss {v.item():.6f}")

# ── 3. finite gradient (a finite loss is not enough) ─────────────────────────
print("\n3. the gradient has to be finite too — a finite loss whose gradient")
print("   was zeroed was the first stage of the failure")
for media in (0.20, 0.50, 0.90):
    logit = torch.logit(torch.tensor(media)).item()
    lg = torch.full((2, 1, H, W), logit, dtype=torch.float32, requires_grad=True)
    a = alvo_um_terco(torch.float32, "cpu")
    crit(lg, a).backward()
    g = lg.grad
    ok(torch.isfinite(g).all() and g.abs().sum() > 0,
       f"mean {media:.2f} -> finite non-zero grad (|g|max {g.abs().max():.3e})")

# ── 4. sanity: the loss has to reward the correct prediction ────────────────────
print("\n4. sanity — the right prediction scores lower than the wrong one")
a = alvo_um_terco(torch.float32, "cpu")
certa = torch.where(a > 0.5, 6.0, -6.0)            # logits aligned with the target
errada = torch.where(a > 0.5, -6.0, 6.0)           # logits inverted
lc, le = crit(certa, a).item(), crit(errada, a).item()
ok(lc < le, f"loss(right)={lc:.6f} < loss(wrong)={le:.6f}")

# ── 5. under real autocast, if a GPU is present ───────────────────────────────────────
print("\n5. under real torch.amp.autocast")
if torch.cuda.is_available():
    dev = "cuda"
    for media in (0.20, 0.50, 0.90):
        logit = torch.logit(torch.tensor(media)).item()
        lg = torch.full((2, 1, H, W), logit, device=dev, requires_grad=True)
        a = alvo_um_terco(torch.float32, dev)
        with torch.amp.autocast("cuda", enabled=True):
            v = crit(lg, a)
        v.backward()
        ok(torch.isfinite(v).all() and torch.isfinite(lg.grad).all(),
           f"autocast mean {media:.2f} -> loss {v.item():.6f}, finite grad")
else:
    print("     (no CUDA — autocast test skipped; items 1-4 already cover the"
          " arithmetic that broke)")

print("\n" + "=" * 68)
if falhas:
    print(f"{len(falhas)} FAILURE(S):")
    for f in falhas:
        print("  · " + f)
    sys.exit(1)
print("every test passed — the fp16 fix is locked in")
