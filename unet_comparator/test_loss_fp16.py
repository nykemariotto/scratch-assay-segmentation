# -*- coding: utf-8 -*-
"""
test_loss_fp16.py — trava a correcao do estouro de float16 no termo Dice.

O QUE FALHOU. O run unet_black_seed42 treinou bem ate a epoca 40 (IoU val 0,846
na 34) e da epoca 41 em diante reportou train_loss = NaN por 60 epocas, com
val_iou congelado em 0,091371 — que e exatamente 18/197, a fracao de imagens
negativas do conjunto de validacao. Pesos do last.pt: 31.049.409 de 31.049.409
parametros NaN. best.pt (epoca 34): zero NaN.

POR QUE. Sob autocast os logits chegam em float16. `p.sum(dim=(1,2,3))` soma
640*640 = 409.600 sigmoids por imagem; o maximo do float16 e 65.504. Basta a
media dos sigmoids passar de 65504/409600 = 0,16 e a soma vira inf. A falha tem
duas etapas — primeiro o Dice trava em 1,0 (finito, sem gradiente), depois o
numerador tambem estoura e da inf/inf = NaN.

POR QUE O GradScaler NAO SALVOU. Ele so protege o opt.step(). As estatisticas do
BatchNorm sao atualizadas no FORWARD; assim que uma ativacao vira inf/NaN os
buffers running_mean/running_var absorvem NaN e nao voltam. Por isso o treino
tem de ABORTAR no primeiro NaN, nao pular o lote.

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
    print(f"  [{'OK ' if cond else 'FALHA'}] {msg}")
    if not cond:
        falhas.append(msg)


def alvo_um_terco(dtype, dev):
    a = torch.zeros(2, 1, H, W, dtype=dtype, device=dev)
    a[..., : H // 3, :] = 1.0
    return a


print(f"pixels por imagem: {N:,} · teto do float16: {TETO16:,.0f}")
print(f"media de sigmoid que satura: {TETO16 / N:.4f}\n")

# ── 1. REGRESSAO: a formulacao antiga estourava mesmo ─────────────────────────
print("1. a formulacao ANTIGA (reducao em fp16) estoura — prova de que o bug era real")
print("   DTYPES COMO NO CODIGO REAL: sob autocast o conv devolve logits fp16, entao")
print("   p = sigmoid(logits) e fp16; o alvo vem do DataLoader em fp32 e NAO e")
print("   recastado. So a soma da PREDICAO estoura — a do alvo, nunca.")
degradado, nan = [], []
for media in (0.05, 0.10, 0.15, 0.20, 0.35, 0.50):
    p16 = torch.full((2, 1, H, W), media, dtype=torch.float16)
    a32 = alvo_um_terco(torch.float32, "cpu")          # alvo em fp32, como no real
    num = 2 * (p16 * a32.half()).sum(dim=(1, 2, 3)) + 1.0
    den = p16.sum(dim=(1, 2, 3)) + a32.sum(dim=(1, 2, 3)) + 1.0
    d = 1 - (num / den).mean()
    e_nan = bool(torch.isnan(d).any())
    # "degradado" = finito mas pregado em 1.0, o primeiro estagio da falha
    e_deg = (not e_nan) and abs(d.item() - 1.0) < 1e-6
    nan.append(e_nan)
    degradado.append(e_deg)
    marca = "  <- NaN" if e_nan else ("  <- pregado em 1.0, sem gradiente" if e_deg else "")
    print(f"     media {media:.2f} -> soma_p_fp16 {str(p16.sum(dim=(1,2,3))[0].item()):>10s}"
          f"  dice {d.item()}{marca}")

ok(not (nan[0] or degradado[0]),
   "regime de area pequena (media 0.05) e sao — por isso o treino correu bem "
   "ate a epoca 40")
ok(any(degradado), "estagio 1 da falha: Dice finito pregado em 1.0 (gradiente morto)")
ok(any(nan), "estagio 2 da falha: Dice = NaN")

# ── 2. a formulacao NOVA e finita em todo o intervalo ─────────────────────────
print("\n2. BCEDiceLoss corrigida: finita para qualquer media de sigmoid")
crit = BCEDiceLoss()
for media in (0.01, 0.05, 0.16, 0.20, 0.35, 0.50, 0.75, 0.99):
    # logit que produz a media de sigmoid desejada
    logit = torch.logit(torch.tensor(media)).item()
    lg = torch.full((2, 1, H, W), logit, dtype=torch.float32)
    a = alvo_um_terco(torch.float32, "cpu")
    v = crit(lg, a)
    ok(torch.isfinite(v).all(), f"media {media:.2f} -> loss {v.item():.6f}")

# ── 3. gradiente finito (nao basta a loss ser finita) ─────────────────────────
print("\n3. o gradiente tambem precisa ser finito — a loss finita com gradiente")
print("   zerado foi o primeiro estagio da falha")
for media in (0.20, 0.50, 0.90):
    logit = torch.logit(torch.tensor(media)).item()
    lg = torch.full((2, 1, H, W), logit, dtype=torch.float32, requires_grad=True)
    a = alvo_um_terco(torch.float32, "cpu")
    crit(lg, a).backward()
    g = lg.grad
    ok(torch.isfinite(g).all() and g.abs().sum() > 0,
       f"media {media:.2f} -> grad finito e nao-nulo (|g|max {g.abs().max():.3e})")

# ── 4. sanidade: a loss precisa premiar a predicao correta ────────────────────
print("\n4. sanidade — predicao certa tem loss menor que predicao errada")
a = alvo_um_terco(torch.float32, "cpu")
certa = torch.where(a > 0.5, 6.0, -6.0)            # logits alinhados ao alvo
errada = torch.where(a > 0.5, -6.0, 6.0)           # logits invertidos
lc, le = crit(certa, a).item(), crit(errada, a).item()
ok(lc < le, f"loss(certa)={lc:.6f} < loss(errada)={le:.6f}")

# ── 5. sob autocast real, se houver GPU ───────────────────────────────────────
print("\n5. sob torch.amp.autocast real")
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
           f"autocast media {media:.2f} -> loss {v.item():.6f}, grad finito")
else:
    print("     (sem CUDA — teste de autocast pulado; os itens 1-4 ja cobrem a"
          " aritmetica que quebrou)")

print("\n" + "=" * 68)
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print("  · " + f)
    sys.exit(1)
print("todos os testes passaram — a correcao do fp16 esta travada")
