# -*- coding: utf-8 -*-
"""
diag_validacao2.py — por que avalia() leva 267 s DEPOIS do treino e 15 s sozinha?

O diag_fases mostrou que a validacao consome 73% da epoca (267 s de 367 s), com a
espera por dado em 0,1%. Mas o diag_validacao mediu a MESMA funcao, nas MESMAS 197
imagens, em 15,1 s.

A unica diferenca e o estado da GPU: la a validacao rodava sozinha, com o alocador
limpo; aqui roda logo apos uma epoca de treino, com o cache do alocador cheio de
blocos grandes. E ela e fp32, FORA do autocast, entao pede o dobro de memoria por
amostra que o treino. Se o alocador tiver de devolver blocos ao driver e pedir
outros, cada `cudaMalloc`/`cudaFree` serializa — o que bate com a GPU marcando 100%
de utilizacao puxando so 53 W de 160 W.

Mede quatro condicoes, sempre APOS uma epoca de treino real:

  A  como esta hoje                         (fp32, alocador cheio)
  B  com torch.cuda.empty_cache() antes     (devolve o cache uma vez, de proposito)
  C  sob autocast fp16                      (metade da memoria de ativacao)
  D  empty_cache + autocast                 (as duas)

Se B ou C resolverem, a epoca cai de ~350 s para perto de 110 s, e o retreino dos
cinco seeds cai de ~48 h para ~15 h.

    python unet_comparator/diag_validacao2.py
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
    """avalia() com o forward sob autocast fp16; o resto identico."""
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
        sys.exit("ABORTADO: sem CUDA")
    livre, tot = torch.cuda.mem_get_info()
    if livre < 4e9:
        sys.exit("ABORTADO: menos de 4 GB livres — a grade está rodando?")
    print(f"GPU: {livre/1e9:.1f} GB livres de {tot/1e9:.1f}\n")

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

    print("rodando UMA época de treino para deixar o alocador no estado real…")
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
    print(f"treino: {time.perf_counter()-t:.1f}s\n")

    def reservado():
        return torch.cuda.memory_reserved() / 1e9

    CONDS = [("A  como está hoje (fp32, alocador cheio)", False, False),
             ("B  empty_cache() antes", True, False),
             ("C  autocast fp16 no forward", False, True),
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
              f"IoU {iou:.6f}  reservado {r0:.2f}->{reservado():.2f} GB")

    print(f"""
LEITURA. Se B ou C derrubarem o tempo, o gargalo é o alocador, não o cômputo.
A época hoje é ~350 s: 95 s de treino + 267 s de validação. Com a validação em
~15 s, ela cai para ~115 s — e os 5 seeds saem de ~48 h para ~16 h.

Atenção: a condição C muda o número. Compare os IoU acima: se diferirem, a escolha
entre B e C deixa de ser só de desempenho.""")


if __name__ == "__main__":
    main()
