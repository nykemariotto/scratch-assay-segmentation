# -*- coding: utf-8 -*-
"""
diag_epoca_inteira.py — a GPU esta esperando dado?

Treino + validacao medidos em lotes curtos dao ~101 s/epoca, que bate com o run
antigo. Mas o run novo gastou 240-322 s. A unica diferenca entre a medicao e o
real e o NUMERO DE LOTES: medi 25-30, a epoca tem 233.

Hipotese: com num_workers=2 a fila de prefetch aguenta algumas dezenas de lotes,
mas ao longo de 233 os workers ficam para tras, a fila esvazia e a GPU passa a
esperar a augmentation, que roda na CPU.

Roda uma epoca inteira e compara a taxa dos primeiros 30 lotes com a dos ultimos
30. Se degradar, o gargalo e dado, nao GPU — e a correcao e aumentar workers /
prefetch, nao mexer no modelo.

    python unet_comparator/diag_epoca_inteira.py
    python unet_comparator/diag_epoca_inteira.py --workers 6
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
from train_unet import set_deterministic                       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        sys.exit("ABORTADO: sem CUDA")
    livre, _ = torch.cuda.mem_get_info()
    if livre < 4e9:
        sys.exit("ABORTADO: menos de 4 GB livres — a grade esta rodando?")
    print(f"{torch.cuda.get_device_name(0)} · {livre/1e9:.1f} GB livres · "
          f"workers={args.workers} · CPUs={os.cpu_count()}\n")

    set_deterministic(42)
    splits = carrega_splits("data.yaml")
    ds = WoundDataset(splits["train"], 640, treino=True, fill=0, seed=42)
    ds.set_epoca(1)
    g = torch.Generator()
    g.manual_seed(SEED_AUG)
    kw = {}
    if args.workers > 0 and args.prefetch:
        kw["prefetch_factor"] = args.prefetch
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=(args.workers > 0), generator=g, **kw)

    modelo = UNet().to(dev)
    crit = BCEDiceLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=5e-4)
    escala = torch.amp.GradScaler("cuda", enabled=True)
    modelo.train()

    t_lote, t_espera = [], []
    t0 = time.perf_counter()
    t_fim_anterior = time.perf_counter()
    for i, (x, y, _, _) in enumerate(dl):
        # tempo entre terminar o lote anterior e receber este = espera por dado
        t_espera.append(time.perf_counter() - t_fim_anterior)
        t = time.perf_counter()
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
        t_lote.append(time.perf_counter() - t)
        t_fim_anterior = time.perf_counter()
        if (i + 1) % 50 == 0:
            print(f"  lote {i+1:3d}/{len(dl)}  "
                  f"gpu {st.median(t_lote[-50:])*1000:6.1f} ms  "
                  f"espera_dado {st.median(t_espera[-50:])*1000:6.1f} ms")
    total = time.perf_counter() - t0

    n = len(t_lote)
    p, u = slice(0, 30), slice(-30, None)
    print(f"\nepoca completa: {total:.1f} s em {n} lotes\n")
    print(f"{'':22s} {'primeiros 30':>14s} {'ultimos 30':>14s}")
    print(f"{'tempo de GPU/lote':22s} {st.median(t_lote[p])*1000:11.1f} ms "
          f"{st.median(t_lote[u])*1000:11.1f} ms")
    print(f"{'espera por dado/lote':22s} {st.median(t_espera[p])*1000:11.1f} ms "
          f"{st.median(t_espera[u])*1000:11.1f} ms")

    soma_gpu, soma_esp = sum(t_lote), sum(t_espera)
    print(f"\ntotal em GPU        : {soma_gpu:6.1f} s ({100*soma_gpu/total:4.1f}%)")
    print(f"total esperando dado: {soma_esp:6.1f} s ({100*soma_esp/total:4.1f}%)")
    print()
    if soma_esp > 0.25 * total:
        print(f"GARGALO E DADO. A GPU ficou {100*soma_esp/total:.0f}% do tempo ociosa\n"
              f"esperando a augmentation na CPU. Corrige-se com mais workers /\n"
              f"prefetch — nao mexendo no modelo. Teste: --workers 6")
    else:
        print("A GPU nao esta esperando dado — o gargalo e computo mesmo.")


if __name__ == "__main__":
    main()
