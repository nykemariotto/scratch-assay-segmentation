# -*- coding: utf-8 -*-
"""
diag_fases.py — onde estão os ~250 s que minhas medições isoladas não acham.

O script real gasta 339-382 s por época. Medido em separado: treino 91,7 s,
validação 15,1 s, spawn dos workers 6,9 s — soma 114 s. A diferença é sistemática e
reprodutível, então está em algo que só aparece com as partes juntas.

Diferenças entre os meus benchmarks e o train_unet.py:
  · o benchmark usava `persistent_workers=True`; o real, False;
  · o benchmark tinha UM DataLoader; o real tem DOIS vivos ao mesmo tempo, cada um
    com 2 workers e `pin_memory=True` (portanto duas threads de pinning);
  · o dataset NÃO faz cache — `cv2.imread` a cada item, 932 PNGs de 2452x2056 por
    época. Com workers persistentes e page cache quente, isso fica barato; sem eles,
    cada época recomeça com processos novos.

Este script replica a estrutura REAL e cronometra as fases separadamente.

    python unet_comparator/diag_fases.py --epochs 2
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
from train_unet import avalia, set_deterministic               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        sys.exit("ABORTADO: sem CUDA")
    livre, _ = torch.cuda.mem_get_info()
    if livre < 4e9:
        sys.exit("ABORTADO: menos de 4 GB livres — a grade está rodando?")

    set_deterministic(42)
    splits = carrega_splits("data.yaml")
    ds_tr = WoundDataset(splits["train"], 640, treino=True, fill=0, seed=42)
    ds_va = WoundDataset(splits["val"], 640, treino=False, fill=0)
    g = torch.Generator()
    g.manual_seed(SEED_AUG)
    # EXATAMENTE como o train_unet.py: dois loaders vivos, sem persistent_workers
    dl_tr = DataLoader(ds_tr, batch_size=4, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=False, generator=g)
    dl_va = DataLoader(ds_va, batch_size=4, shuffle=False, num_workers=args.workers,
                       pin_memory=True)

    modelo = UNet().to(dev)
    crit = BCEDiceLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=5e-4)
    escala = torch.amp.GradScaler("cuda", enabled=True)
    print(f"treino {len(ds_tr)} · val {len(ds_va)} · workers={args.workers}\n")

    for ep in range(1, args.epochs + 1):
        ds_tr.set_epoca(ep)
        modelo.train()
        t_ep = time.perf_counter()

        t = time.perf_counter()
        it = iter(dl_tr)
        t_spawn = time.perf_counter() - t

        t_dado, t_gpu, n = 0.0, 0.0, 0
        t_prev = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            try:
                x, y, _, _ = next(it)
            except StopIteration:
                break
            t_dado += time.perf_counter() - t0            # espera pelo lote

            t0 = time.perf_counter()
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
            t_gpu += time.perf_counter() - t0
            n += 1
        del it
        t_treino = time.perf_counter() - t_ep

        t = time.perf_counter()
        iou, dice = avalia(modelo, dl_va, dev)
        torch.cuda.synchronize()
        t_val = time.perf_counter() - t

        t = time.perf_counter()
        torch.save({"model": modelo.state_dict(), "epoch": ep}, "_diag_best.pt")
        t_save = time.perf_counter() - t

        total = time.perf_counter() - t_ep
        outro = t_treino - t_spawn - t_dado - t_gpu
        print(f"época {ep}: TOTAL {total:6.1f}s")
        print(f"   spawn do iterador      {t_spawn:6.1f}s")
        print(f"   ESPERA POR DADO        {t_dado:6.1f}s  ({100*t_dado/total:4.1f}%)  "
              f"[{n} lotes, {1000*t_dado/n:.0f} ms/lote]")
        print(f"   cômputo em GPU         {t_gpu:6.1f}s  ({100*t_gpu/total:4.1f}%)  "
              f"[{1000*t_gpu/n:.0f} ms/lote]")
        print(f"   outro (laço)           {outro:6.1f}s")
        print(f"   validação              {t_val:6.1f}s")
        print(f"   torch.save (124 MB)    {t_save:6.1f}s")
        print()

    if os.path.isfile("_diag_best.pt"):
        os.remove("_diag_best.pt")
    print("Se ESPERA POR DADO dominar, o gargalo é a decodificação dos PNG de 5 MP")
    print("nos workers — 932 por época, sem cache, com processos recriados a cada")
    print("época. O benchmark anterior não via isso porque usava workers persistentes")
    print("e page cache quente.")


if __name__ == "__main__":
    main()
