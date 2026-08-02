# -*- coding: utf-8 -*-
"""
diag_lentidao.py — de onde vem a diferenca entre 87 s/epoca (bench) e 256 s (real)?

O bench_loop.py isolou as mudancas do patch e elas custam 2 ms/lote (0,5%). Ou
seja: a lentidao NAO esta no loop de treino. Mas o bench projeta 87 s/epoca e o
treino real gasta 256 s. Como o codigo do lote e o mesmo, a diferenca tem de estar
no que o bench nao reproduz. Duas candidatas:

  1. set_deterministic() — o treino liga cudnn.deterministic=True e
     cudnn.benchmark=False, o que impede o cuDNN de escolher o algoritmo mais
     rapido e de autotunar. O bench so chama torch.manual_seed.
  2. avalia() — roda a cada epoca, FORA do autocast, portanto em fp32.

Mede as duas, separadamente, com o mesmo numero de lotes.

    python unet_comparator/diag_lentidao.py --lotes 25
"""
import argparse
import os
import random
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

from unet_data import WoundDataset, carrega_splits, SEED_AUG   # noqa: E402
from unet_model import UNet, BCEDiceLoss                       # noqa: E402
from train_unet import avalia, set_deterministic               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def solta_determinismo():
    """desfaz o que set_deterministic() liga, para medir o contraste."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def treina_lotes(dl, dev, lotes):
    torch.manual_seed(42)
    modelo = UNet().to(dev)
    crit = BCEDiceLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=5e-4)
    escala = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    modelo.train()
    tempos, i = [], 0
    for x, y, _, _ in dl:
        if i >= lotes + 3:
            break
        t = time.perf_counter()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            perda = crit(modelo(x), y)
        escala.scale(perda).backward()
        escala.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=10.0)
        escala.step(opt)
        escala.update()
        float(perda.detach())
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= 3:
            tempos.append(time.perf_counter() - t)
        i += 1
    return modelo, st.median(tempos), len(tempos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotes", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        sys.exit("ABORTADO: sem CUDA")
    livre, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)} · "
          f"{livre/1e9:.1f} GB livres de {total/1e9:.1f} GB")
    if livre < 4e9:
        sys.exit("ABORTADO: menos de 4 GB livres — a grade esta rodando?")

    splits = carrega_splits("data.yaml")

    def faz_loaders():
        ds = WoundDataset(splits["train"], args.imgsz, treino=True, fill=0, seed=42)
        ds.set_epoca(1)
        g = torch.Generator()
        g.manual_seed(SEED_AUG)
        dtr = DataLoader(ds, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True, generator=g)
        dva = DataLoader(WoundDataset(splits["val"], args.imgsz, treino=False, fill=0),
                         batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
        return ds, dtr, dva

    resultados = {}
    for rotulo, liga_det in (("SEM determinismo (como o bench)", False),
                             ("COM determinismo (como o treino real)", True)):
        if liga_det:
            set_deterministic(42)
        else:
            random.seed(42), np.random.seed(42), torch.manual_seed(42)
            solta_determinismo()
        ds, dtr, dva = faz_loaders()
        modelo, med, n = treina_lotes(dtr, dev, args.lotes)
        n_lotes = (len(ds) + args.batch - 1) // args.batch
        resultados[rotulo] = (med, n_lotes)
        print(f"\n{rotulo}")
        print(f"  treino: {med*1000:7.1f} ms/lote (n={n}) -> "
              f"{med*n_lotes:6.1f} s de treino por epoca")

        t = time.perf_counter()
        iou, dice = avalia(modelo, dva, dev)
        torch.cuda.synchronize()
        tv = time.perf_counter() - t
        print(f"  validacao (fp32, {len(dva.dataset)} imgs): {tv:6.1f} s")
        print(f"  EPOCA COMPLETA estimada: {med*n_lotes + tv:6.1f} s")
        resultados[rotulo] = (med * n_lotes, tv)
        del modelo
        torch.cuda.empty_cache()

    print("\n" + "=" * 66)
    (t_sem, v_sem), (t_com, v_com) = resultados.values()
    print(f"custo do determinismo no treino: {t_com - t_sem:+6.1f} s/epoca "
          f"({t_com/t_sem:.2f}x)")
    print(f"custo da validacao em fp32     : {(v_sem + v_com)/2:6.1f} s/epoca")
    print(f"epoca completa com determinismo: {t_com + v_com:6.1f} s "
          f"(observado no treino real: ~256 s)")
    falta = 256 - (t_com + v_com)
    if abs(falta) < 40:
        print("\nEXPLICADO — determinismo + validacao fp32 dao conta da diferenca.")
    else:
        print(f"\nNAO EXPLICADO — faltam {falta:.0f} s. A causa esta em outro lugar.")


if __name__ == "__main__":
    main()
