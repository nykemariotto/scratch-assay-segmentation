# -*- coding: utf-8 -*-
"""
bench_loop.py — mede o custo de cada mudanca que fiz no loop de treino.

MOTIVO. Depois do patch do fp16 a epoca passou de ~105 s para ~256 s (2,4x). Tenho
um suspeito — mudei a posicao do ponto de sincronizacao, que antes vinha DEPOIS do
backward e agora vem ANTES — mas a aritmetica nao fecha: a diferenca e de 0,69 s
por lote, e um sync nao adiciona trabalho. Entao meco, em vez de adivinhar.

Quatro variantes, mesmo modelo, mesmos dados, mesmo numero de lotes:

  A  original      sync depois do step, sem clipping
  B  sync antes    sync antes do backward, sem clipping   <- isola o sync
  C  clipping      sync depois do step, com unscale_+clip <- isola o clipping
  D  atual         sync antes + clipping                  <- o que esta rodando

Se D ≈ A, a lentidao vem de outro lugar (contencao com o desktop, throttling) e
mexer no loop nao resolve.

NAO RODAR COM A GRADE ATIVA — sao 8 GB de GPU e o teste competiria por memoria.

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
              f"{livre/1e9:.1f} GB livres de {total/1e9:.1f} GB")
        if livre < 4e9:
            sys.exit("ABORTADO: menos de 4 GB livres — a grade ainda esta rodando?")

    splits = carrega_splits("data.yaml")
    ds = WoundDataset(splits["train"], args.imgsz, treino=True, fill=0, seed=42)
    ds.set_epoca(1)
    g = torch.Generator()
    g.manual_seed(SEED_AUG)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=(dev == "cuda"), generator=g)

    NOMES = {"A": "original     (sync depois, sem clip)",
             "B": "sync antes   (sync antes,  sem clip)",
             "C": "clipping     (sync depois, com clip)",
             "D": "atual        (sync antes,  com clip)"}
    res = {}
    for v in ("A", "B", "C", "D"):
        t = roda(v, dl, dev, args.lotes)
        res[v] = st.median(t)
        print(f"  {v} · {NOMES[v]:38s} mediana {res[v]*1000:7.1f} ms/lote "
              f"(n={len(t)})")

    base = res["A"]
    print(f"\nrelativo ao original (A = 1,00x):")
    for v in ("B", "C", "D"):
        print(f"  {v} {NOMES[v][:12]:14s} {res[v]/base:5.2f}x  "
              f"({(res[v]-base)*1000:+7.1f} ms/lote)")

    n_lotes_epoca = (len(ds) + args.batch - 1) // args.batch
    print(f"\nprojecao por epoca ({n_lotes_epoca} lotes):")
    for v in ("A", "D"):
        print(f"  {v}: {res[v]*n_lotes_epoca:6.1f} s/epoca -> "
              f"{res[v]*n_lotes_epoca*100*5/3600:5.1f} h para os 5 seeds")

    d_sync, d_clip = res["B"] - base, res["C"] - base
    print(f"\nveredito:")
    print(f"  custo do sync antes : {d_sync*1000:+7.1f} ms/lote")
    print(f"  custo do clipping   : {d_clip*1000:+7.1f} ms/lote")
    if res["D"] / base < 1.25:
        print("  NENHUMA das duas explica a lentidao — a causa esta fora do loop\n"
              "  (contencao com o desktop, throttling termico, estado da GPU).\n"
              "  Mexer no loop NAO resolveria; reverter seria mudar por mudar.")
    else:
        pior = "sync antes" if d_sync > d_clip else "clipping"
        print(f"  o custo esta em: {pior}. Vale corrigir antes de seguir.")


if __name__ == "__main__":
    main()
