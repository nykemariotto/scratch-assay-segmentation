# -*- coding: utf-8 -*-
"""
diag_validacao.py — a validacao e mesmo lenta, ou minha medicao teve artefato?

O diag_lentidao.py mediu validacao em 138 s com determinismo e 19 s sem. Mas o run
ANTIGO usava exatamente o mesmo set_deterministic() e o mesmo avalia() em fp32, e
gastava 105 s na EPOCA INTEIRA — treino incluso. Os numeros nao fecham, o que
sugere artefato: cada chamada mede a PRIMEIRA execucao, que carrega escolha de
algoritmo do cuDNN e partida dos workers do DataLoader.

Aqui a validacao e repetida 3x na mesma configuracao. Se a 1a for lenta e as
seguintes rapidas, era aquecimento e nao ha problema nenhum a resolver.

Tres configuracoes:
  fp32-det   como esta hoje (fora do autocast, cudnn.deterministic=True)
  fp32-nodet fora do autocast, sem determinismo
  fp16-det   DENTRO do autocast, deterministic=True — como o treino, e como o
             Ultralytics valida no braco YOLO

    python unet_comparator/diag_validacao.py
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
    """copia fiel de avalia(), com o autocast como parametro."""
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
        sys.exit("ABORTADO: sem CUDA")
    livre, _ = torch.cuda.mem_get_info()
    if livre < 4e9:
        sys.exit("ABORTADO: menos de 4 GB livres — a grade esta rodando?")
    print(f"{torch.cuda.get_device_name(0)} · {livre/1e9:.1f} GB livres\n")

    splits = carrega_splits("data.yaml")
    CFGS = [("fp32-det   (como esta hoje)", True, False),
            ("fp32-nodet (sem determinismo)", False, False),
            ("fp16-det   (dentro do autocast)", True, True)]

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
              + f"   (mediana das repeticoes 2-3: {st.median(ts[1:]):.1f}s)")
        print(f"  IoU   : " + "  ".join(f"{x:.6f}" for x in ious))
        if max(ious) - min(ious) > 1e-9:
            print("  ATENCAO: IoU variou entre repeticoes da MESMA configuracao")
        print()
        del modelo
        torch.cuda.empty_cache()

    print("Se a 1a repeticao for lenta e as demais rapidas, era aquecimento —")
    print("o treino real paga isso UMA vez, nao a cada epoca, e nao ha o que corrigir.")


if __name__ == "__main__":
    main()
