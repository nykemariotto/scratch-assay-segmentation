# -*- coding: utf-8 -*-
"""
varredura_limiar.py — a U-Net ao longo do limiar, para comparar com o YOLO no
MESMO ponto de operação.

O PROBLEMA. O braço YOLO foi avaliado em `conf = 0,80`, o padrão da interface
publicada; a U-Net não tem confiança, só `sigmoid > 0,50`. Comparar 0,80 com 0,50
compara duas escolhas arbitrárias, não duas arquiteturas. Os regimes nativos são
qualitativamente distintos:

    YOLO M @0,80    recall  87,5%   precisão 100,0%   (nunca inventa, silencia em 12%)
    U-Net @0,50     recall  99,8%   precisão  97,1%   (quase nunca perde, inventa em 21% dos vazios)

O QUE ESTE SCRIPT FAZ. Um único forward por imagem; a partir do mapa de
probabilidade, aplica TODOS os limiares da grade. Assim a varredura custa uma
passada de GPU, não uma por limiar.

Para cada (run, limiar) grava, por imagem:
  · area_px na resolução original (desfaz o letterbox, como o stage3/predict_areas.py)
  · IoU contra a máscara de referência, no espaço 640x640 — a mesma convenção do
    `avalia()` do train_unet.py
  · se a máscara ficou vazia

DEFINIÇÃO DE RECALL/PRECISÃO. Nível de IMAGEM, que é o que vale para os dois
braços: a U-Net não produz instâncias pontuadas, então o recall de detecção da
Table 2 não se aplica a ela.
  recall   = imagens COM ferida anotada em que saiu máscara não-vazia
  precisão = imagens com máscara não-vazia que de fato têm ferida

Saída: `stage3/unet_varredura.csv` (por imagem) e `stage3/unet_varredura_resumo.csv`
(por run e limiar).

    python unet_comparator/varredura_limiar.py
"""
import argparse
import csv
import glob
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
os.chdir(os.path.dirname(AQUI))

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402

from unet_data import WoundDataset, carrega_splits, desfaz_letterbox   # noqa: E402
from unet_model import UNet                                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUNS = os.path.join("runs", "segment", "unet_comparator")
LIMIARES = [round(x, 3) for x in
            (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
             0.90, 0.95, 0.98, 0.99, 0.995, 0.999)]


def grade_ocupada():
    """não roda junto com um treino: 234 imgs x 5 runs disputariam a GPU."""
    import time
    for p in glob.glob(os.path.join(RUNS, "*", "results.csv")):
        d = os.path.dirname(p)
        if not os.path.isfile(os.path.join(d, "COMPLETED.json")) and \
           time.time() - os.path.getmtime(p) < 900:
            return os.path.basename(d)
    return None


@torch.no_grad()
def varre(pesos, dl, dev):
    ck = torch.load(pesos, map_location=dev, weights_only=False)
    m = UNet().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    linhas = []
    for x, y, nomes, lbs in dl:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        prob = torch.sigmoid(m(x).float())            # UM forward por imagem
        for i, nome in enumerate(nomes):
            gt = y[i, 0] > 0.5
            gt_soma = float(gt.sum())
            for t in LIMIARES:
                p = prob[i, 0] > t
                inter = float((p & gt).sum())
                uni = float((p | gt).sum())
                # IoU no espaço 640x640, mesma convenção do avalia()
                iou = 1.0 if uni == 0 else inter / uni
                # área na resolução original, mesma convenção do stage3/predict_areas.py
                mm = desfaz_letterbox(p.byte().cpu().numpy(),
                                      {k: (v[i].item() if torch.is_tensor(v) else v[i])
                                       for k, v in lbs.items()})
                linhas.append({"arquivo": nome, "limiar": t,
                               "area_px": int(mm.sum()),
                               "iou": round(iou, 6),
                               "vazia": int(p.sum() == 0),
                               "tem_ferida": int(gt_soma > 0)})
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ocup = grade_ocupada()
    if ocup and not args.force:
        sys.exit(f"ABORTADO: a grade parece ativa ({ocup}). Esperar, ou --force.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    splits = carrega_splits("data.yaml")
    ds = WoundDataset(splits[args.split], 640, treino=False, fill=0)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2,
                    pin_memory=(dev == "cuda"))

    alvos = sorted(d for d in glob.glob(os.path.join(RUNS, "*"))
                   if os.path.isfile(os.path.join(d, "COMPLETED.json")))
    if not alvos:
        sys.exit("nenhum run da U-Net concluído")
    print(f"{len(alvos)} run(s) · {len(ds)} imagens do {args.split} · "
          f"{len(LIMIARES)} limiares · device {dev}\n")

    todas = []
    for i, d in enumerate(alvos, 1):
        nome = os.path.basename(d)
        import time
        t = time.time()
        for L in varre(os.path.join(d, "best.pt"), dl, dev):
            L["run"] = nome
            todas.append(L)
        print(f"  [{i}/{len(alvos)}] {nome}  ({time.time()-t:.0f}s)")

    os.makedirs("stage3", exist_ok=True)
    p1 = os.path.join("stage3", "unet_varredura.csv")
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "limiar", "arquivo", "area_px",
                                          "iou", "vazia", "tem_ferida"])
        w.writeheader()
        for L in todas:
            w.writerow({k: L[k] for k in w.fieldnames})

    # resumo por (run, limiar) — recall e precisão no nível de IMAGEM
    resumo = []
    for nome in sorted({L["run"] for L in todas}):
        for t in LIMIARES:
            sub = [L for L in todas if L["run"] == nome and L["limiar"] == t]
            pos = [L for L in sub if L["tem_ferida"]]
            neg = [L for L in sub if not L["tem_ferida"]]
            tp = sum(1 for L in pos if not L["vazia"])
            fp = sum(1 for L in neg if not L["vazia"])
            fn = len(pos) - tp
            rec = tp / len(pos) if pos else float("nan")
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            ious = [L["iou"] for L in pos]
            resumo.append({"run": nome, "limiar": t, "n_pos": len(pos),
                           "n_neg": len(neg), "tp": tp, "fp": fp, "fn": fn,
                           "recall": round(rec, 6), "precisao": round(prec, 6),
                           "iou_medio_pos": round(float(np.mean(ious)), 6)})
    p2 = os.path.join("stage3", "unet_varredura_resumo.csv")
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumo[0].keys()))
        w.writeheader()
        w.writerows(resumo)

    print(f"\ngravados {p1} e {p2}")
    print("\nmédia sobre os 5 seeds, por limiar (nível de imagem):")
    print(f"  {'limiar':>7} {'recall':>8} {'precisão':>9} {'IoU pos':>8} "
          f"{'FN':>5} {'FP':>5}")
    for t in LIMIARES:
        sub = [r for r in resumo if r["limiar"] == t]
        print(f"  {t:>7.3f} {np.mean([r['recall'] for r in sub]):>8.4f} "
              f"{np.mean([r['precisao'] for r in sub]):>9.4f} "
              f"{np.mean([r['iou_medio_pos'] for r in sub]):>8.4f} "
              f"{np.mean([r['fn'] for r in sub]):>5.1f} "
              f"{np.mean([r['fp'] for r in sub]):>5.1f}")
    print("""
Para casar o ponto de operação: procure o limiar cujo `recall` iguala o do YOLO de
referência no nível de imagem (M @ conf 0,80 = 0,875) e compare a precisão e o IoU
ali. Os regimes nativos (0,50 para a U-Net, 0,80 para o YOLO) continuam sendo
reportados — casar o ponto responde "quem é melhor a igual sensibilidade", não
substitui a descrição de como cada um opera por padrão.""")


if __name__ == "__main__":
    main()
