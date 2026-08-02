# -*- coding: utf-8 -*-
"""
stage3/iou_por_imagem.py — IoU de máscara por imagem, para YOLO e U-Net, na MESMA base.

MOTIVO. A varredura da U-Net (`unet_comparator/varredura_limiar.py`) mostrou que em
limiar 0,999 ela chega a recall 0,873 e precisão 1,000 — praticamente idênticos ao
YOLO M em conf 0,80 (0,875 e 1,000). É o ponto de operação casado. Mas faltava o
IoU do YOLO: os registros do `stage3/eval_test.py` guardam casamento para mAP, e o
`stage3/predict_areas.py` guardou área — nenhum guardou IoU de máscara.

DECISÃO DE MÉTODO. Tudo é medido na RESOLUÇÃO ORIGINAL (2452 x 2056), com o padrão
de referência rasterizado ali mesmo por `unet_data.rasteriza`. Não em 640 x 640.

O motivo é evitar uma armadilha silenciosa: o Ultralytics faz o próprio letterbox,
e o `unet_data.letterbox` faz o dele. Se as duas convenções de padding diferirem em
um pixel ou no arredondamento, o IoU dos dois braços passaria a ser medido em
espaços ligeiramente diferentes — e a comparação inteira ficaria enviesada por um
detalhe de implementação, não por uma diferença de modelo. Na resolução original
não há convenção a conciliar: é o espaço da imagem como ela foi adquirida.

Consequência a declarar: o IoU aqui NÃO é numericamente igual ao `iou` da varredura,
que foi calculado em 640 x 640. Este script recalcula os dois braços, para que os
números comparados venham do mesmo lugar.

    python stage3/iou_por_imagem.py
"""
import argparse
import csv
import glob
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "unet_comparator"))
os.chdir(RAIZ)

import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

from unet_data import (carrega_splits, desfaz_letterbox, letterbox,   # noqa: E402
                       rasteriza, rotulo_de)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

YOLO_ROOT = os.path.join("runs", "segment", "runs_revision")
UNET_ROOT = os.path.join("runs", "segment", "unet_comparator")
# a configuracao de REFERENCIA do desenho single-variable
YOLO_CFG = "yolo11m-seg_black_coco"
LIMIARES_UNET = [0.50, 0.999]     # nativo e o que casa o recall do YOLO


def iou_de(pred, gt):
    """IoU entre duas máscaras booleanas. Imagem sem ferida e sem predição = 1,0,
    a mesma convenção do avalia() e da varredura."""
    inter = int(np.logical_and(pred, gt).sum())
    uni = int(np.logical_or(pred, gt).sum())
    return 1.0 if uni == 0 else inter / uni


def gt_original(f):
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    return rasteriza(rotulo_de(f), w, h) > 0, w, h


def yolo_masks(pesos, imagens, conf, imgsz, padding):
    import padding_patch
    padding_patch.apply(padding)
    from ultralytics import YOLO
    m = YOLO(pesos)
    out = []
    for f in imagens:
        r = m.predict(f, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]
        h, w = r.orig_shape
        if r.masks is None or len(r.masks) == 0:
            uni = np.zeros((h, w), bool)
        else:
            md = r.masks.data.cpu().numpy() > 0.5
            uni = np.any(md, axis=0)
            if uni.shape != (h, w):
                uni = cv2.resize(uni.astype(np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(uni)
    del m
    return out


def unet_masks(pesos, imagens, imgsz, limiares):
    from unet_model import UNet
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(pesos, map_location=dev, weights_only=False)
    fill = 0 if ck.get("padding", "black") == "black" else 255
    m = UNet().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    out = {t: [] for t in limiares}
    with torch.no_grad():
        for f in imagens:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            lb_img, _, lb = letterbox(img, np.zeros((h, w), np.uint8), imgsz, fill)
            x = torch.from_numpy(lb_img.transpose(2, 0, 1).copy()).float().div_(255)
            prob = torch.sigmoid(m(x.unsqueeze(0).to(dev)).float())[0, 0]
            for t in limiares:
                p = (prob > t).byte().cpu().numpy()
                out[t].append(desfaz_letterbox(p, lb) > 0)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from eval_test import grade_ocupada
    ocup = grade_ocupada()
    if ocup:
        sys.exit(f"ABORTADO: a grade parece ativa ({ocup}).")

    d = carrega_splits("data.yaml")["test"]
    imagens = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                     for f in glob.glob(os.path.join(d, e)))
    print(f"test: {len(imagens)} imagens · YOLO conf={args.conf} · "
          f"U-Net limiares {LIMIARES_UNET}\n")

    print("rasterizando o padrão de referência na resolução original…")
    gts, dims = [], []
    for f in imagens:
        g, w, h = gt_original(f)
        gts.append(g)
        dims.append((w, h))
    n_pos = sum(1 for g in gts if g.any())
    print(f"  {n_pos} imagens com ferida · {len(gts)-n_pos} negativas\n")

    linhas = []
    yolos = sorted(glob.glob(os.path.join(YOLO_ROOT, f"{YOLO_CFG}_seed*")))
    unets = sorted(glob.glob(os.path.join(UNET_ROOT, "unet_black_seed*")))
    for d_ in yolos + unets:
        nome = os.path.basename(d_)
        p = os.path.join(d_, "weights", "best.pt")
        if not os.path.isfile(p):
            p = os.path.join(d_, "best.pt")
        if not os.path.isfile(p):
            print(f"  [PULADO] {nome}: sem best.pt")
            continue
        t0 = time.time()
        if nome.startswith("unet"):
            for t, masks in unet_masks(p, imagens, args.imgsz, LIMIARES_UNET).items():
                for f, pred, gt in zip(imagens, masks, gts):
                    linhas.append({"run": nome, "braco": "U-Net", "ponto": f"sigmoid={t}",
                                   "arquivo": os.path.basename(f),
                                   "iou": round(iou_de(pred, gt), 6),
                                   "pred_vazia": int(not pred.any()),
                                   "tem_ferida": int(gt.any())})
        else:
            from eval_test import padding_do_run
            masks = yolo_masks(p, imagens, args.conf, args.imgsz, padding_do_run(d_))
            for f, pred, gt in zip(imagens, masks, gts):
                linhas.append({"run": nome, "braco": "YOLO M", "ponto": f"conf={args.conf}",
                               "arquivo": os.path.basename(f),
                               "iou": round(iou_de(pred, gt), 6),
                               "pred_vazia": int(not pred.any()),
                               "tem_ferida": int(gt.any())})
        print(f"  {nome}  ({time.time()-t0:.0f}s)")

    saida = os.path.join("stage3", "iou_por_imagem.csv")
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"\ngravado {saida}")

    # ─────────────────────────────── resumo por braço e ponto de operação
    import statistics as st
    print(f"\n{'braço':<8} {'ponto':<15} {'recall':>7} {'precisão':>9} "
          f"{'IoU (pos)':>10} {'IoU (todas)':>12}")
    print("-" * 66)
    chaves = []
    for L in linhas:
        k = (L["braco"], L["ponto"])
        if k not in chaves:
            chaves.append(k)
    for braco, ponto in chaves:
        por_run = {"rec": [], "prec": [], "iou_pos": [], "iou_all": []}
        for run in sorted({L["run"] for L in linhas
                           if L["braco"] == braco and L["ponto"] == ponto}):
            sub = [L for L in linhas if L["run"] == run and L["ponto"] == ponto
                   and L["braco"] == braco]
            pos = [L for L in sub if L["tem_ferida"]]
            neg = [L for L in sub if not L["tem_ferida"]]
            tp = sum(1 for L in pos if not L["pred_vazia"])
            fp = sum(1 for L in neg if not L["pred_vazia"])
            por_run["rec"].append(tp / len(pos))
            por_run["prec"].append(tp / (tp + fp) if (tp + fp) else float("nan"))
            por_run["iou_pos"].append(st.mean(L["iou"] for L in pos))
            por_run["iou_all"].append(st.mean(L["iou"] for L in sub))
        print(f"{braco:<8} {ponto:<15} {st.mean(por_run['rec']):>7.4f} "
              f"{st.mean(por_run['prec']):>9.4f} {st.mean(por_run['iou_pos']):>10.4f} "
              f"{st.mean(por_run['iou_all']):>12.4f}")
    print("""
IoU (pos)   = média sobre as imagens COM ferida anotada
IoU (todas) = inclui as negativas, que valem 1,0 quando a predição também é vazia

Médias sobre os 5 seeds de cada braço. O ponto `sigmoid=0.999` é o que iguala o
recall do YOLO; o `sigmoid=0.5` é o regime nativo da U-Net.""")


if __name__ == "__main__":
    main()
