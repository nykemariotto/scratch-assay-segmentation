# -*- coding: utf-8 -*-
"""
stage3/predict_areas.py — FASE 1b (GPU): área da ferida por imagem do test set.

Complementa o stage3/eval_test.py. Aquele produz os registros de CASAMENTO, que servem
para mAP e IC. Este produz a ÁREA prevista, que é a grandeza clínica — é dela que
sai a closure fraction, e é ela que se compara com o padrão de referência da
Etapa 4 e com o WHST automático.

São coisas diferentes e é bom que sejam: um modelo pode ter mAP alto e área
sistematicamente enviesada, porque mAP mede se o contorno cai no lugar certo com
IoU suficiente, não se a área bate.

DETALHE QUE IMPORTA: quando há mais de uma instância na imagem, as máscaras são
UNIDAS antes de medir (`|`), não somadas. Somar contaria duas vezes o que se
sobrepõe, e o WHST — que mede a ferida como região — não faz isso. A união é a
definição comparável.

  python stage3/predict_areas.py --all
  python stage3/predict_areas.py --run yolo11m-seg_black_coco_seed42

NAO RODAR ENQUANTO A GRADE ESTIVER TREINANDO.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, RAIZ)      # idem: padding_patch esta na raiz
os.chdir(RAIZ)

import numpy as np

from eval_test import grade_ocupada

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUNS_ROOT = os.path.join("runs", "segment", "runs_revision")
UNET_ROOT = os.path.join("runs", "segment", "unet_comparator")
SAIDA = os.path.join("stage3", "areas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--conf", type=float, default=0.8,
                    help="limiar de operação da interface publicada")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ocup = grade_ocupada()
    if ocup and not args.force:
        sys.exit(f"ABORTADO: a grade parece ativa ({ocup}). Esperar, ou --force.")

    import yaml
    dy = yaml.safe_load(open("data.yaml", encoding="utf-8"))
    raiz = dy.get("path", ".")
    pasta = dy["test"] if os.path.isabs(dy["test"]) else os.path.join(raiz, dy["test"])
    imagens = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                     for f in glob.glob(os.path.join(pasta, e)))
    print(f"test: {len(imagens)} imagens · conf={args.conf}")

    alvos, recusados = [], []
    for rr in (RUNS_ROOT, UNET_ROOT):
        e_unet = rr == UNET_ROOT
        for d in sorted(glob.glob(os.path.join(rr, "*"))):
            nome = os.path.basename(d)
            if args.run and nome != args.run:
                continue
            if not os.path.exists(os.path.join(d, "COMPLETED.json")):
                continue
            # GUARDA D12 — mesma do stage3/eval_test.py. Um run YOLO so entra se provar
            # que o padding chegou ao treino. A U-Net nao passa pelo padding_patch
            # (pipeline proprio), entao e isenta.
            if not e_unet:
                ev = {}
                prov = os.path.join(d, "provenance.json")
                if os.path.isfile(prov):
                    try:
                        ev = json.load(open(prov, encoding="utf-8")).get(
                            "evidencia_padding_no_batch", {})
                    except Exception:
                        ev = {}
                if not ev or "erro" in ev or os.path.isfile(
                        os.path.join(d, "AVISO_PADDING.txt")):
                    print(f"  [RECUSADO] {nome}: sem evidencia de padding valida "
                          f"(run anterior a D12?)")
                    recusados.append(nome)
                    continue
            p = os.path.join(d, "weights", "best.pt")
            if not os.path.isfile(p):
                p = os.path.join(d, "best.pt")
            if os.path.isfile(p):
                alvos.append((nome, p, e_unet))
    if recusados:
        print(f"\n{len(recusados)} run(s) recusado(s) pela guarda D12.\n")
    if not alvos:
        sys.exit("nenhum run concluído e válido")

    os.makedirs(SAIDA, exist_ok=True)
    for i, (nome, pesos, e_unet) in enumerate(alvos, 1):
        dest = os.path.join(SAIDA, f"{nome}.csv")
        if os.path.exists(dest) and not args.force:
            print(f"[{i}/{len(alvos)}] {nome}  (já existe)")
            continue
        t = time.time()
        from eval_test import padding_do_run
        pad = padding_do_run(os.path.dirname(os.path.dirname(pesos)))
        linhas = (areas_unet(pesos, imagens, args) if e_unet
                  else areas_yolo(pesos, imagens, args, padding=pad))
        with open(dest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            w.writeheader()
            w.writerows(linhas)
        vazias = sum(1 for r in linhas if r["area_px"] == 0)
        print(f"[{i}/{len(alvos)}] {nome}  padding={pad}  {len(linhas)} imgs · "
              f"{vazias} sem detecção ({time.time()-t:.0f}s)")
    print(f"\náreas em {SAIDA}/ — agora rode stage3/paired_new.py (sem GPU)")


def areas_yolo(pesos, imagens, args, padding="gray"):
    import padding_patch
    padding_patch.apply(padding)          # mesmo padding do treino
    from ultralytics import YOLO
    m = YOLO(pesos)
    out = []
    for f in imagens:
        r = m.predict(f, conf=args.conf, imgsz=args.imgsz, retina_masks=True,
                      verbose=False)[0]
        h, w = r.orig_shape
        if r.masks is None or len(r.masks) == 0:
            area, n = 0, 0
        else:
            md = r.masks.data.cpu().numpy() > 0.5
            uniao = np.any(md, axis=0)           # UNIÃO, não soma
            if uniao.shape != (h, w):
                import cv2
                uniao = cv2.resize(uniao.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            area, n = int(uniao.sum()), int(md.shape[0])
        out.append({"arquivo": os.path.basename(f), "area_px": area,
                    "area_pct": round(100.0 * area / (h * w), 6),
                    "n_instancias": n, "orig_w": w, "orig_h": h})
    return out


def areas_unet(pesos, imagens, args):
    sys.path.insert(0, os.path.join(os.getcwd(), "unet_comparator"))
    import cv2
    import torch
    from unet_data import desfaz_letterbox, letterbox
    from unet_model import UNet
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(pesos, map_location=dev, weights_only=False)
    fill = 0 if ck.get("padding", "black") == "black" else 255
    m = UNet().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    out = []
    with torch.no_grad():
        for f in imagens:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            lb_img, _, lb = letterbox(img, np.zeros((h, w), np.uint8), args.imgsz, fill)
            x = torch.from_numpy(lb_img.transpose(2, 0, 1).copy()).float().div_(255)
            p = (torch.sigmoid(m(x.unsqueeze(0).to(dev))) > 0.5).byte().cpu().numpy()[0, 0]
            mm = desfaz_letterbox(p, lb)
            n = cv2.connectedComponentsWithStats(mm, 8)[0] - 1
            area = int(mm.sum())
            out.append({"arquivo": os.path.basename(f), "area_px": area,
                        "area_pct": round(100.0 * area / (h * w), 6),
                        "n_instancias": n, "orig_w": w, "orig_h": h})
    return out


if __name__ == "__main__":
    main()
