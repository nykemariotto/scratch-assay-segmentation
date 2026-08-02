
# -*- coding: utf-8 -*-
"""
stage3/latencia_inferencia.py — quanto custa uma predição, por configuração.

POR QUE ISTO IMPORTA AGORA. As cinco configurações são estatisticamente
indistinguíveis em mAP (D19) e a U-Net difere delas em modo de falha, não em
qualidade média (D24). Quando a acurácia empata, o critério de escolha para o que
se implanta passa a ser **latência** — e oferecer ao usuário uma opção rápida com
perda declarada de precisão é recurso, não concessão.

MEDE EM CPU E EM GPU, e a CPU é a que decide: um Space gratuito do Hugging Face
roda em CPU. A latência de GPU serve para o parágrafo de custo computacional do
manuscrito; a de CPU serve para escolher o que vai ao ar.

METODOLOGIA
  · 3 imagens de aquecimento descartadas (aloca buffers, resolve autotune);
  · mediana sobre N imagens, não média — uma pausa do SO estraga a média;
  · mesmo conf, mesmo imgsz e mesmo padding do treino de cada run;
  · a U-Net entra com o mesmo pipeline do predict_areas (letterbox + sigmoid).

    python stage3/latencia_inferencia.py --n 40
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# uma seed representativa por configuracao — as cinco seeds sao equivalentes em
# arquitetura, e latencia nao depende do seed
ALVOS = [
    ("S",         "runs/segment/runs_revision/yolo11s-seg_black_coco_seed42", "yolo", 10.1),
    ("M",         "runs/segment/runs_revision/yolo11m-seg_black_coco_seed42", "yolo", 22.4),
    ("X",         "runs/segment/runs_revision/yolo11x-seg_black_coco_seed42", "yolo", 62.1),
    ("M-white",   "runs/segment/runs_revision/yolo11m-seg_white_coco_seed42", "yolo", 22.4),
    ("M-scratch", "runs/segment/runs_revision/yolo11m-seg_black_scratch_seed42", "yolo", 22.4),
    ("U-Net",     "runs/segment/unet_comparator/unet_black_seed42", "unet", 31.0),
]


def pesos_de(d):
    for p in (os.path.join(d, "weights", "best.pt"), os.path.join(d, "best.pt")):
        if os.path.isfile(p):
            return p
    return None


def mede_yolo(pesos, imgs, dev, conf, imgsz, padding, aquec=3):
    import padding_patch
    padding_patch.apply(padding)
    from ultralytics import YOLO
    m = YOLO(pesos)
    m.to(dev)
    ts = []
    for i, f in enumerate(imgs):
        t = time.perf_counter()
        m.predict(f, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False,
                  device=dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= aquec:
            ts.append(time.perf_counter() - t)
    del m
    return ts


def mede_unet(pesos, imgs, dev, imgsz, aquec=3):
    from unet_data import desfaz_letterbox, letterbox
    from unet_model import UNet
    ck = torch.load(pesos, map_location=dev, weights_only=False)
    fill = 0 if ck.get("padding", "black") == "black" else 255
    m = UNet().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    ts = []
    with torch.no_grad():
        for i, f in enumerate(imgs):
            t = time.perf_counter()
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            lb, _, par = letterbox(img, np.zeros((h, w), np.uint8), imgsz, fill)
            x = torch.from_numpy(lb.transpose(2, 0, 1).copy()).float().div_(255)
            p = (torch.sigmoid(m(x.unsqueeze(0).to(dev))) > 0.5).byte().cpu().numpy()[0, 0]
            desfaz_letterbox(p, par)
            if dev == "cuda":
                torch.cuda.synchronize()
            if i >= aquec:
                ts.append(time.perf_counter() - t)
    del m
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from unet_data import carrega_splits
    d = carrega_splits("data.yaml")["test"]
    imgs = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                  for f in glob.glob(os.path.join(d, e)))[:args.n + 3]
    print(f"{len(imgs)-3} imagens medidas (+3 de aquecimento) · conf={args.conf} · "
          f"imgsz={args.imgsz}")
    print(f"CPU: {os.cpu_count()} núcleos · "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ausente'}\n")

    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    linhas = []
    for dev in devs:
        print(f"── {dev.upper()} ──")
        print(f"  {'config':<11} {'params':>8} {'mediana':>10} {'p90':>9} "
              f"{'img/s':>8} {'tam. peso':>10}")
        for rot, dd, tipo, par in ALVOS:
            p = pesos_de(dd)
            if not p:
                print(f"  {rot:<11} [sem best.pt]")
                continue
            from eval_test import padding_do_run
            pad = padding_do_run(dd) if tipo == "yolo" else "black"
            ts = (mede_yolo(p, imgs, dev, args.conf, args.imgsz, pad) if tipo == "yolo"
                  else mede_unet(p, imgs, dev, args.imgsz))
            med = st.median(ts)
            p90 = sorted(ts)[int(0.9 * len(ts))]
            mb = os.path.getsize(p) / 1e6
            print(f"  {rot:<11} {par:>6.1f} M {1000*med:>8.0f} ms {1000*p90:>7.0f} ms "
                  f"{1/med:>8.2f} {mb:>8.1f} MB")
            linhas.append({"config": rot, "device": dev, "params_M": par,
                           "mediana_ms": round(1000 * med, 2),
                           "p90_ms": round(1000 * p90, 2),
                           "img_por_s": round(1 / med, 3),
                           "peso_MB": round(mb, 1), "n": len(ts)})
            if dev == "cuda":
                torch.cuda.empty_cache()
        print()

    dest = os.path.join("stage3", "latencia_inferencia.csv")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"gravado {dest}")

    cpu = {l["config"]: l for l in linhas if l["device"] == "cpu"}
    if len(cpu) > 1:
        rap = min(cpu.values(), key=lambda l: l["mediana_ms"])
        len_ = max(cpu.values(), key=lambda l: l["mediana_ms"])
        print(f"""
LEITURA. Em CPU — que é onde um Space gratuito roda — o mais rápido é
{rap['config']} ({rap['mediana_ms']:.0f} ms) e o mais lento {len_['config']}
({len_['mediana_ms']:.0f} ms), uma razão de {len_['mediana_ms']/rap['mediana_ms']:.1f}x.

Como as cinco configurações YOLO são indistinguíveis em mAP, essa razão é o único
critério objetivo que resta entre elas. A U-Net é caso à parte: difere em modo de
falha, então a escolha entre ela e o detector não é de velocidade.""")


if __name__ == "__main__":
    main()
