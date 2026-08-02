# -*- coding: utf-8 -*-
"""
stage3/eval_test.py — FASE 1 (GPU): prediz no test set e guarda os registros de casamento.

Um run de cada vez. Para cada imagem do test set:
  prediz -> rasteriza o GT do COCO -> casa por IoU em 10 limiares -> guarda
  {scores, tp[10][n], n_gt} e o grupo da imagem.

O que sai NAO sao mascaras (seriam gigabytes para 25 runs): sao os registros, de
alguns KB. Eles contem tudo que o AP precisa, e permitem recomputar o AP em
qualquer subconjunto — que e o que o cluster bootstrap do D4 exige.

  python stage3/eval_test.py --all
  python stage3/eval_test.py --run yolo11m-seg_black_coco_seed42

CONF BAIXA DE PROPOSITO. `--conf 0.001` retem quase toda deteccao, porque o mAP
integra a curva P-R inteira: cortar em 0,8 aqui truncaria a curva e subestimaria
o mAP. O limiar de 0,8 e aplicado DEPOIS, so em precision/recall/F1 (C6).

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
sys.path.insert(0, RAIZ)      # padding_patch vive na raiz; chdir NAO poe no sys.path
os.chdir(RAIZ)

import numpy as np

from ap_core import IOU_THRS, casa_imagem, matriz_iou, rasteriza_poly

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUNS_ROOT = os.path.join("runs", "segment", "runs_revision")
UNET_ROOT = os.path.join("runs", "segment", "unet_comparator")
COCO_TEST = os.path.join("coco_partitions", "instances_test.json")
SAIDA = os.path.join("stage3", "registros")


def grupos_do_test():
    M = list(csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8-sig")))
    return {r["arquivo_b"]: r["split_key"] for r in M if r["partition"] == "test"}


def carrega_gt():
    d = json.load(open(COCO_TEST, encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    por_img = {}
    for a in d["annotations"]:
        por_img.setdefault(a["image_id"], []).append(a)
    return imgs, por_img


def grade_ocupada(tol=900):
    for r in glob.glob(os.path.join(RUNS_ROOT, "*", "results.csv")):
        if os.path.exists(os.path.join(os.path.dirname(r), "COMPLETED.json")):
            continue
        if time.time() - os.path.getmtime(r) < tol:
            return os.path.basename(os.path.dirname(r))
    return None


def padding_do_run(dir_run):
    """Le o padding declarado no provenance.json do run.

    Predizer com o cinza padrao um modelo treinado com preto e' avaliar fora da
    distribuicao de treino. Era consistente entre modelos (comparacao justa) mas
    nao correspondia ao treino de nenhum — pendencia aberta junto com a D12.
    """
    import json
    p = os.path.join(dir_run, "provenance.json")
    if os.path.isfile(p):
        try:
            return json.load(open(p, encoding="utf-8")).get("padding", "gray")
        except Exception:
            pass
    return "gray"


def avalia_run(pesos, nome, imgs, gt_por_img, grupo_de, pasta_img, conf, imgsz,
               padding="gray"):
    import padding_patch
    padding_patch.apply(padding)          # antes de instanciar o modelo
    from ultralytics import YOLO
    modelo = YOLO(pesos)
    registros, sem_grupo = {}, 0

    for iid, im in imgs.items():
        arq = im["file_name"]
        caminho = os.path.join(pasta_img, arq)
        if not os.path.isfile(caminho):
            raise FileNotFoundError(caminho)
        w, h = im["width"], im["height"]

        gts = [rasteriza_poly(a["segmentation"], w, h) for a in gt_por_img.get(iid, [])]

        r = modelo.predict(caminho, conf=conf, imgsz=imgsz, retina_masks=True,
                           verbose=False)[0]
        preds, scores = [], []
        if r.masks is not None and len(r.masks) > 0:
            md = r.masks.data.cpu().numpy()          # (n, H, W) no espaço da imagem
            cf = r.boxes.conf.cpu().numpy()
            for k in range(md.shape[0]):
                m = md[k] > 0.5
                if m.shape != (h, w):                # segurança: reamostra se preciso
                    import cv2
                    m = cv2.resize(m.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                preds.append(m)
                scores.append(float(cf[k]))

        reg = casa_imagem(scores, matriz_iou(preds, gts))
        reg["grupo"] = grupo_de.get(arq)
        if reg["grupo"] is None:
            sem_grupo += 1
        registros[arq] = reg

    if sem_grupo:
        raise RuntimeError(f"{sem_grupo} imagens sem grupo — o bootstrap ficaria inválido")

    n_gt = sum(r["n_gt"] for r in registros.values())
    n_det = sum(len(r["scores"]) for r in registros.values())
    return {"run": nome, "pesos": pesos, "conf": conf, "imgsz": imgsz,
            "padding": padding,
            "iou_thrs": IOU_THRS.tolist(), "n_imagens": len(registros),
            "n_gt": n_gt, "n_deteccoes": n_det, "registros": registros}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="nome de um run específico")
    ap.add_argument("--all", action="store_true", help="todos os runs concluídos")
    ap.add_argument("--incluir-unet", action="store_true")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ocup = grade_ocupada()
    if ocup and not args.force:
        sys.exit(f"ABORTADO: a grade parece ativa ({ocup}). Esperar, ou --force.")

    import yaml
    dy = yaml.safe_load(open("data.yaml", encoding="utf-8"))
    raiz = dy.get("path", ".")
    pasta_img = dy["test"] if os.path.isabs(dy["test"]) else os.path.join(raiz, dy["test"])

    imgs, gt_por_img = carrega_gt()
    grupo_de = grupos_do_test()
    print(f"test: {len(imgs)} imagens · {sum(len(v) for v in gt_por_img.values())} GT · "
          f"{len(set(grupo_de[a['file_name']] for a in imgs.values()))} grupos")

    alvos, recusados = [], []
    raizes = [RUNS_ROOT] + ([UNET_ROOT] if args.incluir_unet else [])
    for rr in raizes:
        for d in sorted(glob.glob(os.path.join(rr, "*"))):
            nome = os.path.basename(d)
            if args.run and nome != args.run:
                continue
            if not os.path.exists(os.path.join(d, "COMPLETED.json")):
                continue
            # GUARDA D12: um run so entra se provar que o padding chegou ao
            # treino. Os 25 runs pre-correcao nao tem esse campo — e por isso
            # nao ha como confundi-los com os novos, mesmo tendo o mesmo nome.
            prov = os.path.join(d, "provenance.json")
            ev = {}
            if os.path.isfile(prov):
                try:
                    ev = json.load(open(prov, encoding="utf-8")).get(
                        "evidencia_padding_no_batch", {})
                except Exception:
                    ev = {}
            if not ev or "erro" in ev:
                print(f"  [RECUSADO] {nome}: sem evidencia de padding no batch "
                      f"(run anterior a D12?) — nao sera avaliado")
                recusados.append(nome)
                continue
            if os.path.isfile(os.path.join(d, "AVISO_PADDING.txt")):
                print(f"  [RECUSADO] {nome}: tem AVISO_PADDING.txt")
                recusados.append(nome)
                continue
            p = os.path.join(d, "weights", "best.pt")
            if not os.path.isfile(p):
                p = os.path.join(d, "best.pt")
            if os.path.isfile(p):
                alvos.append((nome, p))
    if recusados:
        print(f"\n{len(recusados)} run(s) recusado(s) pela guarda D12.\n")
    if not alvos:
        sys.exit("nenhum run concluído e válido encontrado")

    os.makedirs(SAIDA, exist_ok=True)
    print(f"runs a avaliar: {len(alvos)}\n")
    for i, (nome, pesos) in enumerate(alvos, 1):
        destino = os.path.join(SAIDA, f"{nome}.json")
        if os.path.exists(destino) and not args.force:
            print(f"[{i}/{len(alvos)}] {nome}  (já existe, pulando)")
            continue
        t = time.time()
        pad = padding_do_run(os.path.dirname(os.path.dirname(pesos)))
        out = avalia_run(pesos, nome, imgs, gt_por_img, grupo_de, pasta_img,
                         args.conf, args.imgsz, padding=pad)
        json.dump(out, open(destino, "w", encoding="utf-8"))
        print(f"[{i}/{len(alvos)}] {nome}  padding={pad}  "
              f"{out['n_deteccoes']} detecções  ({time.time()-t:.0f}s)")
    print(f"\nregistros em {SAIDA}/ — agora rode stage3/aggregate.py (não precisa de GPU)")


if __name__ == "__main__":
    main()
