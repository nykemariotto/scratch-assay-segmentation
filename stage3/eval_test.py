# -*- coding: utf-8 -*-
"""
stage3/eval_test.py — PHASE 1 (GPU): predicts on the test set and stores the
matching records.

One run at a time. For each test-set image:
  predict -> rasterise the COCO ground truth -> match by IoU at 10 thresholds ->
  store {scores, tp[10][n], n_gt} and the image's group.

What comes out are not masks (they would be gigabytes over 25 runs): they are the
records, a few kB each. They contain everything AP needs, and they allow AP to be
recomputed over any subset, which is what the cluster bootstrap requires.

  python stage3/eval_test.py --all
  python stage3/eval_test.py --run yolo11m-seg_black_coco_seed42

A LOW CONF IS DELIBERATE. `--conf 0.001` keeps almost every detection, because mAP
integrates the whole P-R curve: cutting at 0.8 here would truncate the curve and
understate the mAP. The 0.8 threshold is applied AFTERWARDS, only to
precision/recall/F1.

DO NOT RUN WHILE THE GRID IS TRAINING.
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
sys.path.insert(0, RAIZ)      # chdir does NOT put the root on sys.path
sys.path.insert(0, os.path.join(RAIZ, "stage2"))   # padding_patch lives here
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
SAIDA = os.path.join("stage3", "records")


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
    """Reads the padding declared in the run provenance.json.

    Predicting with the default grey on a model trained with black is evaluating
    outside the training distribution. It was consistent across models (a fair
    comparison) but matched no model's training — an open gap at the time.
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
    padding_patch.apply(padding)          # before instantiating the model
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
            md = r.masks.data.cpu().numpy()          # (n, H, W) in image space
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
        raise RuntimeError(f"{sem_grupo} images with no group — the bootstrap would be invalid")

    n_gt = sum(r["n_gt"] for r in registros.values())
    n_det = sum(len(r["scores"]) for r in registros.values())
    return {"run": nome, "pesos": pesos, "conf": conf, "imgsz": imgsz,
            "padding": padding,
            "iou_thrs": IOU_THRS.tolist(), "n_imagens": len(registros),
            "n_gt": n_gt, "n_deteccoes": n_det, "registros": registros}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="name of a single run")
    ap.add_argument("--all", action="store_true", help="every finished run")
    ap.add_argument("--incluir-unet", action="store_true")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ocup = grade_ocupada()
    if ocup and not args.force:
        sys.exit(f"ABORTED: the grid looks active ({ocup}). Wait, or pass --force.")

    import yaml
    dy = yaml.safe_load(open("data.yaml", encoding="utf-8"))
    raiz = dy.get("path", ".")
    pasta_img = dy["test"] if os.path.isabs(dy["test"]) else os.path.join(raiz, dy["test"])

    imgs, gt_por_img = carrega_gt()
    grupo_de = grupos_do_test()
    print(f"test: {len(imgs)} images · {sum(len(v) for v in gt_por_img.values())} GT · "
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
            # PADDING GUARD: a run only enters if it proves the padding reached
            # training. The 25 pre-correction runs do not carry that field, which
            # is why they cannot be confused with the new ones despite sharing names.
            prov = os.path.join(d, "provenance.json")
            ev = {}
            if os.path.isfile(prov):
                try:
                    ev = json.load(open(prov, encoding="utf-8")).get(
                        "evidencia_padding_no_batch", {})
                except Exception:
                    ev = {}
            if not ev or "erro" in ev:
                print(f"  [REFUSED] {nome}: no evidence of padding in the batch "
                      f"(run predating the padding fix?) — will not be evaluated")
                recusados.append(nome)
                continue
            if os.path.isfile(os.path.join(d, "AVISO_PADDING.txt")):
                print(f"  [REFUSED] {nome}: it has an AVISO_PADDING.txt")
                recusados.append(nome)
                continue
            p = os.path.join(d, "weights", "best.pt")
            if not os.path.isfile(p):
                p = os.path.join(d, "best.pt")
            if os.path.isfile(p):
                alvos.append((nome, p))
    if recusados:
        print(f"\n{len(recusados)} run(s) refused by the padding guard.\n")
    if not alvos:
        sys.exit("no finished and valid run found")

    os.makedirs(SAIDA, exist_ok=True)
    print(f"runs to evaluate: {len(alvos)}\n")
    for i, (nome, pesos) in enumerate(alvos, 1):
        destino = os.path.join(SAIDA, f"{nome}.json")
        if os.path.exists(destino) and not args.force:
            print(f"[{i}/{len(alvos)}] {nome}  (already exists, skipping)")
            continue
        t = time.time()
        pad = padding_do_run(os.path.dirname(os.path.dirname(pesos)))
        out = avalia_run(pesos, nome, imgs, gt_por_img, grupo_de, pasta_img,
                         args.conf, args.imgsz, padding=pad)
        json.dump(out, open(destino, "w", encoding="utf-8"))
        print(f"[{i}/{len(alvos)}] {nome}  padding={pad}  "
              f"{out['n_deteccoes']} detecções  ({time.time()-t:.0f}s)")
    print(f"\nrecords in {SAIDA}/ — now run stage3/aggregate.py (no GPU needed)")


if __name__ == "__main__":
    main()
