# -*- coding: utf-8 -*-
"""
stage3/make_fixtures.py — SYNTHETIC records to validate stage3/aggregate.py without a GPU.

Uses the REAL STRUCTURE of the test set — the 234 images, the 37 groups with their
unequal sizes (2 to 22 images) and the real GT count per image, read from
`instances_test.json` and the mapping. Only the predictions are invented.

That matters: the difference between grouped and naive bootstrap depends entirely
on the cluster structure. Testing with 37 equal groups of size 6 would not exercise
the real case, where one group has 22 images and another has 2.

The generator embeds three things stage3/aggregate.py has to be able to see:
  - an ordering between configurations (x > m > s)
  - a small and CONSISTENT advantage for black padding (the padding ablation)
  - correlation WITHIN the group — hard fields are hard for every seed
    (which is what makes the naive CI lie)

The numbers have no scientific meaning. Only the mechanics are being tested.
"""
import csv
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
os.chdir(os.path.dirname(AQUI))

import numpy as np

from ap_core import IOU_THRS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEST = os.path.join("stage3", "records_fixture")

# configuration -> "true" quality (probability of a hit at IoU 0.50)
CFG = {
    "yolo11s-seg_black_coco": 0.86,
    "yolo11m-seg_black_coco": 0.91,
    "yolo11x-seg_black_coco": 0.93,
    "yolo11m-seg_white_coco": 0.89,          # 2 pp abaixo do black: a ablation
    "yolo11m-seg_black_scratch": 0.72,
}
SEEDS = [42, 43, 44, 45, 46]


def estrutura_real():
    d = json.load(open(os.path.join("coco_partitions", "instances_test.json"),
                      encoding="utf-8"))
    n_gt = {im["file_name"]: 0 for im in d["images"]}
    por_id = {im["id"]: im["file_name"] for im in d["images"]}
    for a in d["annotations"]:
        n_gt[por_id[a["image_id"]]] += 1
    M = list(csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8-sig")))
    grupo = {r["arquivo_b"]: r["split_key"] for r in M if r["partition"] == "test"}
    faltando = [k for k in n_gt if k not in grupo]
    if faltando:
        sys.exit(f"{len(faltando)} images with no group — invalid fixture")
    return n_gt, grupo


def gera(qualidade, seed, n_gt, grupo):
    rng = np.random.default_rng(seed * 7919 + int(1000 * qualidade))
    # difficulty per GROUP, constant across seeds -> intra-cluster correlation
    grupos = sorted(set(grupo.values()))
    rg = np.random.default_rng(20260726)                 # fixo: o campo é o mesmo
    dif = {g: rg.normal(0, 0.18) for g in grupos}
    # training noise, per seed
    desloc = rng.normal(0, 0.012)

    regs = {}
    for arq, k in n_gt.items():
        p = float(np.clip(qualidade + desloc - dif[grupo[arq]], 0.02, 0.995))
        scores, tp = [], [[] for _ in IOU_THRS]
        for _ in range(k):                                # one detection per GT
            acerta = rng.random() < p
            s = float(np.clip(rng.beta(9, 2) if acerta else rng.beta(2, 4), 0.001, 0.999))
            scores.append(s)
            for t_i, t in enumerate(IOU_THRS):
                # the stricter the threshold, the fewer TP survive
                sobra = acerta and (rng.random() < np.clip(1.25 - 1.35 * (t - 0.5), 0, 1))
                tp[t_i].append(1 if sobra else 0)
        n_fp = rng.poisson(0.12)                          # falsos positivos esparsos
        for _ in range(n_fp):
            scores.append(float(np.clip(rng.beta(2, 6), 0.001, 0.999)))
            for t_i in range(len(IOU_THRS)):
                tp[t_i].append(0)
        o = np.argsort(-np.asarray(scores))
        regs[arq] = {"scores": [scores[i] for i in o],
                     "tp": [[t[i] for i in o] for t in tp],
                     "n_gt": k, "grupo": grupo[arq]}
    return regs


def main():
    n_gt, grupo = estrutura_real()
    print(f"real structure: {len(n_gt)} images · {sum(n_gt.values())} GT · "
          f"{len(set(grupo.values()))} grupos")
    tam = {}
    for a, g in grupo.items():
        tam[g] = tam.get(g, 0) + 1
    print(f"group size: min {min(tam.values())} · max {max(tam.values())}")

    os.makedirs(DEST, exist_ok=True)
    n = 0
    for cfg, q in CFG.items():
        for s in SEEDS:
            nome = f"{cfg}_seed{s}"
            regs = gera(q, s, n_gt, grupo)
            json.dump({"run": nome, "pesos": "(fixture)", "conf": 0.001, "imgsz": 640,
                       "iou_thrs": IOU_THRS.tolist(), "n_imagens": len(regs),
                       "n_gt": sum(r["n_gt"] for r in regs.values()),
                       "n_deteccoes": sum(len(r["scores"]) for r in regs.values()),
                       "FIXTURE": True, "registros": regs},
                      open(os.path.join(DEST, nome + ".json"), "w", encoding="utf-8"))
            n += 1
    print(f"\n{n} synthetic records in {DEST}/")
    print("WARNING: these are fictitious. They only validate the mechanics of stage3/aggregate.py.")


if __name__ == "__main__":
    main()
