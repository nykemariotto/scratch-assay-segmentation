# -*- coding: utf-8 -*-
"""
stage1/prepare_examples.py — monta `examples/`, o conjunto minimo executavel do
repositorio (responde ao pedido de "dataset de exemplo" + "guia passo-a-passo").

Goal: someone who clones the repository can run the pipeline end to end WITHOUT
downloading the gigabytes from Zenodo — 4 test-set images, with YOLO annotations,
covering the cases that matter:

  1. HUVEC with an open, annotated wound    (the typical case)
  2. HUVEC negative (0 instances)           (closed wound — the negatives are
                                             deliberados)
  3. SKOV-3 em resolucao nativa             (a outra linha celular, 2452x2056)
  4. HUVEC em outro timepoint               (mostra a progressao temporal)

Deterministic choice (the smallest files satisfying each criterion), so that
re-running produces exactly the same set.
"""
import csv, hashlib, os, shutil
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MAP = "data/mapping_dataset_final_strat.csv"
IMG = os.path.join("dataset", "images", "test")
LAB = os.path.join("dataset", "labels", "test")
OUT = "examples"


def info(r):
    f = r["arquivo_b"]
    p, lb = os.path.join(IMG, f), os.path.join(LAB, os.path.splitext(f)[0] + ".txt")
    if not (os.path.exists(p) and os.path.exists(lb)):
        return None
    n = sum(1 for ln in open(lb) if ln.strip())
    with Image.open(p) as im:
        dim = im.size
    return dict(f=f, p=p, lb=lb, size=os.path.getsize(p), n=n, dim=dim,
                cl=r["linha_celular"], tp=int(r["timepoint_h"]))


rows = [r for r in csv.DictReader(open(MAP, encoding="utf-8")) if r["partition"] == "test"]
cand = [x for x in (info(r) for r in rows) if x]
cand.sort(key=lambda x: x["size"])            # deterministico: menor primeiro

sel, usados = [], set()


def pega(pred, rot):
    for c in cand:
        if c["f"] in usados or not pred(c):
            continue
        usados.add(c["f"])
        sel.append((c, rot))
        return c
    print(f"  [warning] no candidate for: {rot}")
    return None


pega(lambda c: c["cl"] == "HUVEC" and c["n"] > 0, "HUVEC · ferida anotada")
pega(lambda c: c["cl"] == "HUVEC" and c["n"] == 0, "HUVEC · negativo (ferida fechada)")
pega(lambda c: c["cl"] != "HUVEC" and c["n"] > 0, "SKOV-3 · resolucao nativa")
_t = {c["tp"] for c, _ in sel}
pega(lambda c: c["cl"] == "HUVEC" and c["n"] > 0 and c["tp"] not in _t,
     "HUVEC · outro timepoint")

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)

man = []
tot = 0
for c, rot in sel:
    shutil.copy2(c["p"], os.path.join(OUT, "images", c["f"]))
    shutil.copy2(c["lb"], os.path.join(OUT, "labels", os.path.basename(c["lb"])))
    tot += c["size"]
    man.append({"arquivo": c["f"], "papel": rot, "cell_line": c["cl"],
                "timepoint_h": c["tp"], "largura": c["dim"][0], "altura": c["dim"][1],
                "n_instancias": c["n"], "tamanho_kb": round(c["size"] / 1e3),
                "md5": hashlib.md5(open(c["p"], "rb").read()).hexdigest()})
    print(f"  {rot:<38} {c['dim']} {c['n']} inst  {c['size']/1e3:>6.0f} KB  {c['f'][:40]}")

with open(os.path.join(OUT, "MANIFEST.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(man[0].keys()))
    w.writeheader(); w.writerows(man)

# data.yaml pointing at the examples (lets you run val/predict directly)
with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
    f.write("# Example set — 4 test-set images, to try the pipeline out without\n"
            "# downloading the full dataset from Zenodo.\n"
            f"path: {os.path.abspath(OUT).replace(os.sep, '/')}\n"
            "train: images\nval: images\ntest: images\n"
            "names:\n  0: wound\n")

print(f"\n{len(man)} exemplos em {OUT}/ ({tot/1e6:.1f} MB) + MANIFEST.csv + data.yaml")
