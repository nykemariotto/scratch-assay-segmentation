# -*- coding: utf-8 -*-
"""
stage1/validate_640_transform.py — GATE 2.0. Descobre QUAL transformacao levou a imagem
raw (2452x2056) to the Roboflow 640x640, before assuming an anisotropic stretch.

Testa hipoteses concorrentes correlacionando a imagem 640 real contra
reconstrucoes a partir da crua:
  A) stretch  : resize direto 2452x2056 -> 640x640 (anisotropico)
  B) crop-c   : center-crop quadrado 2056x2056 -> resize 640
  C) letterbox: encaixa preservando AR, preenche bordas
"""
import os, json, csv, random
import numpy as np
from PIL import Image

B640 = (os.environ.get("ROBOFLOW_EXPORT", "<roboflow_export>")
        + r"\Pre Eclampsia.coco-segmentation\train")
RAWROOT = (os.environ.get("LAB_SHARE", "<lab_share>")
           + r"\Projetos\Projeto - Pré-eclâmpsia\Banco de dados")
COCO = os.path.join(B640, "_annotations.coco.json")
N = 10

coco = json.load(open(COCO, encoding="utf-8"))
small = {im["file_name"] for im in coco["images"] if (im["width"], im["height"]) == (640, 640)}
rows = {r["arquivo_b"]: r for r in csv.DictReader(
    open("data/mapping_dataset_final_strat.csv", encoding="utf-8"))}
cand = [rows[f] for f in small if f in rows and rows[f]["arquivo_a"].strip()]
random.Random(0).shuffle(cand)
cand = cand[:N]
print(f"Sampling {len(cand)} 640x640 images with a raw source\n")


def z(a):
    a = a.astype(np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def corr(a, b):
    za, zb = z(a), z(b)
    return float(np.dot(za, zb) / len(za))


res = {"stretch": [], "crop_center": [], "letterbox": []}
for r in cand:
    p640 = os.path.join(B640, r["arquivo_b"])
    praw = os.path.join(RAWROOT, r["pasta_a"].replace("/", os.sep), r["arquivo_a"])
    if not os.path.exists(praw):
        print(f"  [skipped] raw not found: {praw}")
        continue
    img640 = np.asarray(Image.open(p640).convert("L").resize((256, 256), Image.BILINEAR))
    raw = Image.open(praw).convert("L")
    W, H = raw.size

    # A) stretch anisotropico
    a = np.asarray(raw.resize((640, 640), Image.BILINEAR).resize((256, 256), Image.BILINEAR))
    # B) center crop quadrado -> 640
    s = min(W, H)
    l, t = (W - s) // 2, (H - s) // 2
    b = np.asarray(raw.crop((l, t, l + s, t + s)).resize((640, 640), Image.BILINEAR)
                   .resize((256, 256), Image.BILINEAR))
    # C) letterbox preservando AR
    sc = min(640 / W, 640 / H)
    nw, nh = int(round(W * sc)), int(round(H * sc))
    cv = Image.new("L", (640, 640), 114)
    cv.paste(raw.resize((nw, nh), Image.BILINEAR), ((640 - nw) // 2, (640 - nh) // 2))
    c = np.asarray(cv.resize((256, 256), Image.BILINEAR))

    ca, cb, cc = corr(img640, a), corr(img640, b), corr(img640, c)
    res["stretch"].append(ca)
    res["crop_center"].append(cb)
    res["letterbox"].append(cc)
    win = max((ca, "stretch"), (cb, "crop_center"), (cc, "letterbox"))[1]
    print(f"  {r['arquivo_a'][:26]:<26} stretch={ca:.4f}  crop={cb:.4f}  "
          f"letterbox={cc:.4f}  -> {win}")

print("\n=== medias ===")
for k, v in res.items():
    if v:
        print(f"  {k:<12} media={np.mean(v):.4f}  min={np.min(v):.4f}  max={np.max(v):.4f}")
best = max(res, key=lambda k: np.mean(res[k]) if res[k] else -9)
print(f"\nHIPOTESE VENCEDORA: {best}")
if best == "stretch":
    print("  -> FX=2452/640=3.8312, FY=2056/640=3.2125 (anisotropico) CONFIRMADO")
else:
    print("  -> WARNING: the simple anisotropic remapping does NOT apply.")
