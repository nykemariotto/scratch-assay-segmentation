# -*- coding: utf-8 -*-
"""
stage3/figura6_exemplo.py — Figure 6, the before/after panel, made reproducible.

WHY IT EXISTS. Figures 3, 4 and 5 come out of stage3/figuras_concordancia.py and can
be regenerated on demand. Figure 6 could not: no script produced it, so the version
in the manuscript is a hand-made file nobody can rebuild, and it sits at 220 ppi like
every other image Word compressed on insert.

WHY THE OLD ONE COULD NOT SIMPLY BE RE-EXPORTED. The example set carries the Roboflow
640x640 copies. Two 640 px panels side by side is 1280 px, which at 180 mm is 181 dpi
— under the 300 dpi Wiley asks of images and far under the 1800 px floor it states
for any figure. The dataset does hold the same acquisitions at native 2452x2056: 233
HUVEC frames at 12 h, 45 of them in the held-out test split with an annotated wound.
Two native panels are 4904 px, which is 692 dpi at 180 mm.

WHAT IT PRODUCES. A PDF where the panel titles are vector — the format Wiley prefers
for line art, and the reason this is not just a PNG — with the photographs embedded
at their acquired resolution. A 600 dpi PNG is written alongside as the raster
fallback.

THE IMAGE IS FROM THE TEST SPLIT, on purpose. A demonstration figure drawn from the
data the models trained on would show the tool at its most flattering and would not
be evidence of anything.

CPU ONLY, and not as a preference: `device="cpu"` is passed explicitly so this never
contends for a GPU that may be busy with someone else's work.

    python stage3/figura6_exemplo.py
    python stage3/figura6_exemplo.py --image "dataset/images/test/<file>.png"
"""
import argparse
import csv
import glob
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "stage2"))   # padding_patch lives here
os.chdir(RAIZ)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEST = "figures"
MAPA = "data/mapping_dataset_final_strat.csv"
# the same operating point the published interface uses, and the same padding the
# models were trained with — predicting with a different fill is predicting outside
# the training distribution
CONF = 0.8
IMGSZ = 640
PADDING = "black"
# B1 at 12 h is the well and timepoint the current Figure 6 shows; this is the
# native-resolution frame of that same well in the test split
PREFERIDA = "B1 12HR 1"


def escolhe_imagem():
    """A HUVEC frame at 12 h, test split, native resolution, with a wound annotated."""
    from PIL import Image
    disco = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join("dataset", "images", "test", "*.*"))}
    linhas = [r for r in csv.DictReader(open(MAPA, encoding="utf-8-sig"))
              if r.get("excluida") not in ("sim", "1", "True")]
    cand = []
    for r in linhas:
        if r.get("linha_celular") != "HUVEC" or r.get("timepoint_h") != "12":
            continue
        p = disco.get(os.path.splitext(r.get("arquivo_b", ""))[0])
        if not p:
            continue
        if Image.open(p).size[0] < 2000:
            continue
        rot = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        rot = os.path.splitext(rot)[0] + ".txt"
        n = sum(1 for _ in open(rot)) if os.path.isfile(rot) else 0
        if n:
            cand.append(p)
    if not cand:
        sys.exit("no native-resolution HUVEC frame at 12 h in the test split")
    for p in sorted(cand):
        if PREFERIDA.lower() in os.path.basename(p).lower():
            return p
    return sorted(cand)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--weights", default=None, help="defaults to the deployed M")
    ap.add_argument("--largura-mm", type=float, default=180.0)
    args = ap.parse_args()

    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_p = args.image or escolhe_imagem()
    pesos = args.weights
    if pesos is None:
        for c in (os.path.join("_publicar", "huggingface", "weights", "M.pt"),
                  os.path.join("webapp", "weights", "M.pt")):
            if os.path.isfile(c):
                pesos = c
                break
    if not pesos or not os.path.isfile(pesos):
        sys.exit("M.pt not found — pass --weights")

    print(f"image   {img_p}")
    print(f"weights {pesos}")

    import padding_patch
    fill = padding_patch.apply(PADDING)
    from ultralytics import YOLO
    modelo = YOLO(pesos)
    # explicit CPU: never take the card
    r = modelo.predict(img_p, conf=CONF, imgsz=IMGSZ, retina_masks=True,
                       device="cpu", verbose=False)[0]

    bgr = cv2.imread(img_p, cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    over = rgb.copy()
    area_px = 0
    if r.masks is not None and len(r.masks):
        md = r.masks.data.cpu().numpy() > 0.5
        uni = np.any(md, axis=0)
        if uni.shape != (h, w):
            uni = cv2.resize(uni.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        area_px = int(uni.sum())
        cont, _ = cv2.findContours(uni.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        # same contour convention as predict_example.py: thickness scaled to the
        # image so it reproduces at any output size
        cv2.drawContours(over, cont, -1, (255, 0, 0), max(3, w // 200))
    print(f"detections {0 if r.masks is None else len(r.masks)} · "
          f"area {area_px} px ({100.0*area_px/(w*h):.2f}% of the field)")

    pol = args.largura_mm / 25.4
    alt = pol / 2 * (h / w) + 0.28      # two panels plus room for the titles
    fig, axes = plt.subplots(1, 2, figsize=(pol, alt))
    for ax, im, tit in ((axes[0], rgb, "Original (12 h after scratch)"),
                        (axes[1], over, "AI segmentation")):
        ax.imshow(im)
        ax.set_title(tit, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.3)

    os.makedirs(DEST, exist_ok=True)
    base = os.path.join(DEST, "Figure6_example_segmentation")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)

    # `bbox_inches="tight"` trims whitespace, so the saved canvas comes out narrower
    # than figsize and the PNG lands a few dpi under target. Measure the trim and
    # re-save compensated, rather than shipping a file that reads 596 when the
    # guideline says 600.
    from PIL import Image
    ALVO_DPI = 600
    fig.savefig(f"{base}.png", dpi=ALVO_DPI, bbox_inches="tight", pad_inches=0.02)
    pw, _ = Image.open(f"{base}.png").size
    preciso = ALVO_DPI * pol
    if pw < preciso:
        fig.savefig(f"{base}.png", dpi=ALVO_DPI * preciso / pw,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    im = Image.open(f"{base}.png")
    im.save(f"{base}.png", "PNG", dpi=(ALVO_DPI, ALVO_DPI), optimize=True)
    pw, ph = Image.open(f"{base}.png").size
    print(f"\n  {base}.pdf  (vector titles, photographs at {w}x{h})")
    print(f"  {base}.png  {pw} x {ph} px · {pw/pol:.0f} dpi at {args.largura_mm:.0f} mm"
          f" · floor 1800 px {'ok' if pw >= 1800 else 'FAIL'}")
    print("\nSubmit as a separate file. Pasting into the .docx puts it through Word's "
          "'Print (220 ppi)'.")


if __name__ == "__main__":
    main()
