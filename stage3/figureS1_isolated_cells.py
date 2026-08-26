# -*- coding: utf-8 -*-
"""
stage3/figureS1_isolated_cells.py — Supplementary Figure S1.

POR QUE EXISTE. Quantificar a ferida como um unico contorno superestima a area
aberta quando celulas se destacam e migram individualmente para dentro do vao. O manuscrito ja declara isso no terceiro item
das Limitacoes; esta figura mostra.

O QUE ELA MOSTRA, E O QUE NAO MOSTRA. Ela e ILUSTRATIVA e nao quantitativa. Nao ha
deteccao de celula individual neste trabalho, entao a fracao da area ocupada por
celulas ja dentro do vao NAO e medida aqui nem em lugar nenhum do artigo — mede-la
exigiria um metodo que o estudo nao tem. O que a figura demonstra e o mecanismo, e
um ponto que importa mais: OS DOIS FLUXOS INCLUEM AS MESMAS CELULAS. O contorno do
padrao de referencia, tracado por um observador humano, e o do modelo enquadram a
mesma regiao, entao o vies corre na mesma direcao nos dois lados da comparacao
pareada — que e exatamente o que o texto das Limitacoes afirma e o que preserva a
validade do pareamento.

A IMAGEM. HUVEC, 12 h pos-risco, do conjunto de teste retido. Escolhida por ter
aglomerados isolados visiveis dentro do vao e por ter contorno corrigido no padrao
de referencia, de modo que os dois contornos existem sobre o MESMO referencial
(2452x2056, resolucao de aquisicao) e nao dependem de reescala.

O RECORTE do painel B foi escolhido visualmente, sobre a regiao onde os
aglomerados sao mais evidentes. E decisao editorial de uma figura ilustrativa, nao
medicao, e esta declarada aqui para que ninguem a leia como criterio.

CONTRASTE. Referencia em linha continua, modelo em tracejada: distinguiveis em
escala de cinza, que e requisito das figuras deste manuscrito.

    python stage3/figureS1_isolated_cells.py
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "stage2"))   # padding_patch lives here
os.chdir(RAIZ)

import csv                                          # noqa: E402
import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TESTE = "D3 12H 2_png.rf.bk65RyMB9bMlU59S1RY7.png"
PESOS = "runs/segment/runs_revision/yolo11m-seg_black_coco_seed42/weights/best.pt"
CONF = 0.80
LARGURA_MM = 180
DPI = 600
PISO_PX = 1800
# recorte do painel B, em pixels da imagem adquirida. Escolha visual, nao medida.
RECORTE = (1020, 60, 1580, 460)      # x0, y0, x1, y1

REF = "#0072B2"      # padrao de referencia
MOD = "#D55E00"      # configuracao implantada


def contorno(mask):
    c, _ = cv2.findContours((mask > 0).astype(np.uint8),
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return c


def main():
    linha = next(x for x in csv.DictReader(
        open("whst_output/overlays_sorted_map.csv", encoding="utf-8-sig"))
        if x["test_image"].strip() == TESTE)
    base = os.path.splitext(linha["whst_input_file"])[0]
    p_img = os.path.join("dataset", "images", "test", TESTE)
    p_ref = f"whst_output/rois_corrected/masks/{base}_mask.png"
    for p in (p_img, p_ref, PESOS):
        if not os.path.isfile(p):
            sys.exit(f"ausente: {p}")

    im = cv2.imread(p_img, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(p_ref, cv2.IMREAD_GRAYSCALE)
    if im.shape != ref.shape:
        sys.exit(f"referenciais diferentes: {im.shape} vs {ref.shape} — abortado, "
                 f"sobrepor contornos exigiria reescalar um deles")
    h, w = im.shape
    print(f"imagem {w}x{h} · referencia no mesmo referencial")

    import padding_patch
    padding_patch.apply("black")
    from ultralytics import YOLO
    r = YOLO(PESOS).predict(p_img, conf=CONF, imgsz=640, retina_masks=True,
                            device="cpu", verbose=False)[0]
    if r.masks is None or not len(r.masks):
        sys.exit("o modelo nao detectou nada nesta imagem")
    mod = np.any(r.masks.data.cpu().numpy() > 0.5, axis=0)
    if mod.shape != (h, w):
        mod = cv2.resize(mod.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    print(f"referencia {100*(ref>0).mean():.2f}% do campo · "
          f"modelo {100*mod.mean():.2f}%")

    x0, y0, x1, y1 = RECORTE
    pol = LARGURA_MM / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(pol, pol * 0.42),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})

    for ax, (img, cref, cmod, tit) in zip(axes, [
        (im, contorno(ref), contorno(mod.astype(np.uint8)), "Full field"),
        (im[y0:y1, x0:x1],
         contorno(ref[y0:y1, x0:x1]),
         contorno(mod[y0:y1, x0:x1].astype(np.uint8)),
         "Detail: cells already inside the gap"),
    ]):
        ax.imshow(img, cmap="gray", interpolation="nearest")
        for c in cref:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color=REF, lw=1.6, ls="-")
        for c in cmod:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color=MOD, lw=1.6, ls="--")
        ax.set_title(tit, fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor="black", lw=0.9, ls=":"))
    axes[0].plot([], [], color=REF, lw=1.6, ls="-", label="Reference standard")
    axes[0].plot([], [], color=MOD, lw=1.6, ls="--", label="Automated workflow")
    axes[0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    fig.tight_layout(pad=0.3)
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/FigureS1_isolated_cells.{ext}", dpi=DPI,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    from PIL import Image
    px = Image.open("figures/FigureS1_isolated_cells.png").size
    print(f"\n  figures/FigureS1_isolated_cells.pdf + .png")
    print(f"  {px[0]} x {px[1]} px · {DPI} dpi at {LARGURA_MM} mm · "
          f"floor {PISO_PX} px {'ok' if px[0] >= PISO_PX else 'FAIL'}")
    if px[0] < PISO_PX:
        sys.exit("abaixo do minimo da Wiley")


if __name__ == "__main__":
    main()
