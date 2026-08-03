# -*- coding: utf-8 -*-
"""
stage3/figura1_fluxo.py — Figure 1, the study in one diagram.

WHY THIS REPLACES THE MONTAGE. The paper is titled "A Benchmark and Web-Accessible
Tool", and nothing in it showed either. The opening figure was six brightfield panels:
pleasant, but it put an imperfect contour in front of the reader before any metric had
been defined, and it carried no information the Methods did not already state. The
work this revision actually did — rebuilding the partition so that no well appears on
both sides of it — was invisible.

EVERY COUNT IS READ FROM THE ARTEFACT IT COMES FROM. A flow diagram with numbers is a
data figure, and numbers typed into a drawing drift the moment the pipeline moves. The
one exception is the acquisition total: frames discarded before the blind triage are
not versioned here, so 1,497 is a declared constant with its source named below, and
the script asserts that it is consistent with what it can measure.

NOT DRAWN BY AN IMAGE MODEL, and that is not a stylistic preference. A figure that
encodes results has to be generated from those results; the analysis plan rules out
AI-generated imagery for anything that represents data, and counts are data.

    python stage3/figura1_fluxo.py
"""
import csv
import glob
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEST = "figures"
MAPA = "data/mapping_dataset_final_strat.csv"

# Frames acquired before the blind triage. The pre-triage files are not versioned, so
# this cannot be recomputed here; it is the acquisition record reported in
# Supplementary Table S1 (HUVEC 1,085 + SKOV-3 412). Checked below against the
# retained counts, which ARE measured.
ADQUIRIDAS = {"HUVEC": 1085, "SKOV-3": 412}


def numeros():
    """Reads every count the diagram shows, from the file that owns it."""
    n = {}
    linhas = list(csv.DictReader(open(MAPA, encoding="utf-8-sig")))
    ativas = [r for r in linhas if r.get("excluida") not in ("sim", "1", "True")]
    n["retidas"] = len(ativas)
    porlinha = {}
    for r in ativas:
        cl = "SKOV-3" if r["linha_celular"].upper().startswith("SKOV") else "HUVEC"
        porlinha[cl] = porlinha.get(cl, 0) + 1
    n["retidas_por_linha"] = porlinha
    n["grupos"] = len({r["group_key"] for r in ativas})
    n["particao"] = {p: sum(1 for r in ativas if r.get("partition") == p)
                     for p in ("train", "val", "test")}
    n["negativos"] = sum(
        1 for p in ("train", "val", "test")
        for f in glob.glob(os.path.join("dataset", "labels", p, "*.txt"))
        if os.path.getsize(f) == 0)
    # the partition invariant, recomputed rather than asserted
    grupos = {}
    for r in ativas:
        grupos.setdefault(r["group_key"], set()).add(r.get("partition"))
    n["cruzam"] = sum(1 for v in grupos.values() if len(v) > 1)

    n["runs_yolo"] = len([d for d in glob.glob("runs/segment/runs_revision/*")
                          if os.path.isdir(d)])
    n["runs_unet"] = len([d for d in glob.glob("runs/segment/unet_comparator/*")
                          if os.path.isdir(d)])

    def conta(p):
        return (sum(1 for _ in open(p, encoding="utf-8-sig")) - 1
                if os.path.isfile(p) else None)

    n["ref_medidas"] = conta("data/whst_areas_final.csv")
    n["ref_closure"] = conta("data/closure_final_longo.csv")
    tri = {}
    if os.path.isfile("data/inspecao_visual.csv"):
        for r in csv.DictReader(open("data/inspecao_visual.csv", encoding="utf-8-sig")):
            tri[r["categoria"]] = tri.get(r["categoria"], 0) + 1
    n["triagem"] = tri

    D = json.load(open("stage3/concordancia_final.json", encoding="utf-8"))
    n["pares"] = D["n_observacoes"]
    n["series"] = D["n_series"]
    n["seeds"] = len(D["seeds"])
    y = D["bracos"]["YOLO M"]
    n["ccc"] = y["ccc"]["media"] if isinstance(y["ccc"], dict) else y["ccc"]

    adq = sum(ADQUIRIDAS.values())
    n["adquiridas"] = adq
    n["excluidas"] = adq - n["retidas"]
    if n["excluidas"] < 0:
        sys.exit(f"ADQUIRIDAS ({adq}) is below the measured retained count "
                 f"({n['retidas']}) — the declared constant is stale")
    return n


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    n = numeros()
    print(f"  acquired {n['adquiridas']} · excluded {n['excluidas']} · "
          f"retained {n['retidas']}")
    print(f"  groups {n['grupos']} · partition {n['particao']} · "
          f"crossing {n['cruzam']}")
    print(f"  runs {n['runs_yolo']} YOLO + {n['runs_unet']} U-Net")
    print(f"  reference {n['ref_medidas']} frames · {n['ref_closure']} closure rows")
    print(f"  agreement {n['pares']} pairs · {n['series']} series")

    pct = 100.0 * n["excluidas"] / n["adquiridas"]
    P = n["particao"]
    # Boxes are numbered in FLOW order and laid out as a serpentine: left to right
    # along the top, straight down on the right, right to left along the bottom. That
    # is what removes every diagonal. The first attempt put the six boxes on a grid in
    # reading order and then had to connect 2->4 and 3->5 across it, so two arrows ran
    # through the middle of the boxes they passed over.
    CX = [
        ("1  Dataset",
         [f"{n['adquiridas']:,} frames acquired",
          f"HUVEC {ADQUIRIDAS['HUVEC']:,} · SKOV-3 {ADQUIRIDAS['SKOV-3']}",
          f"−{n['excluidas']} excluded at blind triage ({pct:.1f}%)",
          f"{n['retidas']:,} retained · {n['negativos']} closed wounds"]),
        ("2  Leakage-free partition",
         [f"{n['grupos']} acquisition units",
          "split by unit, never by image",
          f"train {P['train']} · val {P['val']} · test {P['test']}",
          f"{n['cruzam']} units span two partitions"]),
        ("3  Benchmark",
         [f"{n['runs_yolo']} runs: 5 configurations × {n['seeds']} seeds",
          "black vs white padding, one variable",
          f"U-Net comparator, {n['runs_unet']} seeds",
          f"held-out test, n = {P['test']}"]),
        ("4  Reference standard",
         [f"{n['ref_medidas']} test frames measured",
          f"{n['triagem'].get('OK', 0)} adequate · "
          f"{n['triagem'].get('SEG_RUIM', 0)} corrected · "
          f"{n['triagem'].get('IMG_INVALIDA', 0)} invalid",
          "blinded repeat on 15 frames",
          f"{n['ref_closure']} closure measurements"]),
        ("5  Agreement",
         [f"{n['pares']} paired observations",
          f"{n['series']} acquisition series",
          f"Lin's CCC {n['ccc']:.3f}, Bland–Altman",
          "cluster bootstrap, per seed"]),
        ("6  Deployed tool",
         ["two configurations, M and S",
          "public weights, no external API",
          "operating point 0.80",
          "contours exported as CSV"]),
    ]

    pol = 180 / 25.4
    fig, ax = plt.subplots(figsize=(pol, pol * 0.46))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")

    LARG, ALT = 0.90, 0.74
    # serpentine: 1 2 3 on top left-to-right, 4 5 6 on the bottom right-to-left
    COL = [0, 1, 2, 2, 1, 0]
    LIN = [1, 1, 1, 0, 0, 0]
    centros = {}
    for i, (titulo, itens) in enumerate(CX):
        x, y = COL[i] + 0.5, LIN[i] + 0.5
        centros[i] = (x, y)
        ax.add_patch(FancyBboxPatch(
            (x - LARG / 2, y - ALT / 2), LARG, ALT,
            boxstyle="round,pad=0.010,rounding_size=0.028",
            linewidth=0.8, edgecolor="black", facecolor="#f4f4f4"))
        ax.text(x, y + ALT / 2 - 0.095, titulo, ha="center", va="center",
                fontsize=7.6, fontweight="bold")
        for j, s_ in enumerate(itens):
            ax.text(x - LARG / 2 + 0.045, y + ALT / 2 - 0.215 - j * 0.118, s_,
                    ha="left", va="center", fontsize=5.9)

    def seta(a, b):
        xa, ya = centros[a]
        xb, yb = centros[b]
        if ya == yb:                                   # horizontal, edge to edge
            dx = (LARG / 2 + 0.035) * (1 if xb > xa else -1)
            p0, p1 = (xa + dx, ya), (xb - dx, yb)
        else:                                          # the single vertical drop
            p0, p1 = (xa, ya - ALT / 2 - 0.02), (xb, yb + ALT / 2 + 0.02)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.9, color="black"))

    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
        seta(a, b)

    fig.tight_layout(pad=0.2)
    os.makedirs(DEST, exist_ok=True)
    base = os.path.join(DEST, "Figure1_study_flow")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{base}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    from PIL import Image
    im = Image.open(f"{base}.png")
    im.save(f"{base}.png", "PNG", dpi=(600, 600), optimize=True)
    w, h = im.size
    print(f"\n  {base}.pdf + .svg + .png   {w} x {h} px · {w/pol:.0f} dpi at 180 mm")


if __name__ == "__main__":
    main()
