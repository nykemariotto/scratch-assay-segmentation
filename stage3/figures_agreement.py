
# -*- coding: utf-8 -*-
"""
stage3/figures_agreement.py — Figuras 3, 4 e 5 a partir do pareamento leakage-free.

All three used to plot the 225 observations of the withdrawn analysis, while the
captions and the text already described the 97. Here they come from the real data,
by code, without a single generated image.

SOURCE: `stage3/agreement_final_long.csv` (970 rows = 97 observations × 5 seeds
× 2 arms), produced by `stage3/agreement_final.py`. The values in the captions
come from `stage3/agreement_final.json`, read here — none is typed in.

THE DELICATE POINT: what to plot when there are 5 seeds. The statistics (bias, LoA,
regression) are computed PER SEED and then averaged. Plotting only the mean across
seeds would produce a tighter cloud than the LoA lines describe — the figure would
say the method is more consistent than the number claims. Plotting all 485
seed-observations would visually inflate an n of 97.

Solution: one point per observation, at the mean across seeds, PLUS a thin bar from
minimum to maximum across the seeds. The reader sees the true n and the variability
the lines describe. The caption says exactly that.

ACCESSIBILITY: Okabe-Ito palette (colourblind-safe), distinct markers as well as
colour (legible in greyscale), and no significance stars — the Cytometry Part A
author guidelines explicitly discourage them, asking for effect sizes and intervals
instead.

OUTPUT: SVG (vector, for the journal), PNG at 600 dpi, and the CSV of what was
plotted.

    python stage3/figures_agreement.py
"""
import csv
import json
import os
import sys
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(AQUI))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEST = "figures"
os.makedirs(DEST, exist_ok=True)
BRACO = "YOLO M"          # a configuração implantada

# Okabe-Ito, muted.
#
# The hues are Okabe-Ito, which is the colourblind-safe set; the saturation is not.
# Published guidance on figure colour — Crameri, Current Protocols 2024, in Wiley's
# own series — argues against fully saturated hues: they are visually intense, they
# clash with each other, and they are the first thing to collapse when the figure is
# printed or photocopied in greyscale. The Nature-family convention is muted
# qualitative palettes rather than primaries.
#
# Here the cost of muting is nil, because the encoding is already redundant: in
# Figures 4 and 5 each group carries its own MARKER SHAPE as well as its colour, so
# hue is the second channel, not the only one. Blending 45% toward mid grey keeps the
# hues apart and takes the intensity out.
#
# Figure 3 uses no colour at all — there the grouping is on the x axis and colour
# encoded nothing. See the note on that panel.
def _suaviza(hexa, k=0.45, cinza=0x88):
    v = int(hexa[1:], 16)
    return "#" + "".join(
        f"{round((v >> s & 0xFF) * (1 - k) + cinza * k):02x}" for s in (16, 8, 0))


AZUL, LARANJA, VERDE, ROSA, AREIA = (
    _suaviza(c) for c in ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#F0E442"))
CINZA, PRETO = "#999999", "#000000"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "svg.fonttype": "none",
})

D = json.load(open("stage3/agreement_final.json", encoding="utf-8"))
Y = D["bracos"][BRACO]
with open("stage3/agreement_final_long.csv", encoding="utf-8-sig") as fh:
    L = [r for r in csv.DictReader(fh) if r["braco"] == BRACO]

# key -> (cell line, timepoint, reference, [ai per seed])
obs = defaultdict(lambda: {"ai": []})
for r in L:
    k = (r["series_key"], r["campo"], r["timepoint_h"])
    o = obs[k]
    o["cell_line"] = r["cell_line"]
    o["tp"] = int(r["timepoint_h"])
    o["ref"] = float(r["referencia"])
    o["ai"].append(float(r["ai"]))
K = sorted(obs, key=lambda k: (obs[k]["cell_line"], obs[k]["tp"]))
for k in K:
    o = obs[k]
    o["ai_m"] = float(np.mean(o["ai"]))
    o["ai_lo"], o["ai_hi"] = min(o["ai"]), max(o["ai"])
    o["dif"] = o["ai_m"] - o["ref"]
    o["dif_lo"] = min(a - o["ref"] for a in o["ai"])
    o["dif_hi"] = max(a - o["ref"] for a in o["ai"])
    o["media"] = (o["ai_m"] + o["ref"]) / 2
print(f"{len(K)} observations · {len(L)} rows ({len(L)//len(K)} seeds)")

GRUPOS = [("HUVEC", 8, AZUL, "o"), ("HUVEC", 12, LARANJA, "s"),
          ("HUVEC", 24, VERDE, "^"), ("SKOV-3", 24, ROSA, "D"),
          ("SKOV-3", 48, PRETO, "v")]


def rot(cl, tp):
    return f"{cl} {tp} h"


def salva(fig, nome):
    # PDF is what the journal asks for. Wiley, "Guidelines for the Preparation of
    # Figures": these panels are line art, and for line art the preferred file types
    # are EPS and PDF — vector, so there is no resolution to get wrong and no floor
    # to clear. SVG stays because it is the format that edits; PNG stays as the
    # raster fallback, at the 600 dpi the same guideline asks of line art.
    for ext in ("pdf", "svg", "png"):
        p = os.path.join(DEST, f"{nome}.{ext}")
        fig.savefig(p, dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"  {nome}.pdf + .svg + .png")


# ═══════════════════════════════════ Figure 3 — closure by timepoint
fig, ax = plt.subplots(figsize=(5.2, 3.1))
larg, pos, rots = 0.36, [], []
for i, (cl, tp, cor, mk) in enumerate(GRUPOS):
    ks = [k for k in K if obs[k]["cell_line"] == cl and obs[k]["tp"] == tp]
    ref = np.array([obs[k]["ref"] for k in ks]) * 100
    ai = np.array([obs[k]["ai_m"] for k in ks]) * 100
    ax.bar(i - larg / 2, ref.mean(), larg, yerr=ref.std(ddof=1), color=CINZA,
           edgecolor=PRETO, linewidth=0.6, capsize=2.5,
           error_kw={"linewidth": 0.7}, label="Reference standard" if i == 0 else None)
    # NO COLOUR IN THIS PANEL, and not as a matter of taste.
    #
    # It began as the per-group `cor` from GRUPOS, which encodes the timepoint. That
    # is right for Figures 4 and 5, where every point has to be traceable to its
    # group, and wrong here: the grouping is already on the x axis, so the second bar
    # came out blue, orange, green, pink and black under a legend showing one blue
    # swatch labelled "Automated workflow" — correct for one group out of five.
    #
    # Collapsing that to a single blue fixed the legend but left colour carrying no
    # information at all. This panel encodes exactly one contrast, reference against
    # automated, and fill plus texture carry it completely. A hue on top of that is
    # decoration, and a saturated one is decoration that costs: it is the first thing
    # to fail in greyscale, which the analysis plan requires, and Wiley's own guidance
    # on figure colour (Crameri, Current Protocols 2024) argues the same way.
    ax.bar(i + larg / 2, ai.mean(), larg, yerr=ai.std(ddof=1), color="white",
           edgecolor=PRETO, linewidth=0.6, capsize=2.5, hatch="///",
           error_kw={"linewidth": 0.7}, label="Automated workflow" if i == 0 else None)
    pos.append(i)
    rots.append(f"{rot(cl, tp)}\n(n = {len(ks)})")
ax.set_xticks(pos)
ax.set_xticklabels(rots)
ax.set_ylabel("Wound closure (%)")
ax.set_ylim(0, 118)
ax.legend(frameon=False, loc="upper left", ncol=2)
# no divider between the cell lines: the x labels already carry "HUVEC" and "SKOV-3",
# so the rule was drawing a boundary the reader can read straight off the axis
salva(fig, "Figure3_closure_by_timepoint")

# ═══════════════════════════════════ Figure 4 — agreement
fig, ax = plt.subplots(figsize=(5.6, 3.9))
lo, hi = -0.35, 1.12
ax.plot([lo, hi], [lo, hi], "--", color=CINZA, linewidth=0.9, label="Line of identity",
        zorder=1)
s, b = Y["slope"]["media"], Y["intercept"]["media"]
ax.plot([lo, hi], [s * lo + b, s * hi + b], "-", color=PRETO, linewidth=1.1,
        label=f"Regression (slope {s:.3f}, intercept {b:.3f})", zorder=2)
for cl, tp, cor, mk in GRUPOS:
    ks = [k for k in K if obs[k]["cell_line"] == cl and obs[k]["tp"] == tp]
    for k in ks:
        ax.plot([obs[k]["ref"]] * 2, [obs[k]["ai_lo"], obs[k]["ai_hi"]],
                "-", color=cor, linewidth=0.5, alpha=0.5, zorder=3)
    ax.scatter([obs[k]["ref"] for k in ks], [obs[k]["ai_m"] for k in ks],
               s=17, marker=mk, facecolor=cor, edgecolor=PRETO, linewidth=0.35,
               label=rot(cl, tp), zorder=4)
ax.set_xlabel("Reference standard, closure fraction")
ax.set_ylabel("Automated workflow, closure fraction")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5),
          fontsize=6.8, handletextpad=0.5)
ax.text(0.03, 0.97, f"n = {len(K)}\nr = {Y['r']['media']:.3f} ± {Y['r']['dp']:.3f}\n"
        f"CCC = {Y['ccc']['media']:.3f} ± {Y['ccc']['dp']:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=7)
salva(fig, "Figure4_agreement_scatter")

# ═══════════════════════════════════ Figure 5 — Bland-Altman
fig, ax = plt.subplots(figsize=(6.2, 3.4))
vies, blo, bhi = Y["vies"]["media"], Y["loa_lo"]["media"], Y["loa_hi"]["media"]
ax.axhline(0, color=CINZA, linewidth=0.6, linestyle=":")
ax.axhline(vies, color="#D55E00", linewidth=1.1,
           label=f"Bias {vies:+.3f}")
for v, r in ((blo, "Lower"), (bhi, "Upper")):
    ax.axhline(v, color=CINZA, linewidth=0.9, linestyle="--",
               label=f"95% limits of agreement ({blo:+.3f}, {bhi:+.3f})"
               if r == "Lower" else None)
for cl, tp, cor, mk in GRUPOS:
    ks = [k for k in K if obs[k]["cell_line"] == cl and obs[k]["tp"] == tp]
    for k in ks:
        ax.plot([obs[k]["media"]] * 2, [obs[k]["dif_lo"], obs[k]["dif_hi"]],
                "-", color=cor, linewidth=0.5, alpha=0.5, zorder=3)
    ax.scatter([obs[k]["media"] for k in ks], [obs[k]["dif"] for k in ks],
               s=17, marker=mk, facecolor=cor, edgecolor=PRETO, linewidth=0.35,
               label=rot(cl, tp), zorder=4)
dentro = sum(1 for k in K if blo <= obs[k]["dif"] <= bhi)
ax.set_xlabel("Mean of the two measurements, closure fraction")
ax.set_ylabel("Automated − reference")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5),
          fontsize=6.8, handletextpad=0.5)
ax.text(0.02, 0.93, f"n = {len(K)}", transform=ax.transAxes, fontsize=7)
salva(fig, "Figure5_bland_altman")

# ═══════════════════════════════════ the CSV of what was plotted
p = os.path.join(DEST, "figures_plotted_data.csv")
with open(p, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["series_key", "campo", "timepoint_h", "cell_line", "referencia",
                "ai_media_5seeds", "ai_min", "ai_max", "diferenca",
                "media_dos_dois"])
    for k in K:
        o = obs[k]
        w.writerow([k[0], k[1], k[2], o["cell_line"], f"{o['ref']:.6f}",
                    f"{o['ai_m']:.6f}", f"{o['ai_lo']:.6f}", f"{o['ai_hi']:.6f}",
                    f"{o['dif']:.6f}", f"{o['media']:.6f}"])
print(f"  {p} ({len(K)} rows)")
print(f"\n  within the LoA: {dentro}/{len(K)} = {100*dentro/len(K):.1f}% "
      f"(the text reports {Y['dentro_loa']['media']:.1f}%, the per-seed mean)")
