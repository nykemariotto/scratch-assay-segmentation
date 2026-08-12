
# -*- coding: utf-8 -*-
"""
stage4/intraobs_ci.py — intraobserver reproducibility, with CIs and WITH OUTPUT ON DISK.

GRRAS item 13: *"Report estimates of reliability and agreement including measures
of statistical uncertainty."*

INPUT: `data/correction_agreement.csv`, the `iou_pass1_vs_pass2` column and the
areas of both passes. These are the BLINDED re-corrections — the operator knew
neither which frames had been drawn nor how they had corrected them the first time.

TWO THINGS CHANGED ON 2026-07-31.

1 · EXCLUSION OF THE DEGENERATE PAIRS. The earlier version computed over all 14
    pairs. Four of them do not measure contour reproducibility:

      · THREE have ZERO area on both passes. They score IoU = 1.0 by the
        "empty over empty" convention — they agree that "there is nothing here",
        which is a different thing. It is the same convention that inflates the
        detector's 1.0000 in a closed well.
      · ONE (`f8f76efe0c`) has an identical area on both passes and an IoU of
        exactly 1.0000. For two independent manual corrections of a visible wound
        that is implausible; it looks like ROI reuse, not re-correction.

    The script reports both sets. The exclusion is visible, not implicit.

2 · THE OUTPUT IS PERSISTED. The script used to compute, print, and let documents
    copy from the console. The number ended up in EIGHT documents before anyone
    checked how it had been computed — and it was inflated. It now writes
    `stage3/intraobs_ci.json` (the estimates) and `stage3/intraobs_pares.csv`
    (pair by pair, marking who was excluded and why).

    The rule this establishes: **every number that appears in more than one
    document needs a single source on disk.**

HOW TO READ IT, with small n:
  * the CI of the MEDIAN is a staircase, not continuous — the bootstrap median can
    only take values already present in the sample. Reported, but with the mean
    alongside;
  * the CCC is dominated by the range of the areas. With few pairs and a wide
    range it is high almost by construction; saying so is more honest than letting
    the number speak for itself.

Percentile bootstrap, B = 10000, fixed seed, resampling the PAIRS.

    python stage4/intraobs_ci.py
"""
import csv
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

B, SEED, ALPHA = 10000, 42, 0.05
SAIDA_JSON = os.path.join("stage3", "intraobs_ci.json")
SAIDA_CSV = os.path.join("stage3", "intraobs_pares.csv")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def lin_ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return (float(2 * ((x - x.mean()) * (y - y.mean())).mean() / den)
            if den else float("nan"))


def ic(vals, alpha=ALPHA):
    v = np.asarray(vals, float)
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


R = list(csv.DictReader(open("data/correction_agreement.csv", encoding="utf-8-sig")))
todos = []
for r in R:
    i, x1, x2 = (num(r["iou_pass1_vs_pass2"]), num(r["area_pct_pass1"]),
                 num(r["area_pct_pass2"]))
    if i is None or x1 is None or x2 is None:
        continue
    if x1 == 0 and x2 == 0:
        motivo = "empty-empty: IoU=1 by convention, does not measure a contour"
    elif i >= 1.0 - 1e-9:
        motivo = "IoU exactly 1.0000 with a real area: implausible"
    else:
        motivo = ""
    todos.append({"base": r["base"], "iou": i, "area_pass1": x1, "area_pass2": x2,
                  "excluido": int(bool(motivo)), "motivo_exclusao": motivo})

limpos = [p for p in todos if not p["excluido"]]
print("=" * 72)
print("INTRAOBSERVER REPRODUCIBILITY · blinded re-correction")
print(f"percentile bootstrap · B = {B} · seed = {SEED}")
print("=" * 72)
print(f"  pairs measured on both passes  : {len(todos)}")
for p in todos:
    if p["excluido"]:
        print(f"    excluded  {p['base'][:34]:<36} {p['motivo_exclusao']}")
print(f"  pairs used in the estimate     : {len(limpos)}")


def estima(conj, rot):
    n = len(conj)
    iou = np.array([p["iou"] for p in conj])
    a1 = np.array([p["area_pass1"] for p in conj])
    a2 = np.array([p["area_pass2"] for p in conj])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (B, n))
    med_b, mean_b = np.median(iou[idx], axis=1), iou[idx].mean(axis=1)
    ccc_b = np.array([lin_ccc(a1[i], a2[i]) for i in idx])
    vies_b = (a2[idx] - a1[idx]).mean(axis=1)
    out = {"conjunto": rot, "n": n,
           "iou_mediana": [float(np.median(iou)), *ic(med_b)],
           "iou_media": [float(iou.mean()), *ic(mean_b)],
           "ccc_areas": [lin_ccc(a1, a2), *ic(ccc_b)],
           "vies_pp": [float((a2 - a1).mean()), *ic(vies_b)],
           "amplitude_area_pct": [float(a1.min()), float(a1.max())],
           "valores_distintos_mediana_bootstrap": int(len(np.unique(med_b)))}
    print(f"\n-- {rot} (n = {n}) --")
    print(f"{'estimate':<22} {'point':>9}  {'95% CI':>24}")
    for k, r in (("iou_mediana", "IoU · median"), ("iou_media", "IoU · mean"),
                 ("ccc_areas", "Lin CCC (areas)"), ("vies_pp", "bias (p2-p1), pp")):
        p, lo, hi = out[k]
        print(f"{r:<22} {p:>9.4f}  [{lo:>9.4f}, {hi:>9.4f}]")
    return out


res = {"todos_os_pares": estima(todos, "all pairs (INFLATED - do not use)"),
       "pares_limpos": estima(limpos, "clean pairs (the ones that count)")}

lp = res["pares_limpos"]
print(f"\ndistinct values of the bootstrap median: "
      f"{lp['valores_distintos_mediana_bootstrap']}")
print("  -> with small n the CI of the median is a staircase, not continuous; the")
print("     mean sits alongside because it varies smoothly and its interval reads.")
print(f"range of the areas: {lp['amplitude_area_pct'][0]:.2f} - "
      f"{lp['amplitude_area_pct'][1]:.2f} % of the image")
print("  -> the CCC is high partly because the range is wide; saying so is more")
print("     honest than letting the number speak for itself.")

os.makedirs("stage3", exist_ok=True)
with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(todos[0].keys()))
    w.writeheader()
    w.writerows(todos)
json.dump({"fonte": "data/correction_agreement.csv", "B": B, "seed": SEED, "alpha": ALPHA,
           "resultados": res,
           "nota": "the `pares_limpos` values are the ones that go to the "
                   "manuscript; `todos_os_pares` is kept on record because it is "
                   "what was published before the correction"},
          open(SAIDA_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\nwrote {SAIDA_CSV} and {SAIDA_JSON}")

m, mlo, mhi = lp["iou_mediana"]
a, alo, ahi = lp["iou_media"]
c, clo, chi = lp["ccc_areas"]
v, vlo, vhi = lp["vies_pp"]
print(f"""
SENTENCE FOR THE MANUSCRIPT (values from `pares_limpos`)
  On the {lp['n']} usable blinded repeat pairs the median intersection over union
  was {m:.3f} (95% CI {mlo:.3f}-{mhi:.3f}) and the mean was {a:.3f}
  (95% CI {alo:.3f}-{ahi:.3f}); Lin's concordance correlation coefficient between
  the two sets of measured areas was {c:.3f} (95% CI {clo:.3f}-{chi:.3f}), with a
  mean bias of {v:+.2f} percentage points (95% CI {vlo:+.2f} to {vhi:+.2f}).
  Intervals are percentile bootstrap over the pairs (B = {B}, fixed seed); with a
  sample of this size they should be read as indicative rather than precise.""")
