# -*- coding: utf-8 -*-
"""
stage3/ap_core.py — mask AP in the COCO convention, implemented here.

WHY NOT JUST USE ULTRALYTICS' val(). It returns the mAP aggregated over the whole
test set. What is needed is a CONFIDENCE INTERVAL THAT RESAMPLES GROUPS (37
groups, 234 images), and resampling requires recomputing AP inside every
resample. The aggregate number alone makes that impossible.

The solution is to split it in two phases:

  phase 1 (GPU, once per run)   -> match predictions against ground truth and
                                    store, per image, the MATCHING RECORDS: the
                                    score of each detection, whether it is a TP at
                                    each IoU threshold, and the number of GT.
  phase 2 (CPU, as often as needed) -> recompute AP from those records over any
                                    subset of images.

The records are tiny (a few kB per run) and hold everything AP needs. Grouped
bootstrap then reduces to resampling the keys of a dictionary.

CONVENTIONS (COCO):
  - IoU thresholds 0.50:0.05:0.95 (10 values); mAP@50 is index 0, mAP@75 is 5
  - greedy matching by descending score, each GT used at most once
  - AP by 101-point interpolation
  - an image with no GT contributes only false positives (it does not enter recall)
"""
import numpy as np

IOU_THRS = np.round(np.arange(0.5, 1.0, 0.05), 2)     # 0.50 … 0.95
IDX_50, IDX_75 = 0, 5


# --------------------------------------------------------------- geometry
def rasteriza_poly(segs, w, h):
    """COCO segmentation (list of flattened polygons) -> boolean mask."""
    import cv2
    m = np.zeros((h, w), np.uint8)
    if not segs:
        return m.astype(bool)
    if isinstance(segs[0], (int, float)):
        segs = [segs]
    for s in segs:
        p = np.asarray(s, np.float64).reshape(-1, 2)
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m.astype(bool)


def iou_mask(a, b):
    """IoU between two boolean masks of the same size."""
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    return inter / (np.count_nonzero(a) + np.count_nonzero(b) - inter)


def matriz_iou(preds, gts):
    """[n_pred x n_gt] of IoU. Pre-filters by bbox: a 5 M pixel mask is
    expensive, and the overwhelming majority of pairs do not even touch."""
    n, m = len(preds), len(gts)
    M = np.zeros((n, m), np.float32)
    if n == 0 or m == 0:
        return M

    def bbox(x):
        ys, xs = np.nonzero(x)
        if ys.size == 0:
            return None
        return xs.min(), ys.min(), xs.max(), ys.max()

    bp = [bbox(p) for p in preds]
    bg = [bbox(g) for g in gts]
    for i in range(n):
        if bp[i] is None:
            continue
        for j in range(m):
            if bg[j] is None:
                continue
            if (bp[i][0] > bg[j][2] or bp[i][2] < bg[j][0] or
                    bp[i][1] > bg[j][3] or bp[i][3] < bg[j][1]):
                continue                       # disjoint bboxes -> IoU 0
            M[i, j] = iou_mask(preds[i], gts[j])
    return M


# --------------------------------------------------------------- matching
def casa_imagem(scores, iou_pg, thrs=IOU_THRS):
    """Greedy matching by score, one GT per detection, at each threshold.

    Returns a serialisable dict:
      scores : list, in descending order
      tp     : list of lists (len(thrs) x n_pred) of 0/1
      n_gt   : integer
    """
    scores = np.asarray(scores, np.float64)
    ordem = np.argsort(-scores)
    scores = scores[ordem]

    # BUG FIXED (review, 2026-07-28). The earlier version used `iou_pg.size` as
    # the guard:
    #     n_pred, n_gt = iou_pg.shape if iou_pg.size else (len(scores), 0)
    # An image WITH ground truth and NO detection produces a (0, m) matrix, whose
    # .size is ZERO — so n_gt became 0 and that ground truth vanished from the
    # recall denominator. Effect: the model was NOT penalised for finding nothing,
    # and AP came out inflated. Silent, and worse the worse the model is.
    # The shape, not the size, is what knows how many GT exist.
    n_pred = int(scores.shape[0])
    n_gt = int(iou_pg.shape[1]) if iou_pg.ndim == 2 else 0
    if n_pred and n_gt:
        iou_pg = iou_pg[ordem]

    tp = np.zeros((len(thrs), n_pred), np.uint8)
    for k, t in enumerate(thrs):
        used = np.zeros(n_gt, bool)
        for i in range(n_pred):
            if n_gt == 0:
                break
            cand = np.where((~used) & (iou_pg[i] >= t))[0]
            if cand.size:
                j = cand[np.argmax(iou_pg[i, cand])]
                used[j] = True
                tp[k, i] = 1
    return {"scores": scores.tolist(), "tp": tp.tolist(), "n_gt": int(n_gt)}


# --------------------------------------------------------------- metrics
def average_precision(regs, k):
    """AP at threshold index k, over a collection of per-image records."""
    scores, tps, n_gt = [], [], 0
    for r in regs:
        n_gt += r["n_gt"]
        if r["scores"]:
            scores.append(np.asarray(r["scores"], np.float64))
            tps.append(np.asarray(r["tp"][k], np.float64))
    if n_gt == 0:
        return float("nan")           # with no GT, AP is undefined, not zero
    if not scores:
        return 0.0
    s = np.concatenate(scores)
    t = np.concatenate(tps)
    o = np.argsort(-s)
    t = t[o]
    ctp = np.cumsum(t)
    cfp = np.cumsum(1 - t)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    # envelope: the maximum precision to the right of each point
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    r101 = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, r101, side="left")
    p101 = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(p101.mean())


def mapa_5095(regs):
    v = [average_precision(regs, k) for k in range(len(IOU_THRS))]
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def prf(regs, conf, k=IDX_50):
    """Precision, recall and F1 at a confidence threshold — these DO depend on it.

    That is the distinction that matters: mAP integrates the whole curve, while
    these three are pointwise estimates.
    """
    tp = fp = 0
    n_gt = 0
    for r in regs:
        n_gt += r["n_gt"]
        for s, t in zip(r["scores"], r["tp"][k]):
            if s >= conf:
                tp += int(t)
                fp += 1 - int(t)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    return {"precision": p, "recall": rc, "f1": f1, "tp": tp, "fp": fp, "n_gt": n_gt}


def metricas(regs, conf=0.8):
    m = prf(regs, conf)
    m.update({"mAP50": average_precision(regs, IDX_50),
              "mAP75": average_precision(regs, IDX_75),
              "mAP50_95": mapa_5095(regs)})
    return m


# ------------------------------------------------------- cluster bootstrap
def cluster_bootstrap(regs_por_img, grupo_de, fn, B=2000, seed=42, alpha=0.05):
    """Percentile CI resampling GROUPS, not images.

    The test set has 234 images in 37 groups, and frames of the same field are
    highly correlated. Resampling images treats them as independent and narrows
    the interval artificially. Here the resampling unit is the group: 37 groups
    are drawn with replacement and ALL images of each drawn group are taken.

    Returns {'obs','lo','hi','B_validos'}.
    """
    porg = {}
    for k in regs_por_img:
        porg.setdefault(grupo_de[k], []).append(k)
    chaves = sorted(porg)
    rng = np.random.default_rng(seed)

    obs = fn([regs_por_img[k] for k in regs_por_img])
    vals = []
    for _ in range(B):
        sorteio = rng.integers(0, len(chaves), len(chaves))
        sub = []
        for i in sorteio:
            sub.extend(regs_por_img[k] for k in porg[chaves[i]])
        v = fn(sub)
        if v == v:                       # discard NaN (a resample with no GT at all)
            vals.append(v)
    if not vals:
        return {"obs": obs, "lo": float("nan"), "hi": float("nan"), "B_validos": 0}
    vals = np.sort(vals)
    return {"obs": obs,
            "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)),
            "B_validos": len(vals)}


def cluster_bootstrap_pareado(regs_A, regs_B, grupo_de, fn, B=2000, seed=42, alpha=0.05):
    """CI of the DIFFERENCE A−B, resampling the same groups for both.

    WHY NOT COMPARE TWO SEPARATE CIs. Interval overlap is NOT a test of
    difference — it is far too conservative, and calls "indistinguishable" things
    that are distinguishable. Both models are evaluated ON THE SAME images, and
    much of the variance belongs to the test set rather than to the model: a hard
    field drags both down together. Resampling the same groups for A and B and
    taking the difference INSIDE each resample cancels that shared variance.

    If the CI of the difference excludes zero, there is a difference at level alpha.
    """
    porg = {}
    for k in regs_A:
        porg.setdefault(grupo_de[k], []).append(k)
    chaves = sorted(porg)
    rng = np.random.default_rng(seed)

    obs = fn([regs_A[k] for k in regs_A]) - fn([regs_B[k] for k in regs_B])
    difs = []
    for _ in range(B):
        sorteio = rng.integers(0, len(chaves), len(chaves))
        ka = []
        for i in sorteio:
            ka.extend(porg[chaves[i]])
        a, b = fn([regs_A[k] for k in ka]), fn([regs_B[k] for k in ka])
        if a == a and b == b:
            difs.append(a - b)
    if not difs:
        return {"obs": obs, "lo": float("nan"), "hi": float("nan"),
                "exclui_zero": False, "B_validos": 0}
    difs = np.sort(difs)
    lo = float(np.quantile(difs, alpha / 2))
    hi = float(np.quantile(difs, 1 - alpha / 2))
    return {"obs": float(obs), "lo": lo, "hi": hi,
            "exclui_zero": bool(lo > 0 or hi < 0), "B_validos": len(difs)}


def cluster_bootstrap_config(regs_por_seed, grupo_de, fn, B=2000, seed=42, alpha=0.05):
    """CI of the CONFIGURATION MEAN over the seeds, resampling groups.

    WHY THIS EXISTS (review, 2026-07-28). The earlier version computed the CI over
    ONE run — `runs[len(runs)//2]`, which, with the list ordered by seed number,
    is always seed 44, not the median-performing run. Consequences:

      * the CI was centred on a number different from the mean reported in
        Table 2 (measured divergence of up to +1.08 pp);
      * seed 44 happened to be the BEST of 5 in two configurations and the WORST
        in two others, so the arbitrary choice moved the centre of the interval in
        opposite directions depending on the table row;
      * 4 of the 5 runs were discarded.

    Here the group resampling is done ONCE per iteration and applied to ALL seeds;
    the statistic of the iteration is the mean across seeds. The interval is then
    of the configuration mean — exactly the number in Table 2.

    `regs_por_seed`: list of dicts {image_key: record}, one per seed.
    """
    if not regs_por_seed:
        return {"obs": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "B_validos": 0, "n_seeds": 0}
    porg = {}
    for k in regs_por_seed[0]:
        porg.setdefault(grupo_de[k], []).append(k)
    chaves = sorted(porg)
    rng = np.random.default_rng(seed)

    def media(sel):
        v = [fn([r[k] for k in sel if k in r]) for r in regs_por_seed]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    todas = list(regs_por_seed[0])
    obs = media(todas)
    vals = []
    for _ in range(B):
        sel = []
        for i in rng.integers(0, len(chaves), len(chaves)):
            sel.extend(porg[chaves[i]])
        v = media(sel)
        if v == v:
            vals.append(v)
    if not vals:
        return {"obs": obs, "lo": float("nan"), "hi": float("nan"),
                "B_validos": 0, "n_seeds": len(regs_por_seed)}
    vals = np.sort(vals)
    return {"obs": obs, "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)),
            "B_validos": len(vals), "n_seeds": len(regs_por_seed)}


def cluster_bootstrap_pareado_config(seeds_A, seeds_B, grupo_de, fn,
                                     B=2000, seed=42, alpha=0.05):
    """CI of the difference between the MEANS of two configurations, resampling groups.

    DEFECT FIXED (2026-07-29). The earlier version compared ONE run of each
    configuration — `runs[len(runs)//2]`, which is always seed 44. On the real data
    that produced an open contradiction inside the report itself:

        section 3 (paired by seed, mean of 5):  black - white = -0.11 pp
        section 4 (seed 44 only):               white - black = +2.24 pp, with a
                                                CI excluding zero

    Seed 44 happened to be the worst of the black arm (91.87 against a mean of
    93.40). The "significant difference" was an artefact of which seed was picked,
    and it would have licensed the manuscript to claim white padding was superior
    on the strength of nothing.

    Here the group resampling is done ONCE per iteration and applied to ALL seeds
    of both configurations; the statistic is the difference between the means. Two
    variances cancel at once: that of the test set (the same groups for both) and
    that of the seed (the mean over the five).
    """
    if not seeds_A or not seeds_B:
        return {"obs": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "exclui_zero": False, "B_validos": 0}
    porg = {}
    for k in seeds_A[0]:
        porg.setdefault(grupo_de[k], []).append(k)
    chaves = sorted(porg)
    rng = np.random.default_rng(seed)

    def media(regs_por_seed, sel):
        v = [fn([r[k] for k in sel if k in r]) for r in regs_por_seed]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    todas = [k for k in seeds_A[0] if all(k in r for r in seeds_B)]
    obs = media(seeds_A, todas) - media(seeds_B, todas)
    difs = []
    for _ in range(B):
        sel = []
        for i in rng.integers(0, len(chaves), len(chaves)):
            sel.extend(porg[chaves[i]])
        a, b = media(seeds_A, sel), media(seeds_B, sel)
        if a == a and b == b:
            difs.append(a - b)
    if not difs:
        return {"obs": obs, "lo": float("nan"), "hi": float("nan"),
                "exclui_zero": False, "B_validos": 0}
    difs = np.sort(difs)
    lo = float(np.quantile(difs, alpha / 2))
    hi = float(np.quantile(difs, 1 - alpha / 2))
    return {"obs": float(obs), "lo": lo, "hi": hi,
            "exclui_zero": bool(lo > 0 or hi < 0), "B_validos": len(difs)}


def bootstrap_ingenuo_config(regs_por_seed, fn, B=2000, seed=42, alpha=0.05):
    """Resamples IMAGES, but over the configuration mean.

    It exists so that the comparison against `cluster_bootstrap_config` is
    apples-to-apples. The cluster interval used to be over the mean of the 5 seeds
    while the naive one was over ONE run — the ratio between the widths compared
    different things and came out near 1.0, hiding the effect of grouping.
    """
    chaves = sorted(regs_por_seed[0])
    rng = np.random.default_rng(seed)

    def media(sel):
        v = [fn([r[k] for k in sel if k in r]) for r in regs_por_seed]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    vals = []
    for _ in range(B):
        sel = [chaves[i] for i in rng.integers(0, len(chaves), len(chaves))]
        v = media(sel)
        if v == v:
            vals.append(v)
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "B_validos": 0}
    vals = np.sort(vals)
    return {"lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)), "B_validos": len(vals)}


def bootstrap_ingenuo(regs_por_img, fn, B=2000, seed=42, alpha=0.05):
    """Resamples IMAGES. It exists only to demonstrate how much that
    understates the uncertainty."""
    chaves = sorted(regs_por_img)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        sub = [regs_por_img[chaves[i]] for i in rng.integers(0, len(chaves), len(chaves))]
        v = fn(sub)
        if v == v:
            vals.append(v)
    if not vals:                      # guarda: np.quantile([]) levanta
        return {"lo": float("nan"), "hi": float("nan"), "B_validos": 0}
    vals = np.sort(vals)
    return {"lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)), "B_validos": len(vals)}
