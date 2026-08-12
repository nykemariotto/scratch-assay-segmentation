
# -*- coding: utf-8 -*-
"""
stage3/_agreement_stats.py — the agreement estimators, in one place.

WHY THIS MODULE EXISTS. `stage3/stats_for_224.py` produced the numbers the
manuscript's Discussion already reports. `stage3/agreement_final.py` needs the
same numbers plus a larger set (by timepoint, by cell line, regression, % within
the LoA) for the Results and the captions. Reimplementing `ccc` and `tost` in the
second script would create two definitions of the same statistic, which diverge
without warning — and the divergence would surface as a discrepancy between the
Results and the Discussion of the same paper.

One definition, imported by both.

TOST: two one-sided tests on the paired difference. Equivalence at a margin δ is
declared when the (1−2α) CI of the mean difference falls entirely inside
[−δ, +δ]. Margins pre-specified in §2.8: ±0.10 (primary) and ±0.05 (sensitivity).
"""
import math
import statistics as st

import numpy as np
from scipy import stats as sps


def ccc(x, y):
    """Lin's concordance correlation coefficient."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * ((x - x.mean()) * (y - y.mean())).mean() / den)


def tost(dif, delta, alpha=0.05):
    """equivalence at ±delta: the (1-2*alpha) CI of the mean inside [-delta, +delta]."""
    n = len(dif)
    m, s = st.mean(dif), st.stdev(dif)
    ep = s / math.sqrt(n)
    tc = sps.t.ppf(1 - alpha, n - 1)
    lo, hi = m - tc * ep, m + tc * ep
    p1 = sps.t.sf((m + delta) / ep, n - 1)          # H0: diff <= -delta
    p2 = sps.t.cdf((m - delta) / ep, n - 1)         # H0: diff >= +delta
    return {"p": max(p1, p2), "lo": lo, "hi": hi,
            "equivalente": (lo > -delta) and (hi < delta)}
