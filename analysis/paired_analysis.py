"""
Full paired statistical analysis: AI vs ImageJ closure fraction.

Reproduces the method-agreement statistics reported in the companion paper:
  - Pearson r with 95% CI (Fisher z transform)
  - Spearman rho
  - Lin's Concordance Correlation Coefficient with bootstrap 95% CI
  - Bland-Altman analysis (bias, SD of differences, 95% LoA, % within LoA)
  - Paired t-test (whole dataset)
  - Two One-Sided Tests (TOST) for equivalence at +/-0.10 and +/-0.05 margins
  - Per-timepoint breakdowns (Pearson, CCC, Bland-Altman)
  - Per-timepoint descriptive statistics (ImageJ and AI mean +/- SD, % closure)
  - Per-timepoint paired t-tests with Cohen's d_z effect size
  - Linear regression of AI on ImageJ (slope, intercept, Pearson r)

Input  : paired_imagej_vs_ai_clean.csv (deposited alongside this script).
Output : printed to stdout. Three primary analyses are run on different cuts of
         the data (raw / clipped to [0,1] / non-negative only) plus per-timepoint
         breakdowns and the additional descriptive, per-timepoint paired-t and
         regression blocks listed above.

         NOTA (decisao D1 da revisao): a desagregacao por grupo clinico
         (early-/late-onset) was REMOVED — agreement is a property of the image
         and of the algorithm, not of the biological treatment.

Usage:
    pip install pandas scipy numpy
    python3 paired_analysis.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _default_csv_path() -> Path:
    """Locate the deposited CSV relative to this script.

    Layout expected (logical, as documented in stage3/README.md):
        <root>/
            analysis/paired_analysis.py     <- this file
            paired_data/paired_imagej_vs_ai_clean.csv
    Falls back to the same directory as the script if the layout is flat
    (the case when files are downloaded individually from Zenodo).
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / 'paired_data' / 'paired_imagej_vs_ai_clean.csv',
        here / 'paired_imagej_vs_ai_clean.csv',
        Path.cwd() / 'paired_imagej_vs_ai_clean.csv',
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "paired_imagej_vs_ai_clean.csv not found. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def lin_ccc(x, y):
    """Lin's Concordance Correlation Coefficient."""
    x, y = np.asarray(x), np.asarray(y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=0), y.var(ddof=0)
    cov = np.mean((x - mx) * (y - my))
    return 2 * cov / (vx + vy + (mx - my) ** 2)


def lin_ccc_ci(x, y, n_boot=5000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for Lin's CCC (BCa-free percentile method)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = lin_ccc(x[idx], y[idx])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return lo, hi


def pearson_ci(r, n, alpha=0.05):
    """Fisher z transform 95% CI for Pearson r."""
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = z - zcrit * se, z + zcrit * se
    return np.tanh(lo_z), np.tanh(hi_z)


def tost_paired(x, y, low, high, alpha=0.05):
    """
    Two One-Sided Tests for equivalence on paired data.
    H0: mean(x - y) <= low OR mean(x - y) >= high
    H1: low < mean(x - y) < high   (equivalence)
    Reject H0 (conclude equivalence) when BOTH one-sided t-tests are
    significant at level alpha; the overall p-value is the maximum of
    the two one-sided p-values.
    """
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    md = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t_low = (md - low) / se
    t_high = (md - high) / se
    p_low = 1 - stats.t.cdf(t_low, df=n - 1)
    p_high = stats.t.cdf(t_high, df=n - 1)
    p_tost = max(p_low, p_high)
    return {
        'mean_diff': md, 'sd_diff': sd, 'n': n,
        't_low': t_low, 'p_low': p_low,
        't_high': t_high, 'p_high': p_high,
        'p_tost': p_tost,
        'equivalent': p_tost < alpha,
        'margin': (low, high),
    }


def report(df_sub, label):
    """Print the full agreement-analysis report for a given subset."""
    bar = '=' * 72
    print(f"\n{bar}")
    print(f"  {label}  (n = {len(df_sub)})")
    print(bar)

    x = df_sub['imagej'].values
    y = df_sub['ai'].values

    # Pearson
    r, p_r = stats.pearsonr(x, y)
    r_lo, r_hi = pearson_ci(r, len(x))
    print(f"\n  Pearson r        = {r:.4f}  (95% CI: {r_lo:.4f} to {r_hi:.4f})  p < {p_r:.1e}")

    # Spearman
    rho, p_rho = stats.spearmanr(x, y)
    print(f"  Spearman rho     = {rho:.4f}  p < {p_rho:.1e}")

    # Lin's CCC
    ccc = lin_ccc(x, y)
    ccc_lo, ccc_hi = lin_ccc_ci(x, y)
    print(f"  Lin's CCC        = {ccc:.4f}  (95% CI: {ccc_lo:.4f} to {ccc_hi:.4f})")

    # Bland-Altman
    d = y - x
    bias = d.mean()
    sd_d = d.std(ddof=1)
    loa_lo = bias - 1.96 * sd_d
    loa_hi = bias + 1.96 * sd_d
    bias_ci = stats.t.interval(0.95, len(d) - 1, loc=bias, scale=sd_d / np.sqrt(len(d)))
    within_loa = ((d >= loa_lo) & (d <= loa_hi)).mean() * 100
    print(f"\n  Bland-Altman:")
    print(f"    Bias (AI - ImageJ)  = {bias:+.4f}  (95% CI: {bias_ci[0]:+.4f} to {bias_ci[1]:+.4f})")
    print(f"    SD of differences   = {sd_d:.4f}")
    print(f"    95% LoA             = [{loa_lo:+.4f}, {loa_hi:+.4f}]")
    print(f"    Within LoA          = {within_loa:.1f}%")

    # Paired t-test
    t_p, p_p = stats.ttest_rel(x, y)
    print(f"\n  Paired t-test    : t = {t_p:.3f},  p = {p_p:.4f}")

    # TOST equivalence (margin +/-0.10)
    tost1 = tost_paired(y, x, low=-0.10, high=+0.10)
    print(f"\n  TOST equivalence (margin +/-0.10 of closure fraction):")
    print(f"    mean_diff = {tost1['mean_diff']:+.4f}")
    print(f"    p_lower   = {tost1['p_low']:.4f}   p_upper = {tost1['p_high']:.4f}")
    print(f"    p_TOST    = {tost1['p_tost']:.4f}   ->  Equivalent: {tost1['equivalent']}")

    # TOST with tighter margin +/-0.05
    tost2 = tost_paired(y, x, low=-0.05, high=+0.05)
    print(f"\n  TOST equivalence (margin +/-0.05 - stricter):")
    print(f"    p_TOST    = {tost2['p_tost']:.4f}   ->  Equivalent: {tost2['equivalent']}")

    return {'pearson_r': r, 'ccc': ccc, 'bias': bias,
            'loa_lo': loa_lo, 'loa_hi': loa_hi}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        '--csv', type=Path, default=None,
        help='Path to paired_imagej_vs_ai_clean.csv (auto-located by default).'
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else _default_csv_path()
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=['imagej', 'ai']).copy()
    df['diff'] = df['ai'] - df['imagej']
    df['mean'] = (df['ai'] + df['imagej']) / 2.0

    # Three analyses: raw / clipped to [0,1] / non-negative subset
    report(df, "RAW DATA (matches the companion paper's reported numbers)")

    df_clip = df.copy()
    df_clip['imagej'] = df_clip['imagej'].clip(0, 1)
    df_clip['ai']     = df_clip['ai'].clip(0, 1)
    report(df_clip, "CLIPPED to [0, 1]")

    df_pos = df[(df['imagej'] >= 0) & (df['ai'] >= 0)].copy()
    report(df_pos, "NON-NEGATIVE ONLY (excluded 8 obs with negatives)")

    # Per-timepoint breakdown on raw
    bar = '=' * 72
    print(f"\n\n{bar}\n  PER-TIMEPOINT BREAKDOWN (raw)\n{bar}")
    for t in [8, 12, 24]:
        sub = df[df['timepoint_h'] == t]
        r, _ = stats.pearsonr(sub['imagej'], sub['ai'])
        ccc = lin_ccc(sub['imagej'].values, sub['ai'].values)
        bias = (sub['ai'] - sub['imagej']).mean()
        sd_d = (sub['ai'] - sub['imagej']).std(ddof=1)
        print(f"\n  t = {t:2d}h  (n = {len(sub)})")
        print(f"    Pearson r = {r:.4f}  |  CCC = {ccc:.4f}  |  "
              f"bias = {bias:+.4f}  |  LoA = [{bias-1.96*sd_d:+.4f}, {bias+1.96*sd_d:+.4f}]")

    # ------------------------------------------------------------------
    # REMOVIDO — desagregacao por grupo clinico (Early-/Late-onset).
    #
    # A concordancia entre a
    # medida automatizada e a de referencia e propriedade da IMAGEM e do
    # ALGORITHM, not of the biological treatment: disaggregating by clinical group
    # suggests a mechanistically unjustified effect and is confounded with
    # poco, timepoint e lote. Cytometry Part A e revista de metodo, e a
    # dimensao clinica esta fora do escopo declarado.
    #
    # Independent reinforcement: the test set does not support per-arm metrics —
    # the effective n is 1 in 8 of the 10 arms.
    #
    # The clinical provenance of the images remains declared in the Methods,
    # without being used as an analysis variable. If the disaggregation is
    # reintroduzida, restaurar tambem as assercoes em
    # .github/workflows/reproduce-stats.yml.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-timepoint descriptive statistics (mean +/- SD), as percent
    # closure (consistent with the reporting convention in Section 3 of
    # the companion paper).
    # ------------------------------------------------------------------
    print(f"\n\n{bar}\n  PER-TIMEPOINT DESCRIPTIVE STATS (raw, % closure)\n{bar}")
    print(f"\n  {'Timepoint':<11}{'ImageJ (mean +/- SD)':<26}{'AI (mean +/- SD)'}")
    print(f"  {'-'*65}")
    for t in [8, 12, 24]:
        sub = df[df['timepoint_h'] == t]
        ij_m, ij_s = sub['imagej'].mean() * 100, sub['imagej'].std(ddof=1) * 100
        ai_m, ai_s = sub['ai'].mean()     * 100, sub['ai'].std(ddof=1)     * 100
        print(f"  {t:>2d}h        "
              f"{ij_m:5.1f} +/- {ij_s:5.1f}             "
              f"{ai_m:5.1f} +/- {ai_s:5.1f}")

    # ------------------------------------------------------------------
    # Per-timepoint paired t-test on (AI - ImageJ) with Cohen's d_z for
    # paired samples (standardised mean difference of within-pair
    # differences).
    # ------------------------------------------------------------------
    print(f"\n\n{bar}\n  PER-TIMEPOINT PAIRED t-TEST + COHEN's d (raw)\n{bar}")
    print(f"\n  {'Timepoint':<11}{'t':>9}{'p':>11}{'Cohen d':>13}")
    print(f"  {'-'*44}")
    for t in [8, 12, 24]:
        sub = df[df['timepoint_h'] == t]
        diffs = sub['ai'].values - sub['imagej'].values
        t_stat, p_val = stats.ttest_rel(sub['ai'], sub['imagej'])
        cohen_d = np.mean(diffs) / np.std(diffs, ddof=1)
        print(f"  {t:>2d}h     {t_stat:>+8.3f}  {p_val:>9.4f}  {cohen_d:>+10.3f}")
    print(f"\n  Test convention: paired_t(AI, ImageJ); positive t / d  =>  AI > ImageJ.")
    print(f"  Cohen's d_z = mean(AI - ImageJ) / SD(AI - ImageJ); small |d| < 0.20.")

    # ------------------------------------------------------------------
    # Linear regression of AI on ImageJ (whole dataset, n = 225)
    # ------------------------------------------------------------------
    print(f"\n\n{bar}\n  LINEAR REGRESSION  AI = slope * ImageJ + intercept  (raw, n = 225)\n{bar}")
    slope, intercept, r_lr, p_lr, stderr_slope = stats.linregress(df['imagej'], df['ai'])
    print(f"\n  slope     = {slope:.4f}   (SE = {stderr_slope:.4f})")
    print(f"  intercept = {intercept:.4f}")
    print(f"  Pearson r = {r_lr:.4f}      p < {p_lr:.1e}")
    print(f"\n  Interpretation: slope < 1 indicates mild compression of the AI dynamic")
    print(f"  range relative to ImageJ (slight overestimation at low closure, slight")
    print(f"  underestimation at high closure); the systematic component is small")
    print(f"  relative to the random component reported in the Bland-Altman block.")


if __name__ == '__main__':
    main()
