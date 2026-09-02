# `analysis/` — the SUBMITTED version's agreement analysis (superseded)

> **This directory is a historical record, not the analysis the paper reports.**
> It holds the method-agreement analysis of the **originally submitted** version: 225
> observations from 75 HUVEC wells, measured against three annotators' ImageJ readings,
> with a model trained on a partition that turned out to leak and with no restriction to
> a held-out set.
>
> The analysis the manuscript reports is `stage3/agreement_final.py`, on the 97 paired
> observations from 45 acquisition series that the ten leakage-free runs return in common,
> across both cell lines. See [`stage3/README.md`](../stage3/README.md).
>
> It is kept because the submitted version is part of the record. Nothing here should be
> cited or re-run as a current result.

## Files

| File | What it does |
|---|---|
| `paired_analysis.py` | The agreement statistics of the submitted version: Pearson, Spearman, Lin's CCC, Bland–Altman, TOST |
| `generate_figures.py` | Figures 3–5 **of the submitted version**, at 300 DPI |
| `parse_prism.py` | Ancillary utility — extracts the paired CSV from GraphPad Prism `.pzfx` source files. The `.pzfx` files are not redistributed |

The figure names (`Figure_3_method_comparison.png`, `Figure_4_correlation_scatter.png`,
`Figure_5_bland_altman.png`) are the submitted version's numbering. The Figures 3–5 of the
revised manuscript are different figures, produced by `stage3/figures_agreement.py`.

## Running it

```bash
pip install -r requirements-analysis.txt
python analysis/paired_analysis.py
python analysis/generate_figures.py   # writes the three PNGs to the CURRENT directory
```

`generate_figures.py` calls `savefig` with bare file names, so the PNGs land wherever you
run it from, not in `analysis/`.

Dependencies come from [`requirements-analysis.txt`](../requirements-analysis.txt) — the
same pins as `requirements.txt`, so this directory runs under the environment the rest of
the repository declares. It needs `matplotlib` in addition, which that file does not pin;
any recent version renders these figures.

`paired_analysis.py` uses a fixed bootstrap seed (`seed=42`) for the 5,000-replicate CCC
confidence interval, so its output is deterministic.

## Continuous integration

`.github/workflows/reproduce-stats.yml` can re-run `paired_analysis.py`, but **only on
manual dispatch**. It was deliberately taken off push and pull-request triggers: while it
ran automatically, the repository was certifying as reproducible exactly the numbers the
revision retracted. Its header carries the same warning.

The workflow that runs on push is
[`reproduce-agreement.yml`](../.github/workflows/reproduce-agreement.yml), which checks 24
values of the current analysis against the ones the manuscript reports.
