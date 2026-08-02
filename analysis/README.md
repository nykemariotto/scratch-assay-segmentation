# `analysis/` — Statistical analysis pipeline

Reproduces the method-agreement statistics and figures reported in the companion paper (Section 3 and Figures 3–5).

## Files

| File | Purpose |
|---|---|
| `paired_analysis.py` | Computes Pearson r, Spearman ρ, Lin's CCC, Bland-Altman, paired *t*-test, and TOST equivalence on the deposited paired CSV. Runs in ~30 seconds on a laptop. |
| `generate_figures.py` | Produces Figures 3 (bar chart by timepoint), 4 (correlation scatter), and 5 (Bland-Altman plot) at 300 DPI PNG. Requires the same paired CSV. |
| `parse_prism.py` | **Ancillary utility** — extracts the paired CSV from GraphPad Prism `.pzfx` source files. The `.pzfx` source files are NOT redistributed (they contain intermediate annotation drafts); typical reproducibility users go straight to `paired_analysis.py`. |

## Quick run

From the repository root:

```bash
pip install scipy==1.13 pandas==2.2 numpy matplotlib==3.8
python3 analysis/paired_analysis.py     # prints statistics to stdout
python3 analysis/generate_figures.py    # writes Figure_3/4/5_*.png to ./analysis
```

For `parse_prism.py` (only if you have your own Prism `.pzfx` files):

```bash
python3 analysis/parse_prism.py \
    --input-dir  /path/to/folder/with/pzfx_files \
    --output-csv ./paired_data_raw.csv
```

## Determinism

`paired_analysis.py` uses a fixed bootstrap seed (`seed=42`) for the 5,000-replicate CCC confidence interval. All 56 reported statistics reproduce bit-for-bit on Linux, macOS, and Windows.

The GitHub Actions workflow at `.github/workflows/reproduce-stats.yml` re-runs this script on every push and fails the build if any value drifts.

## Dependencies

| Package | Tested version |
|---|---|
| Python | 3.11 |
| `scipy` | 1.13 |
| `pandas` | 2.2 |
| `numpy` | (any recent) |
| `matplotlib` | 3.8 (only required by `generate_figures.py`) |
