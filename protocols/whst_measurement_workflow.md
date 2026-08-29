# Bulk measurement workflow + CSV structure — WHST
## Manuscript 4336348, Cytometry Part A

> Operational guide for measuring hundreds of images consistently and reproducibly, and the structure
> of the CSV that consolidates everything for the paired analysis.

> **Note, 2026-08-02.** This is the workflow **as planned** before the campaign. Two things came out
> differently in execution and are flagged inline below: the consolidated CSV was delivered without
> the `treatment` column, and the consumer of the closure fractions is `stage3/paired_new.py`. What
> was actually produced is `data/whst_areas_final.csv` (223 rows) and `data/whst_pass1_qc.csv`, both deposited.

---

## 1. The obstacle and the solution

WHST opens an **interactive dialog** (variance radius, threshold, saturated pixels, diagonal). That
makes full macro automation awkward, because a macro cannot easily confirm the dialog. Recommended
solution: use the **Wound Healing Size Stacks Tool**, built to process multiple images at once with a
single dialog.

## 2. Recommended workflow (2 passes)

### Pass 1 — automated measurement per well (Stacks Tool)

For each well, group its images (all timepoints) and measure them in one go:

1. **Organise the images by well.** Ideally one subfolder per well containing that well's timepoints:
   ```
   HUVEC_A2/  →  A2_0hr.tiff, A2_8hr.tiff, A2_12hr.tiff, A2_24hr.tiff
   SKOV3_34/  →  Snap-34-0h.tiff, Snap-34-24h.tiff, Snap-34-48h.tiff, Snap-34-72h.tiff
   ```
2. **File → Import → Image Sequence**: load the well's subfolder as a stack (in timepoint order).
3. **Remove the scale from the stack:** Analyze → Set Scale → Click to Remove Scale (or Image →
   Properties → unit = pixel, pixel width/height = 1). This applies to the whole stack.
4. **Plugins → Wound healing size stacks tool**: run with the frozen parameters (radius = 20,
   threshold = 100, saturated = 0.001, diagonal = Yes).
5. The plugin measures every slice (timepoint) of the stack at once → a Results table with one row per
   slice.
6. **Export the Results table** (File → Save As, from the Results window) to CSV.

### Review — flag suspects

After pass 1, run the sanity check (in Python or by hand):

- **Non-monotonic closure**: if the area at one timepoint exceeds that of the previous timepoint (the
  wound "grew"), flag it.
- **Area above baseline**: negative closure, flag it.
- **Anomalous width standard deviation**: well above the median of that well's series, flag it.

Only the flagged images need visual review — not all of them.

### Pass 2 — manual correction of the flagged images

For each flagged image:

1. Open the individual image.
2. Remove the scale.
3. Plugins → Wound healing size **manual** tool.
4. Trace the correct contour (Polygon/Freehand), following the border criterion (§9.2 of the
   protocol).
5. Measure (Analyze → Measure).
6. Replace the value in the CSV and mark `method = manual`.

## 3. Structure of the consolidated CSV

Planned final file: `whst_measurements.csv`. One row per image.

| Column | Type | Description | Example |
| --- | --- | --- | --- |
| `filename` | text | Original file name | `A2_24hr.tiff` |
| `cell_line` | text | HUVEC or SKOV3 | `HUVEC` |
| `well_id` | text | Well with a cell-line prefix (avoids collision) | `HUVEC_A2` |
| `timepoint_h` | integer | Hours post-scratch | `24` |
| `treatment` | text | ~~Clinical group / treatment~~ — **not delivered**, see note below | — |
| `area_px` | float | Wound area in pixels² | `145089` |
| `area_pct` | float | Area as % of the whole image (cross-check) | `2.878` |
| `width_px` | float | Mean width (not used in the metric; recorded) | `78.028` |
| `width_sd` | float | Width standard deviation (instability flag) | `25.054` |
| `method` | text | `auto` or `manual` | `manual` |
| `flag` | text | Reason for flagging, if any | `non_monotonic` |
| `notes` | text | Operator observations | `wound nearly closed` |

> **The `treatment` column was not delivered, deliberately.** The plan was for it to carry the
> clinical group of the plasma donor. That analysis was dropped from this study — the test set does
> not support per-treatment metrics (effective n = 1 in most arms), so reporting them would have been
> a false precision. The delivered files (`data/whst_areas_final.csv`, `data/whst_pass1_qc.csv`) therefore carry
> provenance and acquisition keys but no clinical-group column. The clinical provenance of the HUVEC
> images is stated in the manuscript's Methods; it is simply not an analysis variable here.

### Closure-fraction computation (post-CSV)

For each well, the baseline is `timepoint_h = 0`. For each timepoint t:

```
closure_fraction(t) = (area_px[t0] − area_px[t]) / area_px[t0]
```

`area_pct` may be used instead of `area_px` (it is a ratio, so the result is identical) — useful if
some measurement escaped in a different unit.

## 4. Worked example (real measured data)

```csv
filename,cell_line,well_id,timepoint_h,area_px,area_pct,width_px,width_sd,method,flag,notes
A2_0hr.tiff,HUVEC,HUVEC_A2,0,1310043,25.986,635.786,24.015,auto,,baseline
A2_8hr.tiff,HUVEC,HUVEC_A2,8,,15.942,,,auto,,
A2_12hr.tiff,HUVEC,HUVEC_A2,12,743764,14.753,385.295,64.609,auto,,
A2_24hr.tiff,HUVEC,HUVEC_A2,24,145089,2.878,78.028,25.054,manual,auto_overestimated,corrected: automated included monolayer
Snap-34-0h.tiff,SKOV3,SKOV3_34,0,2901456,57.554,1411.785,404.514,auto,,baseline
Snap-34-24h.tiff,SKOV3,SKOV3_34,24,1330256,26.387,693.807,315.770,auto,,
Snap-34-48h.tiff,SKOV3,SKOV3_34,48,944906,18.743,439.912,194.024,auto,,
```

> Note: where the area in px was not recorded (the measurement came out in inches), `area_pct` covers
> it — closure fraction can be computed from that. Ideally, re-measure in px for full consistency.

## 5. Transparency metrics to report (Methods)

From the consolidated CSV, extract and report:

- **Total images measured** per cell line.
- **Manual-correction rate** = number of `method=manual` / total. Report separately by cell line and
  by timepoint (a higher rate is expected in SKOV-3 and at late timepoints).
- **Number of non-monotonic cases** and how they were handled (corrected vs retained as a real
  phenomenon).

## 6. Workflow checklist

- [ ] Images organised by well (subfolders)
- [ ] Stacks Tool tested on one well (verify it measures the whole stack)
- [ ] Pass 1: measure all wells (auto)
- [ ] Sanity check (monotonicity, negatives, standard deviation)
- [ ] Pass 2: correct the flagged images (manual)
- [ ] Consolidated CSV filled in
- [ ] Manual-correction rate computed
- [ ] Blinding maintained (without seeing the AI output)
