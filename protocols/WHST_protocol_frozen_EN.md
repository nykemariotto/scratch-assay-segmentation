# Frozen WHST protocol — reference standard
## Gate 4 of the experimental protocol · Manuscript 4336348

> **English translation.** The document frozen before the measurement campaign was written in
> Portuguese and is deposited unchanged as `protocols/WHST_protocol_frozen.md`. This is a faithful
> translation of it, provided because the deposit is read in English. Every parameter, count and
> statistic is identical to the original; where the original carries a dated correction note, the
> note is translated with it. If the two ever disagree, the Portuguese original is the record.

> **Purpose.** Freeze all measurement parameters of the Wound Healing Size Tool (WHST) BEFORE any
> measurement, so that the reference standard is fully documented and reproducible. This settles the
> citation, the angle correction and the parameters definitively, and forecloses second-round
> questions.
>
> **Golden rule.** Once frozen, the parameters do not change during the measurement campaign. Any
> deviation is documented and justified in writing.

---

## 1. Tool and citation

| Item | Value |
| --- | --- |
| Plugin | Wound Healing Size Tool (WHST) |
| Version | *updated* (with the manual-selection button for unidentified areas = "Manual Tool") |
| Platform | Fiji (ImageJ) 64-bit |
| Citation | Suarez-Arnedo A, Torres Figueroa F, Clavijo C, Arbeláez P, Cruz JC, Muñoz-Camargo C. An ImageJ plugin for the high throughput image analysis of in vitro scratch wound healing assays. PLoS ONE. 2020;15(7):e0232565. doi:10.1371/journal.pone.0232565 |
| Plugin repository | github.com/AlejandraArnedo/Wound-healing-size-tool |

## 2. Downloads

| Item | URL |
| --- | --- |
| Fiji | https://fiji.sc/ |
| WHST updated (recommended) | https://github.com/AlejandraArnedo/Wound-healing-size-tool/blob/6210027c8dec7a346bd0a39da37c8d8facf7270d/Wound_healing_size_tool_updated.zip?raw=true |
| Video tutorial | https://www.youtube.com/watch?v=OgzyJJi-0Ik |
| Manual (S2 File of the paper) | Supplementary material of the PLoS ONE publication |

## 3. How the algorithm works (context for the documentation)

WHST uses classical computer vision based on the intensity variance of neighbouring pixels:

1. **Enhance Contrast** — increases the variance inside the cell monolayer.
2. **Variance filter** — computes intensity variance in the neighbourhood of each pixel (radius =
   variance window radius). Monolayer → high variance; open wound → low variance.
3. **Threshold** — binarises the variance image (value set by the user).
4. **Hole filling** (morphological reconstruction by erosion) — includes isolated cells or islands
   inside the wound as part of the wound.
5. **Largest connected component** — selects the largest region as the true wound (eliminates false
   positives).

> **Critical note.** Step 4 (hole filling) makes WHST **include** isolated cells inside the gap as
> part of the wound — that is, WHST **also** does not subtract cells migrating into the gap. This
> matters: our AI (single contour) and the reference standard (WHST) behave the same way in this
> respect, which is what makes the comparison fair. Documenting it in the Discussion strengthens the
> response on isolated cells.

## 4. The 3 adjustable parameters (TO BE FROZEN)

The plugin exposes exactly three parameters. Ranges validated by the authors (from the paper, S5 Fig):

| Parameter | Function | Validated range | **HUVEC (frozen)** | **SKOV-3 (frozen)** |
| --- | --- | --- | --- | --- |
| **Variance window radius** | Radius of the variance filter. Too low → fails to recognise the scratch; too high → underestimates the area | 3 – 25 | **20** | **20** |
| **Threshold** | Binarisation value. Increasing it increases the detected area | 50 – 150 | **100** | **100** |
| **% saturated pixels** | Contrast. Increasing it detects smaller areas. Must be > 0 | 0.001 – 0.4 | **0.001** | **0.001** |
| **Scratch is diagonal** | Enables angle correction (affects width only, not area) | Yes/No | **Yes** (does not affect area) | **Yes** (does not affect area) |
| **Scale** | Image calibration | — | **Removed → measurement in pixels²** | **Removed → measurement in pixels²** |

> **Unified HUVEC + SKOV-3 parameters, frozen.** HUVEC validated on 4 images of well A2
> (0/8/12/24 h); SKOV-3 validated on well 34 (Snap-34, 0/24/48 h) with the **same parameters** — a
> monotonic and plausible series (closure 0 → 0.542 → 0.674). Using a single parameter set for both
> cell lines reduces degrees of freedom and is more defensible (less suspicion of tuning).
>
> **Reliability caveat.** With the parameters fixed, the quality of automated segmentation **varies
> by well and by timepoint**. "Easy" wells (such as 34) segment well even at 48 h; "difficult" wells
> already fail from 24 h onward. The critical variable is the intrinsic difficulty of the image
> (monolayer density, contrast, how far the wound has closed), not the parameter. Consequence: a
> **substantial manual-correction rate is expected in SKOV-3**, higher than in HUVEC. Report the rate
> in the Methods.

## 5. Angle correction

WHST offers an **inclination correction**: if the scratch is diagonal, the plugin adjusts the width
measurement by the inclination angle (fitting the ROI to an ellipse, a trigonometric correction).

**Frozen decision and rationale:**

- Angle correction affects the measurement of **width**, NOT of **area**.
- Because our reference standard uses **wound area → closure fraction** (not width), angle
  correction **does not affect** the metric we use.
- **Decision:** state explicitly that the extracted metric was wound area, and that the WHST angle
  correction (which operates on width) does not affect an area-based comparison. This is exact: the
  concern about angle correction applies to width; our comparison is by area.

> **CONFIRMED:** the extracted metric is **wound area → closure fraction** (not width). Rationale:
> (1) the AI model produces area (px²), so an apples-to-apples comparison requires area from the
> reference standard; (2) closure fraction is the metric the manuscript and the whole agreement
> analysis already use; (3) WHST angle correction operates on width and therefore does not affect
> area — which closes the angle-correction question definitively.

## 6. Parameter calibration (before the campaign)

Following the authors' own recommendation (test 2–3 images before running everything):

1. Select **3 random images** from each cell line (HUVEC and SKOV-3) at varied timepoints.
2. Test parameter combinations within the validated ranges.
3. Choose the set that best segments the wound by visual inspection, **separately per cell line** if
   necessary (HUVEC and SKOV-3 may require different parameters because of density/contrast).
4. **Freeze** the chosen values and record them in a `whst_params.txt` file.
5. From then on, **do not change them** during the campaign.

> **Rationale for separate calibration:** SKOV-3 is a denser monolayer and may require different
> parameters from HUVEC. Documenting both sets is honest and defensible. What may not be done is
> adjusting parameters image by image — that would be circular tuning.

## 7. Measurement scope (decision A5 already taken: HUVEC + SKOV-3)

| Item | Specification |
| --- | --- |
| Cell lines | **Both** — HUVEC and SKOV-3 |
| Images to measure | The pairs needed for method agreement: baseline (t0) + each post-scratch timepoint, per well |
| HUVEC timepoints | t0, 8 h, 12 h, 24 h |
| SKOV-3 timepoints | t0, 24 h, 48 h, 72 h |
| Extracted metric | Wound area (pixels²) → closure fraction |

### 7.1 File-naming convention (for the Stage 1 parsing)

| Cell line | Pattern | Example | Well encoded in |
| --- | --- | --- | --- |
| HUVEC | `{well}_{timepoint}hr.tiff` | `A2_24hr.tiff` | letter+number before the `_` (e.g. `A2`) |
| SKOV-3 | `Snap-{well}-{timepoint}h.tiff` | `Snap-34-24h.tiff` | number between `Snap-` and the timepoint (e.g. `34`) |

The well is recoverable from the file names in both lines (no external spreadsheet needed). The
Stage 1 parsing regex must handle both patterns. To group by well: HUVEC uses the alphanumeric token
(A2, B3…); SKOV-3 uses the number (34, 35…). **Note:** ensure that well identifiers from the two
lines do not collide in the split (prefix with the cell line, e.g. `HUVEC_A2`, `SKOV3_34`).

## 8. Closure-fraction computation (from the WHST areas)

For each well, at each post-scratch timepoint:

```
closure_fraction(t) = (area_t0 − area_t) / area_t0
```

Where:

- `area_t0` = wound area at baseline (t = 0) for that well
- `area_t` = wound area at timepoint t for the same well

Interpretation: closure_fraction = 0 → no closure; = 1 → complete closure; < 0 → the wound "grew"
(measurement noise or retraction — the negative cases).

> This reproduces exactly the definition used in the paired AI vs ImageJ analysis. The script that
> consumes these closure fractions is `stage3/paired_new.py`.
>
> **Pointer corrected, 2026-08-02.** This line named `paired_analysis.py` as the consumer. That
> script was **retracted**: it trained and evaluated without a partition filter, and the analysis it
> produced does not appear in the manuscript. **The protocol does not change** — the definition of
> closure fraction above is the same. What is corrected is where the reader is sent.

## 9. Human-supervision and manual-correction protocol (frozen rule)

### 9.1 Per-image decision flow

| Situation | Action |
| --- | --- |
| Automated segmentation is correct (typical of 0/8/12 h — wide, well-defined wound) | **Use the automated value** |
| Wound nearly closed / diffuse border (typical of 24/48 h) | **Check visually**; if the automated contour includes monolayer (overestimates) or omits wound, **correct manually** |
| Width standard deviation abnormally high versus the series of the same well | **Warning sign** → check visually |

### 9.2 Objective criterion for the "wound border"

Operational definition (apply consistently): the wound border is **where the confluent monolayer
ends** and the open or sparse region begins. When correcting manually, trace the contour at that
transition, using earlier timepoints of the same well as a reference for the wound geometry.

### 9.3 Manual-correction tool

- **Plugins → Wound healing size manual tool**.
- Remove the scale first (Analyze → Set Scale → Remove Scale) to measure in pixels².
- Trace the contour with a Polygon or Freehand selection, following the real wound.
- Measure (via the Manual Tool or Analyze → Measure).

### 9.4 Mandatory record

- Mark `corrected_manually = yes/no` in the CSV for every image.
- Report the **manual-correction rate** (% of images corrected) in the Methods — an important number
  for transparency.

### 9.5 Documented case: well A2 (HUVEC) — reference example

Time series of well A2, illustrating the decision flow:

| Timepoint | Automated (area %) | Corrected? | Final area % | Closure fraction |
| --- | --- | --- | --- | --- |
| 0 h | 25.986% | No | 25.986% | 0 (baseline) |
| 8 h | 15.942% | No | 15.942% | 0.386 |
| 12 h | 14.753% | No | 14.753% | 0.432 |
| 24 h | 21.189% (overestimated) | **Yes** | 2.878% | 0.889 |

The automated result at 24 h overestimated (it included sparse monolayer as wound; the width standard
deviation jumped to 4.241 versus 0.25–0.67 in the others — a warning sign). Manual correction
recovered the real narrow gap. The corrected series is monotonic and biologically coherent
(progressive closure 0 → 38.6% → 43.2% → 88.9%).

**Lesson incorporated:** automated WHST tends to overestimate in nearly closed wounds with diffuse
borders (24/48 h). These require verification and frequent manual correction. This connects directly
with the discordant cases and is reported honestly.

## 10. Blinding (guard against bias)

- **The WHST operator must not see the AI model output while measuring.** This prevents the reference
  measurement from being unconsciously biased toward the AI.
- Measure the WHST areas **before** comparing with the AI, or with the AI output hidden.
- Document in the manuscript that the measurement was blinded → strengthens the validity of the
  reference standard.

## 11. Record and deliverables

| File | Content |
| --- | --- |
| `whst_params.txt` | Frozen parameters (radius, threshold, saturated pixels) per cell line |
| `whst_measurements.csv` | One row per (well × timepoint): cell_line, well_id, timepoint, area_px, closure_fraction, corrected_manually (yes/no) |
| `whst_protocol_notes.md` | Operating notes: who measured, when, manual-correction criterion, confirmation of blinding |

## 12. Methods text — final version with real numbers

> Replaces the earlier draft (written before execution). All values are those obtained in Stage 4.

### 12.1 Reference standard measurement (Methods)

Reference measurements of wound area were obtained with the Wound Healing Size Tool (WHST;
Suarez-Arnedo et al., PLoS ONE 2020;15(7):e0232565), an ImageJ/Fiji plugin that segments the open
wound via a variance-based algorithm followed by hole filling and selection of the largest connected
component. Measurements were performed on the raw acquisition files (2452 × 2056 px) rather than on
the resized copies used for model training, so that all images within a series shared an identical
field of view.

Plugin parameters were calibrated on images from both cell lines and then held fixed for the entire
campaign (variance window radius = 20; binarisation threshold = 100; saturated pixels = 0.001; all
within the ranges validated by the plugin authors). Image calibration was removed prior to
measurement so that all areas were recorded in pixels², matching the units of the model output; unit
consistency was verified per image. A single parameter set was sufficient for both HUVEC and SKOV-3,
avoiding cell-line-specific tuning. Because our outcome is wound **area** rather than average wound
width, the plugin's angle-correction function — which adjusts width for scratch inclination — does
not affect the reported values.

Because the plugin does not export segmentation geometry, the measurement macro was extended to save,
for every image, the region of interest, a binary mask, the contour polygon, and an overlay for
visual review. The macro is deposited with the analysis code.

### 12.2 Visual triage and manual correction

Automated segmentations were reviewed by a single observer under a rubric defined a priori: a frame
was scored **OK** when a wound was identifiable, the monolayer was confluent on both margins, and the
contour followed the wound; **segmentation failure** when the image was valid but the contour was
incorrect (excess, deficit, or displacement); and **invalid image** when no wound was present in the
field, the frame was severely out of focus, or the monolayer was not confluent. The wound border was
defined as the transition between confluent monolayer and open or sparsely populated area.

Review was performed **blind**: the panels presented only the image, its contour, and the file name,
with no automated quality flags, measured areas, or series statistics. Critically, the entire
measurement and correction campaign was completed **before any segmentation model was trained**, so
the observer could not be biased toward the model output — it did not yet exist.

Of 223 images, 77 (34.5%) were scored OK, 123 (55.2%) as segmentation failure on a valid image, and
23 (10.3%) as invalid images. The last figure constitutes the assay exclusion rate and is distinct
from the method failure rate: an invalid image is removed from analysis, whereas a segmentation
failure on a valid image is corrected. Among failures, excess segmentation predominated, but in a
substantial fraction the mask was displaced and did not overlie the wound at all, indicating
localisation error rather than imprecise delineation.

Frames scored as segmentation failures were re-segmented manually using the plugin's manual-selection
tool, starting from the automated contour. A total of 140 frames were corrected; 132 of 223 images
(59%) carry a manually verified measurement. Manual correction reduced the segmented area in 86% of
cases, with a median change of −22.5 percentage points of image area. 

> **Dated correction, 2026-08-04.** The 86% and the −22.5 pp above belong to different sets and the sentence joined them. The −22.5 pp median is from the subset of **96** frames for which both an automatic and a corrected contour exist (`data/correction_agreement.csv`), and over that set the reduction occurs in **85%** (82/96), not 86% — the 86 is not reproducible from any slice of the data. Over the **132 of the 140** corrected frames for which both areas are recorded, the median is **−11.3 pp** and **71%** are reductions, because the completion, baseline and validation passes include frames where the plugin returned no contour and correction could only add area. The data did not change: the CSV regenerates identical.

### 12.3 Observer reproducibility

To quantify the subjectivity of manual correction, 15 frames were randomly selected (fixed seed) and
re-corrected after an interval, with the observer blind to both the selection and the first
correction. Intra-observer agreement was **IoU 0.861** (median, 95% CI 0.746–0.904) and **Lin's
concordance correlation coefficient 0.996** for measured areas, with a bias of +0.10 percentage
points; the categorical outcome agreed in 14 of 15 frames.

> **Correction, 2026-07-31.** The values first reported here (IoU 0.894, CCC 0.998) were computed
> over all 14 repeat pairs. Four of those pairs do not measure boundary reproducibility: in three the
> observer recorded no wound on either pass, which scores an intersection over union of 1 by the
> empty-mask convention, and in one the two passes returned an identical area and an exact
> intersection over union of 1. **The protocol itself is unchanged** — the frame selection, the fixed
> seed, the blinding and the interval are as frozen. What is corrected is a result computed after
> freezing.
>
> **Extended, 2026-08-02.** The ratio in the paragraph below was derived from the superseded IoU and
> was left at 3.3 when the IoU was corrected. With the clean-pair median, 0.8605 / 0.2665 = 3.23, so
> the figure is **3.2**, and it now reads that way. Same correction, one step further down the chain.
> Verifiable from `data/correction_agreement.csv` and `stage3/intraobs_ci.json`, both deposited.

For comparison, agreement between the automated and the manually corrected segmentation was **IoU
0.267**. The observer therefore agreed with themselves approximately 3.2 times more closely than the
automated method agreed with the observer, indicating that the dominant source of variability is the
segmentation method rather than the human operator.

### 12.4 Closure fraction and analysable series

For each acquisition field, closure fraction at time *t* was computed as (area₀ − areaₜ)/area₀, where
area₀ is the baseline (0 h) measurement of the same field. Series lacking a baseline cannot yield a
closure fraction and were excluded.

Of 68 candidate series, **52 (41 HUVEC, 11 SKOV-3) were analysable**. Nine were excluded for absence
of a usable baseline and seven retained implausible closure trajectories after correction.

> **Correction, 2026-08-01.** The counts in the sentence above are wrong and the sentence contradicts
> itself. Of 69 candidate series (not 68), nine lacked a usable baseline and one contained no valid
> frame, leaving **59 analysable (48 HUVEC, 11 SKOV-3)** — not 52. The 7 series with implausible
> trajectories were *retained*, as the sentence itself states, so they belong inside the analysable
> count; 52 is 59 minus those 7, subtracted in error. The 187 measurements are the total over all 59
> series, which is why that figure was and remains correct. Verifiable from
> `data/whst_series_analysis.csv` and `data/closure_final_long.csv`, both deposited. **The protocol itself is
> unchanged** — the calibration, the border criterion, the blinding and the correction rule are as
> frozen. What is corrected is a count computed after freezing. Median closure at the final timepoint
> was 0.691 (IQR 0.450–0.930); 13 series reached complete closure (≥ 0.99). The final dataset
> comprises 187 paired measurements.

Correcting the segmentations markedly improved the biological plausibility of the series: the number
of series with implausible closure trajectories fell from 15 to 4, and the category of series that
were uniformly over-segmented yet internally consistent disappeared entirely. This indicates that the
implausibility arose from segmentation rather than from the underlying biology. It also shows that
over-segmentation is not a constant multiplicative bias — which would cancel in a ratio — but grows
as the wound narrows, so that the error is largest at precisely the timepoints where closure is most
informative.

### 12.5 Statement of independence

The reference standard was established independently of the models under evaluation: measurements
were made on raw images with a published, previously validated tool, under frozen parameters, by a
blinded observer, before any model was trained. Model outputs were never used to inform reference
measurements at any stage.

## Checklist before opening Fiji

- [x] Fiji downloaded and installed
- [x] WHST *updated* installed (Plugins → Wound healing size tool appears)
- [x] Metric confirmed: area → closure fraction (not width)
- [x] HUVEC parameters calibrated and frozen (radius = 20, threshold = 100, saturated = 0.001)
- [x] SKOV-3 parameters confirmed (same as HUVEC — validated on well Snap-34)
- [x] Naming convention documented (HUVEC `A2_24hr`, SKOV-3 `Snap-34-24h`)
- [x] Scale removed → measurement in pixels²
- [x] Manual-correction rule defined (§9) + case A2 documented
- [ ] Blinding ensured (AI output hidden during measurement)
- [ ] Structure of `whst_measurements.csv` ready
- [ ] Bulk measurement workflow defined (manual vs macro vs stacks)
