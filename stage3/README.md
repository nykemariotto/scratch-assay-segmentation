# Stage 3 — evaluation, uncertainty and tests

Everything that turns 25 trained runs into Table 2 and into the agreement analysis: mean
Average Precision with grouped confidence intervals, the seed-paired padding ablation, and
the comparison of the predicted closure fraction against the supervised reference standard.

Written and validated **before** the training grid finished, against synthetic records
built with the real structure of the test set. Once the runs existed it was two commands.

## Architecture: why two phases

Ultralytics' `val()` returns mAP aggregated over the whole test set. A confidence interval
that resamples the **37 acquisition groups** needs the AP recomputed inside every resample,
and the aggregate number alone cannot give that.

| Phase | Where | What it does | Output |
| --- | --- | --- | --- |
| 1 · `stage3/eval_test.py` | **GPU**, once per run | predicts, rasterises the COCO ground truth, matches by IoU at 10 thresholds | `registros/<run>.json`, a few kB |
| 2 · `stage3/aggregate.py` | **CPU**, freely | Table 2, confidence intervals, tests | `stage3/table2.csv` plus a report |

The records store, per image: the score of each detection, whether it is a true positive at
each IoU threshold, the number of ground-truth instances, and the group. That is all AP
needs, and it reduces the bootstrap to resampling the keys of a dictionary.

## Files

**Segmentation track** — mAP, confidence intervals, padding ablation, multiple seeds:

| File | What it is |
| --- | --- |
| `stage3/ap_core.py` | mask AP in the COCO convention, matching, cluster bootstrap (simple and paired) |
| `stage3/test_ap_core.py` | 9 validation cases — **no GPU** |
| `stage3/eval_test.py` | phase 1 (GPU), with a guard against running while the grid is active |
| `stage3/aggregate.py` | phase 2 (CPU) |
| `stage3/make_fixtures.py` | synthetic records with the real structure, to validate phase 2 |

**Area track** — agreement on the clinical endpoint, Supplementary Table S2:

| File | What it is |
| --- | --- |
| `stage3/predict_areas.py` | phase 1b (GPU): wound area per test-set image |
| `stage3/paired_new.py` | the rebuilt paired analysis and Supplementary Table S2 |
| `stage3/concordancia_final.py` | every agreement statistic the manuscript reports, persisted to JSON |
| `stage3/figuras_concordancia.py` | Figures 3–5, from that data |
| `stage3/make_area_fixture.py` | synthetic areas to validate the **join**, without a GPU |

### Why two tracks

mAP and area measure different things, and keeping them apart is deliberate: a model can
have high mAP and a systematically biased area, because mAP asks whether the contour lands
in the right place with sufficient IoU, not whether the area matches. Area is the quantity
the assay actually reports — the closure fraction comes from it.

One detail that matters in `stage3/predict_areas.py`: when an image has more than one instance,
the masks are **united**, not summed. Summing would double-count the overlap, and the
Wound Healing Size Tool — which measures the wound as a region — does not do that.

## The paired analysis was rebuilt, not updated

`paired_imagej_vs_ai_clean.csv` (225 observations) has an "AI" side from the model trained
**with leakage** and an "ImageJ" side from three annotators under a protocol that Stage 4
replaced. Editing numbers in that file would have been a patch over a broken thing.

| | old | new |
| --- | --- | --- |
| AI side | model trained with leakage | retrained leakage-free, predicted on the held-out set |
| reference side | three annotators | supervised reference standard (Stage 4) |
| cell lines | HUVEC only | **HUVEC and SKOV-3** |
| independence | — | the models never saw the reference standard |

The classical arm does **not** have that independence — see the incorporation bias
documented in `benchmark_classico.py`, at the repository root.

### Pair accounting

```
reference standard          223 images measured
  with a test-set image     218      (5 outside the held-out partition)
data/closure_final_longo.csv     187 observations across 59 series
  analysable                163
  less the baselines (t=0)   59      a baseline has no pair by construction
a single run              97–100 pairs   (three of the five YOLO runs return 97)
common to all ten runs       97 pairs · 45 series · HUVEC 82 · SKOV-3 15
```

That last line is the set every statistic uses. A run returns fewer pairs when it fails to
predict a wound area at **t = 0** for some series: without a baseline the closure fraction
(a₀ − aₜ)/a₀ is undefined, so the whole series is lost, every timepoint at once. That is
the entire difference here — three YOLO runs lose one three-timepoint series that the other
seven runs keep.

Restricting to the observations **all ten runs** return means both arms are measured on an
identical set, so the difference between them cannot be an artefact of composition.

97 is fewer than the 225 of the earlier analysis, and better: leakage-free, both cell
lines, against a supervised standard. The manuscript states that explicitly, because
otherwise the drop in n reads as lost data.

## How to run

Before the grid exists — validates everything, no GPU:

```bash
python stage3/test_ap_core.py
python stage3/make_fixtures.py
python stage3/aggregate.py --dir stage3/registros_fixture --out stage3/table2_fixture.csv --B 400 --permitir-fixture
```

`--permitir-fixture` is required and deliberate: `stage3/aggregate.py` aborts if the directory holds
only synthetic records, so a fixture number can never be mistaken for a result.

Once the runs exist:

```bash
python stage3/eval_test.py --all
python stage3/aggregate.py --B 5000
python stage3/concordancia_final.py
python stage3/figuras_concordancia.py
```

## The four outputs of `stage3/aggregate.py`

**1 · Table 2** — one row per *configuration*, mean ± SD over the 5 seeds. Reporting a
single run as "the performance of the model" is what a single-run design gets criticised
for, and rightly.

**2 · Grouped bootstrap confidence intervals**, with the naive interval alongside and the
ratio of the two widths. Resampling images treats frames of the same field as independent,
and they are not.

**3 · Padding ablation** — paired **by seed**. The same seed shares every source of
training stochasticity except the padding colour; comparing unpaired means would throw away
precisely the information the single-variable design bought.

**4 · Distinguishability** — 95% confidence interval of the **difference** between each
pair, by paired grouped bootstrap.

> Overlapping intervals are **not** a test of difference. The criterion is too conservative
> and calls "indistinguishable" things that are distinguishable. Both models are evaluated
> on the same images: resampling the same groups for both and taking the difference inside
> each resample cancels the variance of the test set.
>
> On the fixtures — which were built with large differences between configurations — the
> overlap criterion finds **3 of 10** pairs distinguishable and the paired test finds
> **9 of 10**. The difference is not cosmetic.
>
> On the **real** records the two criteria agree: **0 of 10** by either. That agreement is
> the result, not a validation of the criterion — with differences this small, the
> conservative test and the correct one reach the same conclusion.

### Sections 3 and 4 are not redundant

| | Tests against | Paired by |
| --- | --- | --- |
| section 3 | training noise | seed |
| section 4 | sampling variation of the test set | acquisition group |

A difference is only solid if it survives both, and the manuscript reports both.

## Validation already done, without a GPU

`stage3/test_ap_core.py`, 9 cases, all passing. The ones that caught something:

- **AP is rank-based.** A false positive scored *below* every true positive does **not**
  reduce AP; only one ranked above hurts. Counterintuitive, and it has to be understood
  before Table 2 is interpreted — pointwise precision falls in both cases, mAP does not.
- **An image with no ground truth** gives undefined AP (NaN), not zero. Zeroing it would
  drag the mean down for no reason, and there are 29 negatives in the test set.
- **Cluster versus naive**, on a synthetic structure built to be strongly correlated within
  groups (40 groups of 6): the grouped interval is **2.45×** wider. On the **real** records
  the ratio is **1.06 to 1.12** (median 1.10) across the five configurations — the real
  acquisition groups are far less internally correlated than that stress case. The
  direction is what the test asserts; the magnitude is a property of the data.

`stage3/aggregate.py` was validated end to end over 25 synthetic records generated with the
**real** structure of the test set: 234 images, 229 ground-truth instances, 37 groups of
size 2 to 22. Testing with equal-sized groups would not have exercised the real case.

> The fixture values are fictitious and exist only to validate the mechanics.
> `stage3/make_fixtures.py` marks every file with `"FIXTURE": true`.

### The area track is validated through the join

The risk in `stage3/paired_new.py` is not the statistics — it is the **key matching**: three files
with different naming conventions (the export name in the test set, the `whst_input` name
with an md5 prefix, and `series_key` with pipe separators). If the join is wrong the script
does not crash; it returns fewer pairs, and nobody notices.

`stage3/make_area_fixture.py --ruido 0 --vies 0` propagates the **real** reference areas backwards
through the join. The result has to be exact:

```
Pearson +1.0000 · Spearman +1.0000 · CCC +1.0000 · bias +0.0000
join against the reference: 218 direct, 0 by md5 prefix, 0 unmatched
```

Any departure from that would be a matching error, not a model error. It passed. (The
script's own console output is still in Portuguese; the numbers are the ones above.)

**One result of that test is worth recording.** Injecting 12% *constant* bias into the area
left the bias in the closure fraction at **+0.0012**, essentially zero. That is not a bug:
the ratio (a₀ − aₜ)/a₀ is invariant to a constant multiplicative factor, and the pipeline
reproduces that correctly.

Only bias that **varies over time** survives — and that is what actually happens. With
`--vies-por-tp 0.35`:

```
Pearson +0.9821   but   Lin's CCC +0.9312   ·   bias −0.0894
```

This is the demonstration of why the agreement analysis needs **CCC rather than
correlation**: Pearson barely moves, because the model still tracks the trend, while the
CCC collapses, because the level is wrong. Reporting only the correlation would hide
exactly the defect a critical reading looks for.
