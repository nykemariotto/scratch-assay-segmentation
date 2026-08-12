# Reference standard — supervised wound measurements

Manuscript 4336348 · *Deep Learning Instance Segmentation for Quantitative Analysis
of Cell Migration in Wound Healing Assays* · Cytometry Part A

This package holds the supervised measurements that serve as the **reference standard**
for the paired analysis, the automatic measurements they correct, and the records of how
the correction was carried out. Every count below was read from the files in this
package, not from the manuscript.

---

## A note on the directory names

Five directories carry Portuguese names. They are kept unchanged, deliberately: the
ImageJ macro `stage4/whst_manual_correction.ijm` wrote to these exact paths during the
annotation campaign, and the analysis code deposited in `code.zip` (`stage4/*.py`) reads
them by name. Renaming them here would make the deposited data stop matching the
deposited code, and the deposited code stop matching the code that produced the
published numbers. The 704 distinct file names are in English, with the single exception
noted in the table.

| directory | reads as |
|---|---|
| `rois_corrigidos/` | corrected ROIs — **the reference standard** |
| `rois_recorrecao/` | re-corrected ROIs (blinded repeat) |
| `rois_validacao/` | provenance-validation ROIs |
| `rois_completacao/` | completion ROIs |
| `_removidas_fora_do_escopo/` | removed, out of scope (quarantine) |
| `MANIFESTO.csv` | manifest |

---

## Automatic measurement — the input the correction acts on

Produced by the Wound Healing Size Tool under the parameters frozen before the campaign
(see *Protocols* below). 223 frames, all with status `ok`.

| path | n | contents |
|---|---:|---|
| `whst_batch_results.csv` | 223 rows | area in px and as % of field, wound width, ROI count, unit and pixel-width checks |
| `rois/` | 223 | ImageJ `.roi`, one per frame |
| `masks/` | 223 | binary mask, `<base>_mask.png` |
| `polygons/` | 223 | vertex list, `<base>_polygon.csv` |
| `overlays_sorted_map.csv` | 223 rows | frame → cell line, group, timepoint, field, source file and its MD5 |

## Manual correction — five passes

Each pass wrote to its own directory; nothing was ever overwritten.

| directory | pass | frames handed to the operator | what it settles |
|---|---|---:|---|
| `rois_corrigidos/` | 1 — correction | 102 | the reference standard for the paired analysis |
| `rois_recorrecao/` | 2 — blinded re-correction | 15 | intra-observer reproducibility |
| `rois_validacao/` | 3 — provenance validation | 10 | whether frames judged OK in triage differ systematically from corrected ones |
| `rois_completacao/` | 4 — completion of mixed series | 24 | removes mixed automatic/corrected provenance by construction |
| `rois_baselines/` | 5 — recovered baselines | 5 | t₀ frames recovered from the raw archive |

Each directory holds the `.roi` files and a `masks/` subdirectory with the corresponding
binary masks. The per-frame record of each pass — what the operator was shown and what
they recorded — is deposited in `code.zip` as `stage4/correcao_manual_pass1.csv`,
`…_pass2.csv`, `…_validacao.csv`, `…_completacao.csv` and `…_baselines.csv`, alongside
the worklists that were handed to them.

---

## Why the `.roi` and mask counts differ

**This is not missing data.** The macro records four possible outcomes per frame, and
two of them produce no contour:

| outcome | `.roi` | mask | meaning |
|---|---|---|---|
| `ok` | yes | traced region | contour corrected; area measured |
| `fechada` | **no** | **all-zero** | wound fully closed — a **valid** measurement of area 0, i.e. closure = 1.0 |
| `pulada` | no | no | not corrected in this pass; returns in the next run |
| `invalida` | no | no | frame discarded, cannot be segmented |

A closed wound has no region to outline, so it yields a mask and no ROI. The counts
follow exactly:

| directory | `.roi` | masks | = `ok` + `fechada` | other outcomes |
|---|---:|---:|---|---|
| `rois_corrigidos/` | 79 | 96 | 79 + 17 | 1 `pulada`, 5 `invalida` |
| `rois_recorrecao/` | 11 | 14 | 11 + 3 | 1 `invalida` |
| `rois_validacao/` | 10 | 10 | 10 + 0 | — |
| `rois_completacao/` | 24 | 24 | 24 + 0 | — |
| `rois_baselines/` | 2 | 2 | 2 + 0 | 3 `invalida` (of 5 candidates) |

All 17 and all 3 zero-area masks were verified to contain no lit pixel. Treating them as
absent would silently drop the completely-closed wounds from the analysis — which are
precisely the late timepoints.

---

## Quarantine — `_removidas_fora_do_escopo/`

Five images the operator identified as **a test of the algorithm rather than experimental
data**. They were moved here rather than deleted, with `MANIFESTO.csv` recording the move.
They were verified absent from the partition definition, so they never belonged to the
train/validation/test splits and their removal does not affect the leakage-free split.
Each is kept with its full artefact set (source `.jpg`, `.tiff`, `.roi`, mask, polygon,
overlay).

---

## Agreement tables (`data/`)

| file | rows | contents |
|---|---:|---|
| `correction_agreement.csv` | 96 | per frame: IoU automatic vs pass 1, areas, signed delta, and IoU pass 1 vs pass 2 where the blinded repeat exists |
| `whst_series_analysis.csv` | 69 | per series: cell line, batch, frame count, baseline presence, closure plausibility, decision, invalid-frame count, and whether the baseline was borrowed |

---

## File naming

```
<md5-10>__<cell line>__<acquisition unit>__<timepoint>__<original file name>
```

`<md5-10>` is the first ten characters of the MD5 of the source file; it matches
`raw_md5` in `overlays_sorted_map.csv` for all 223 frames, and is what links a
measurement back to the exact acquisition it came from. Some recovered-baseline and
quarantined files carry an extra provenance tag (`BASELINE`, `BASELINE_ALT`) after the
hash. Masks append `_mask.png`, polygons `_polygon.csv`.

---

## Protocols

- `protocols/WHST_protocol_frozen.md` — the measurement parameters, frozen in writing
  before any measurement was taken. Portuguese; this is the record.
- `protocols/WHST_protocol_frozen_EN.md` — faithful English translation of the above.
- `protocols/whst_measurement_workflow.md` — the operating sequence.

The manual-correction protocol governing the five passes is documented in the repository
as `PROTOCOLO_CORRECAO_MANUAL.md`; its operative content — the a-priori edge criterion,
the four outcomes, the pass structure and the counts — is reproduced above.

---

## Not included

Per-frame overlay JPEGs (`overlays/`, `overlays_sorted/`) are omitted for size; they are
rendered views of the masks in this package, not independent data.
`overlays_sorted_map.csv` therefore references files that are not deposited — the column
`sorted_basename` names them, and the mask and polygon for every one of those frames is
present here. The overlays of the five quarantined images are kept, as part of their
artefact set.

---

## Licence

CC BY 4.0. See `NOTICE` in the deposit for how the three licences of this work divide.
