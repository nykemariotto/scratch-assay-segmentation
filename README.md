# Scratch Assay Segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20298129.svg)](https://doi.org/10.5281/zenodo.20298129)
[![License: AGPL-3.0](https://img.shields.io/badge/Models-AGPL--3.0-orange.svg)](LICENSE-AGPL-3.0.txt)
[![License: MIT](https://img.shields.io/badge/Analysis%20code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY-4.0.txt)
[![Live demo on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/nmariotto/Scratch-assay-segmentation)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Companion repository for the manuscript:

> **Deep Learning Instance Segmentation for Quantitative Analysis of Cell Migration in Wound Healing Assays: A Benchmark and Web-Accessible Tool**
> Mariotto N, Dos Santos GA, Fioschi GR, Higino ACM, Passeti LFP, Pereira TOB, Zampieri GM, Aal MCE, Delella FK, Sandrim VC, Alves AFF.
> Submitted to *Cytometry Part A* (2026).

## Overview

This repository hosts the **runnable workspace** of a deep-learning benchmark for automated wound-gap segmentation in brightfield scratch (wound-healing) assays. It pairs with:

- A permanent **Zenodo archive** (DOI: [`10.5281/zenodo.20298129`](https://doi.org/10.5281/zenodo.20298129)) — the annotated image dataset (n = 1,363 at native acquisition resolution), the trained weights of all five evaluated configurations, COCO-format polygonal annotations, the supervised reference-standard measurements, and the analysis pipeline.
- A **public model repository** (https://huggingface.co/nmariotto/scratch-assay-segmentation) — the two deployed weights, `M.pt` and `S.pt`, downloadable without a token.
- A **live Hugging Face Space** (https://huggingface.co/spaces/nmariotto/Scratch-assay-segmentation) — inference on user-uploaded images, no installation required.

### Division of artifacts

| Where | What | Why |
|---|---|---|
| **GitHub (this repo)** | Code, small CSVs, docs, CI | Versioned workspace for continuing work |
| **Zenodo** | Dataset at native resolution, weights of the five configurations, frozen copy of the code | Citable archival snapshot (persistent DOI) |
| **Hugging Face** | Deployed weights and the live inference interface | Public access without installation |

The weights and the image dataset are **not duplicated here**; download instructions are in [models/README.md](models/README.md) and [dataset/README.md](dataset/README.md).

## Repository layout

One directory per pipeline stage. Each script writes its outputs beside itself,
so a stage directory holds both the code and the artefacts it produced.

```
scratch-assay-segmentation/
├── examples/                  ← START HERE: 4 annotated images (3 MB)
│   ├── images/  labels/       runnable without downloading the dataset
│   ├── MANIFEST.csv           provenance + MD5 of each example
│   └── data.yaml              ready for `yolo val`/`predict`
├── verify_install.py          checks the environment is ready
├── predict_example.py         runs the pipeline on examples/
├── requirements.txt           pinned dependencies (exact versions used)
│
├── stage1/                    dataset reconstruction: the leakage-free split,
│                              its audit, and the mapping tables
├── stage2/                    training: the 25-run grid and the padding patch
├── stage3/                    evaluation: mAP, cluster-bootstrap CI, ablation,
│                              agreement with the reference standard
├── stage4/                    reference standard: the WHST macros, the manual
│                              correction, and the intra-observer analysis
│
├── data/                      tables read by more than one stage
├── protocols/                 frozen WHST measurement protocol (EN + original)
├── figures/                   Figures 3-5, vector and raster, with their data
│
├── unet_comparator/           the U-Net comparator, with its regression tests
├── benchmark_classico.py      the classical comparator (WHST, automatic mode)
├── analysis/                  agreement analysis of an earlier version, superseded
├── webapp/                    source of the deployed Hugging Face Space
│
├── coco_partitions/           COCO annotations per partition (leakage-free)
├── whst_output/               ROIs and masks of the reference standard
├── paired_data/               the earlier paired measurements
├── dataset/                   full dataset (from Zenodo; git-ignored)
├── models/                    trained weights (from Zenodo; git-ignored)
├── PROTOCOLO_CORRECAO_MANUAL.md   border criterion and manual-correction rules
├── NOTICE                     which licence applies to which component
└── .github/workflows/         CI: validates statistical reproducibility
```

### Analysis scripts, by pipeline stage

Every script runs standalone **from the repository root**, not from inside its
stage directory — paths are resolved relative to the root.

| Stage | Entry points | What it does |
|---|---|---|
| **1 · Dataset reconstruction** | `stage1/build_final_split.py` · `stage1/build_final_split_strat.py` · `stage1/apply_fallback.py` · `stage1/export_coco_per_partition.py` · `stage1/coco_to_yolo_seg.py` | Groups images by physical acquisition field, builds the leakage-free split, exports COCO per partition and converts to YOLO-seg |
| **1 · Audit of the split** | `stage1/verify_final.py` · `stage1/check_coco_dims.py` · `stage1/check_640_population.py` · `stage1/leakage_md5_check.py` | Verifies that no acquisition field crosses partitions, label integrity (COCO dims vs file dims), and absence of duplicate images across splits |
| **2 · Training** | `stage2/run_grid.py` · `stage2/train_config.py` · `stage2/padding_patch.py` | 25-run single-variable ablation (model size · padding · initialisation) × 5 seeds |
| **3 · Evaluation** | `stage3/eval_test.py` → `stage3/aggregate.py` · `stage3/concordancia_final.py` · `stage3/figuras_concordancia.py` | mAP with cluster-bootstrap CI over acquisition groups, seed-paired padding ablation, agreement with the reference standard, Figures 3–5 |
| **3 · Comparators** | `benchmark_classico.py` · `unet_comparator/run_unet_grid.py` | The classical arm (WHST in pure automatic mode) and the deep-learning arm (U-Net under identical conditions) |
| **4 · Reference standard** | `stage4/whst_batch.ijm` → `stage4/whst_manual_correction.ijm` → `stage4/apply_corrections.py` → `stage4/final_closure_table.py` | Automated WHST measurement, blind visual triage, supervised manual correction, closure-fraction table |
| **4 · Validation** | `stage4/correction_agreement.py` · `stage4/intraobs_ci.py` · `stage4/validate_provenance.py` · `stage4/classify_failure_mode.py` | Intra-observer reproducibility (IoU, Lin's CCC), provenance test, failure-mode characterisation |

Stages 3 and 4 have their own documentation: [`stage3/README.md`](stage3/README.md)
for the evaluation pipeline, [`unet_comparator/README.md`](unet_comparator/README.md)
for the deep-learning comparator, and
[`PROTOCOLO_CORRECAO_MANUAL.md`](PROTOCOLO_CORRECAO_MANUAL.md) §7 for the
step-by-step of the reference standard.

Stage 4 ran **before** stage 2: the reference standard was completed before any
model was trained, which is what makes it independent of the models it evaluates.
The numbering follows the pipeline, not the calendar.

## Installation

Requires **Python 3.11+**. All dependencies are pinned in
[`requirements.txt`](requirements.txt) to the exact versions used to produce the
reported results.

```bash
git clone https://github.com/nykemariotto/scratch-assay-segmentation.git
cd scratch-assay-segmentation
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux/macOS:    source .venv/bin/activate
pip install -r requirements.txt
```

**GPU (recommended for training).** `requirements.txt` pins `torch==2.6.0`, but
PyPI serves the CPU build. For CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Inference and the analysis scripts run fine on CPU; only training needs a GPU.

**ImageJ/Fiji** is required only to reproduce the reference-standard measurements
(`stage4/whst_batch.ijm`, `stage4/whst_manual_correction.ijm`). Download from
[fiji.sc](https://fiji.sc/). Not needed for training, inference, or statistics.

Verify the installation:

```bash
python verify_install.py
```

## Quick start — 3 minutes, no large downloads

The [`examples/`](examples/) folder ships **4 annotated images** (3 MB) from the
held-out test set, so the pipeline can be exercised without fetching the full
dataset from Zenodo. It deliberately includes one **negative** (a closed wound
with zero polygons) because negatives are part of the dataset design.

| File role | Cell line | Size | Instances |
|---|---|---|---|
| Wound, annotated | HUVEC | 640 × 640 | 1 |
| **Negative** (closed wound) | HUVEC | 640 × 640 | 0 |
| Native resolution | SKOV-3 | 2452 × 2056 | 2 |
| Different timepoint | HUVEC | 640 × 640 | 1 |

See [`examples/MANIFEST.csv`](examples/MANIFEST.csv) for provenance and MD5 of each.

### 1 · Run inference on an example image

```bash
python predict_example.py
```

Downloads nothing: uses the bundled images. To use your own weights or image:

```python
from ultralytics import YOLO
import padding_patch; padding_patch.apply("black")   # the models were trained with black padding

model = YOLO("M.pt")                                 # huggingface.co/nmariotto/scratch-assay-segmentation
r = model.predict("examples/images/<file>.png", conf=0.8, imgsz=640, retina_masks=True)
```

### 2 · Reproduce the method-agreement statistics (~30 s)

```bash
python stage3/concordancia_final.py
```

All values reproduce bit-for-bit on Linux, macOS, and Windows (fixed bootstrap
seed, `seed=42`). See [Reproducibility](#reproducibility).

### 3 · Retrain (needs GPU + full dataset)

```bash
python stage2/run_grid.py --dry-run     # shows the 25-run ablation grid, trains nothing
python stage2/run_grid.py               # ~30 h on an RTX 4060 Ti; resumable
```

The grid is **resumable and crash-safe**: a run only counts as finished when it
writes a `COMPLETED.json` sentinel whose epoch count cross-checks `results.csv`.
A crashed run is redone, never skipped.

### Run inference on your own image

Three options, in order of convenience:

1. **No-install (browser):** open the [Hugging Face Space](https://huggingface.co/spaces/nmariotto/Scratch-assay-segmentation), upload an image, get the segmented wound gap with closure-fraction estimate.
2. **Local (Python):** download `M.pt` (45 MB, default) or `S.pt` (20 MB, roughly twice as fast on CPU for about four percentage points less recall) from the [model repository](https://huggingface.co/nmariotto/scratch-assay-segmentation):
   ```python
   from ultralytics import YOLO
   import padding_patch; padding_patch.apply("black")

   model = YOLO('M.pt')
   results = model.predict('your_image.png', conf=0.8, imgsz=640, retina_masks=True)
   ```
   `padding_patch` and `retina_masks=True` are not optional if you want the numbers the
   manuscript reports: Ultralytics pads with grey by default and returns masks on an
   internal grid, and both change the measured area.
3. **Full reproduction (retrain):** download the dataset from the [Zenodo deposit](https://doi.org/10.5281/zenodo.20298129) and follow the training protocol in Section 2.6 of the manuscript.

### Download Zenodo artifacts programmatically

```bash
pip install zenodo-get
zenodo_get 10.5281/zenodo.20298129     # downloads ALL files (~500 MB)
# OR fetch individual files:
# or fetch the deployed weights straight from the model repository:
wget https://huggingface.co/nmariotto/scratch-assay-segmentation/resolve/main/M.pt
```

## Reproducibility

`python stage3/concordancia_final.py` recomputes every agreement statistic the
manuscript reports, from the deposited CSVs. It pairs the closure fraction predicted by
each trained model on the held-out partition with the supervised reference standard, over
the 97 observations returned by all ten runs (five seeds of the deployed configuration and
five of the U-Net comparator), drawn from 45 acquisition series of both cell
lines.

Each statistic is computed per seed and reported as mean ± SD across seeds, so the spread
is the seed-to-seed variability of the estimator rather than the sampling error of one run.

| Metric | Deployed configuration (M) | U-Net comparator |
|---|---|---|
| Pearson *r* | 0.820 ± 0.042 | 0.855 ± 0.018 |
| Spearman ρ | 0.841 ± 0.039 | 0.861 ± 0.021 |
| Lin's CCC | 0.803 ± 0.049 | 0.853 ± 0.018 |
| Bland-Altman bias (model − reference) | +0.053 ± 0.018 | +0.012 ± 0.002 |
| 95% limits of agreement | -0.288 to +0.395 | -0.277 to +0.301 |
| Proportion within LoA | 93.4% | 93.2% |
| TOST equivalence, ±0.10 margin | 4 of 5 seeds | 5 of 5 seeds |
| TOST equivalence, ±0.05 margin | **0 of 5 seeds** | 5 of 5 seeds |
| Linear regression, model ~ reference | slope 0.873 ± 0.053, intercept 0.123 ± 0.037 | slope 0.805 ± 0.017, intercept 0.119 ± 0.008 |

**Read the limits of agreement, not only the correlation.** They span roughly two thirds
of the closure-fraction scale for both arms, so a single automated measurement can differ
from a careful manual one by about thirty percentage points of closure. The workflow is
suitable for comparing conditions across many wells; it is not a substitute for measuring
one well by hand. Equivalence at the stricter ±0.05 margin is reached by the U-Net in every
seed and by the detector in none.

Across the five evaluated configurations mean mAP@50 spans 93.3–94.0%, and none of the
ten pairwise differences has a cluster-bootstrap confidence interval excluding zero: on this
test set the configurations are not distinguishable. `stage3/aggregate.py` reproduces that
table.

The `reproduce-agreement.yml` workflow under `.github/workflows/` re-runs the analysis on
Linux, macOS and Windows under Python 3.11 and 3.12, and checks 24 values against the ones
reported here and in the manuscript; the build fails if any drifts.

## Citation

If you use this code, dataset, models, or analysis pipeline, please cite the manuscript and the Zenodo deposit:

```bibtex
@article{Mariotto2026Scratch,
  author  = {Mariotto, Nycolas and Dos Santos, Geovana A. and Fioschi, Geovana R. and
             Higino, Andria C. M. and Passeti, Lu\'is F. P. and Pereira, Thain\'a O. B. and
             Zampieri, Gabriela M. and Aal, Mirian C. E. and Delella, Fl\'avia K. and
             Sandrim, Val\'eria C. and Alves, Allan F. F.},
  title   = {Deep Learning Instance Segmentation for Quantitative Analysis of Cell Migration
             in Wound Healing Assays: A Benchmark and Web-Accessible Tool},
  journal = {Cytometry Part A},
  year    = {2026},
  note    = {Submitted}
}

@dataset{Mariotto2026Zenodo,
  author    = {Mariotto, Nycolas and Dos Santos, Geovana A. and Fioschi, Geovana R. and
               Higino, Andria C. M. and Passeti, Lu\'is F. P. and Pereira, Thain\'a O. B. and
               Zampieri, Gabriela M. and Aal, Mirian C. E. and Delella, Fl\'avia K. and
               Sandrim, Val\'eria C. and Alves, Allan F. F.},
  title     = {Deep Learning Instance Segmentation for Wound Healing Assays --
               Annotated Image Dataset, Trained Models, and Analysis Pipeline},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.20298129},
  url       = {https://doi.org/10.5281/zenodo.20298129}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata. GitHub displays a "Cite this repository" button in the sidebar that exports BibTeX/APA/etc.

## Licensing

This repository carries three licences, because its parts have different origins.

| Component | License | File |
|---|---|---|
| Trained detection weights (S, M, X, M-white, M-scratch) | **AGPL-3.0** | [`LICENSE-AGPL-3.0.txt`](LICENSE-AGPL-3.0.txt) |
| Code that imports Ultralytics (training, evaluation, prediction) | **AGPL-3.0** | [`LICENSE-AGPL-3.0.txt`](LICENSE-AGPL-3.0.txt) |
| Statistical analysis code and the U-Net comparator | **MIT License** | [`LICENSE`](LICENSE) |
| Image dataset and COCO annotations | **CC BY 4.0** | [`LICENSE-CC-BY-4.0.txt`](LICENSE-CC-BY-4.0.txt) |

The detection models are trained with Ultralytics YOLO11, which is AGPL-3.0.
Ultralytics' position is that weights trained with their software, and code built
on it, are derivative works and must carry AGPL-3.0 unless a commercial licence is
obtained; none was. An earlier version of this repository and of the manuscript
released the weights under MIT, which was incorrect.

The U-Net comparator is an independent PyTorch implementation of Ronneberger et al.
(2015) and does not derive from Ultralytics, so it and its weights remain MIT.

[`NOTICE`](NOTICE) lists which licence applies to which file.

## Funding

This research was funded by São Paulo Research Foundation (FAPESP) under grants 2019/01869-7, 2021/12010-7, and 2024/01849-4, and by the National Council for Scientific and Technological Development (CNPq) under grants 308504/2021-6, 444682/2024-4, and 302614/2025-7.

## Contact

**Allan Felipe Fattori Alves, PhD** (corresponding author)
Department of Biophysics and Pharmacology
Institute of Biosciences, Botucatu — São Paulo State University (UNESP)
allan.alves@unesp.br · ORCID: [0000-0002-0954-9919](https://orcid.org/0000-0002-0954-9919)

For software issues specifically, please open a [GitHub Issue](https://github.com/nykemariotto/scratch-assay-segmentation/issues) instead of emailing.
