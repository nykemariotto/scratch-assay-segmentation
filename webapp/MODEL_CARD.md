---
license: agpl-3.0
tags:
  - image-segmentation
  - instance-segmentation
  - ultralytics
  - yolo
  - cell-biology
  - microscopy
  - wound-healing-assay
library_name: ultralytics
pipeline_tag: image-segmentation
---

# Scratch-assay wound segmentation — YOLO11-seg

Instance-segmentation weights for the cell-free gap in brightfield scratch (wound-healing)
assay images. Two scales are provided: `M.pt` is the default and `S.pt` is a faster
alternative that trades about four percentage points of recall for roughly half the CPU
latency.

These weights accompany the manuscript *"Deep Learning Instance Segmentation for
Quantitative Analysis of Cell Migration in Wound Healing Assays"* (Cytometry Part A, under
revision).

## ⚠️ These are not the weights of the originally submitted version

The first submission evaluated models trained on a dataset partitioned at the **image**
level, so frames of the same acquisition field appeared in both training and test sets. Every
performance figure in that version was optimistic. The dataset was rebuilt and partitioned at
the level of the **physical acquisition field**, and all configurations were retrained. The
weights here are from the corrected partition. The superseded weights stay in the earlier
deposit (10.5281/zenodo.20298130), which is marked as superseded in full and should not
be cited. They are retained for the record, not for use.

## Files

| File | Scale | Parameters | Size | CPU (ms/img) | GPU (ms/img) | mAP@50 | Recall |
|---|---|---|---|---|---|---|---|
| `M.pt` | YOLO11m-seg | 22.4 M | 45.2 MB | 345 | 92 | 93.40 ± 1.08 | 78.34 ± 2.99 |
| `S.pt` | YOLO11s-seg | 10.1 M | 20.5 MB | 174 | 85 | 93.98 ± 0.74 | 74.32 ± 2.30 |

Accuracy is mean ± SD over five training seeds on a held-out test set of 234 images from 37
acquisition groups. Latency is the median over 40 images, CPU on 16 cores, GPU on an NVIDIA
GeForce RTX 4060 Ti. Both weights are the seed-42 checkpoint.

**The two scales are not statistically distinguishable in mAP.** Across all five
configurations evaluated, mean mAP@50 spans 93.3–94.0% and none of the ten pairwise
differences has a cluster-bootstrap confidence interval excluding zero. The reason to prefer
`M` is recall at the deployed operating point, not accuracy in aggregate.

## Training

| | |
|---|---|
| Architecture | YOLO11-seg (Ultralytics 8.4.102) |
| Initialisation | COCO checkpoint |
| Input | 640 × 640, letterbox with black padding |
| Schedule | 100 epochs, batch 4, early stopping disabled |
| Augmentation | HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5, mosaic (off for the last 10 epochs) |
| Framework | PyTorch 2.6.0, CUDA 12.4 |
| Seeds | 42–46; the checkpoint published here is seed 42 |

Training data: 932 images (197 validation, 234 held-out test) of HUVEC and SKOV-3 monolayers,
brightfield, 5× objective, single laboratory. One class, `wound`.

## Usage

```python
from ultralytics import YOLO

model = YOLO("M.pt")                  # or S.pt for the fast variant
r = model.predict("scratch.png", conf=0.80, retina_masks=True)
mask = r[0].masks                     # polygon and binary mask of the gap
```

`conf=0.80` is the operating point at which the reported precision and recall were measured
and the default of the companion web interface.

## What these weights are good for, and where they fail

Agreement with a supervised reference standard, over 97 paired observations from both cell
lines: Pearson r 0.820 ± 0.042, Lin's concordance correlation coefficient 0.803 ± 0.049, mean
bias +0.053 ± 0.018 in the closure fraction, 95% limits of agreement −0.288 to +0.395.

Read that last number carefully. **A single automated measurement can differ from a careful
manual one by about thirty percentage points of closure.** The workflow is suitable for
comparing conditions across many wells; it is not a substitute for manual measurement of an
individual well.

Two known limits:

- **Small wounds.** Performance is governed by wound size rather than elapsed time. Above 10%
  of the field the model agrees closely with the reference standard; between 2% and 5% mean
  intersection over union falls to 0.362. Measurements taken while the residual gap is below
  roughly 5% of the field are the least reliable part of a series.
- **Isolated cells.** The gap is quantified as the region enclosed by the segmented contour,
  so cells that detach and migrate individually into an otherwise continuous gap are not
  subtracted from it.

The ceiling is not the architecture. On blinded repeat corrections the human observer
reproduced their own delineation at a median intersection over union of 0.861 on the 10
pairs that measure boundary reproducibility, and 0.894 over all 14, about what
the models achieve. What limits agreement here is the reproducibility of the wound boundary
itself.

Acquisition envelope: one inverted microscope, brightfield, 5× objective, one institution. No
external testing was performed. Robustness to other microscopes, magnifications or contrast
modalities is unknown.

## Licence

**AGPL-3.0.** These weights are trained with Ultralytics YOLO11, which is AGPL-3.0, and no
commercial licence was obtained; the trained weights are a derivative work and carry the same
licence. The image dataset is released separately under CC BY 4.0, and the statistical
analysis code under MIT.

## Citation

Data and code archive: https://doi.org/10.5281/zenodo.20298129 (concept DOI, always the
latest). The figures on this page are those of **v2.0.0**,
https://doi.org/10.5281/zenodo.21779854 — cite that one to pin the version these weights
came from.
Code repository: https://github.com/nykemariotto/scratch-assay-segmentation
Web interface: https://huggingface.co/spaces/nmariotto/Scratch-assay-segmentation

Contact: Allan F. F. Alves — allan.alves@unesp.br — ORCID 0000-0002-0954-9919
