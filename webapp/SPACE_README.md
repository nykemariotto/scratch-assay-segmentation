---
title: Scratch Assay Segmentation
emoji: 🔬
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.61.1
app_file: app.py
pinned: false
license: agpl-3.0
models:
- nmariotto/scratch-assay-segmentation
---

# Scratch Assay Segmentation Tool

Automated segmentation of the cell-free gap in brightfield scratch (wound-healing) assay
images, and measurement of its area.

Companion to *"Deep Learning Instance Segmentation for Quantitative Analysis of Cell
Migration in Wound Healing Assays"* (Mariotto et al., Cytometry Part A, under revision).

## What it does

Upload one image or a folder. The tool returns the segmented gap, its area in pixels² and
— if you supply a scale — in µm², and the fraction of the field it occupies. Results
export as a table, an overlay image and the contour coordinates.

Two model scales are offered. They differ **only** in size: initialisation, padding and
training schedule are identical, and across the five configurations evaluated in the study
no pairwise difference in mean Average Precision is distinguishable. The choice is latency
against recall, not accuracy.

| | mAP@50 | Recall | CPU |
|---|---|---|---|
| **M**: default | 93.4 ± 1.1% | 78.3 ± 3.0% | ~345 ms |
| **S**: fast mode | 94.0 ± 0.7% | 74.3 ± 2.3% | ~174 ms |

Mean ± SD over five training seeds on a held-out test set of 234 images.

## What it is not for

Agreement with a careful manual measurement has 95% limits of agreement of roughly ±0.3 in
closure fraction. **A single automated measurement is not a substitute for a single manual
one.** The tool is for comparing conditions across many wells.

Below about 5% of the field the gap becomes hard to delineate and agreement degrades; the
interface flags those measurements.

## Weights

Downloaded at run time from <https://huggingface.co/nmariotto/scratch-assay-segmentation>
(`M.pt`, `S.pt`). The repository is public and needs no token, so the exact file behind any
prediction can be retrieved, checked and redeployed independently.

## Licence

**AGPL-3.0.** This application and the weights it serves derive from Ultralytics YOLO11,
which is AGPL-3.0; no commercial licence was obtained. The image dataset is released
separately under CC BY 4.0 and the statistical analysis code under MIT.

## Links

- Data and code archive: <https://doi.org/10.5281/zenodo.20298129> (concept DOI, always
  the latest); this interface serves the weights of **v2.0.0**,
  <https://doi.org/10.5281/zenodo.21779854>
- Source: <https://github.com/nykemariotto/scratch-assay-segmentation>

Developed by the Medical Physics Laboratory, Department of Biophysics and Pharmacology,
IBB — UNESP. FAPESP 2024/01849-4. Coordination: Prof. Allan Alves. Development: Nycolas
Mariotto.