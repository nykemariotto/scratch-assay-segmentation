# `dataset/` — Annotated images and COCO annotations (download from Zenodo)

The image dataset (`dataset.zip`, 65 MB) **is not redistributed in this Git repository** because its permanent archive is on Zenodo, which is the appropriate platform for research data (versioned, citable, persistent DOI).

## Where to get the dataset

> **https://doi.org/10.5281/zenodo.20298129** (concept DOI — always resolves to latest version)
> **https://doi.org/10.5281/zenodo.20298130** (version-locked DOI for v1.0.0, archived for the manuscript)

## Download

```bash
wget https://zenodo.org/records/20298130/files/dataset.zip
unzip dataset.zip -d dataset/
```

Or with `zenodo_get`:

```bash
pip install zenodo-get
zenodo_get 10.5281/zenodo.20298130 --glob="dataset.zip"
unzip dataset.zip -d ./
```

## Contents (after unzipping)

```
dataset/
├── train/                          740 JPEG images + 1 COCO JSON
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/                          270 JPEG images + 1 COCO JSON
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/                           139 JPEG images + 1 COCO JSON
    ├── _annotations.coco.json
    └── *.jpg
```

| Property | Value |
|---|---|
| Total images | 1,149 (815 HUVEC + 334 SKOV-3) |
| Format | JPEG, 640 × 640 px (resized from 2,452 × 2,056 px native with black-edge or white-edge padding) |
| Train / Valid / Test | 740 / 270 / 139 (fixed split, identical across all 6 model configurations) |
| Annotation format | COCO Instance Segmentation (polygonal) |
| Total polygons | 1,151 across 1,020 images |
| Images without polygon | 129 (wells with complete wound closure, retained as negative examples) |
| Single class | `center` (the wound gap to be segmented) |

## Roboflow version equivalence

The four Roboflow versions used to train the deposited models (v23 for Model 1, v24 for Model 2, v36 for Model 5, v37 for Model 6) share **identical images, annotations, and train/valid/test split**, differing only in preprocessing (padding color: black-edge vs white-edge) and augmentation parameters. The deposited `dataset.zip` corresponds to **v37** specifically; the others can be reconstructed by re-applying the corresponding padding/augmentation to the same underlying images.

The higher-resolution native acquisitions (2,452 × 2,056 px PNG) used to generate all Roboflow versions are **archived separately** and available from the corresponding author upon reasonable request.

## Licensing

The image dataset and COCO annotations are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. See [`LICENSE-CC-BY-4.0.txt`](../LICENSE-CC-BY-4.0.txt) at the repository root.

You are free to share, adapt, and use the dataset commercially, provided you give appropriate attribution (cite the Zenodo deposit and/or the companion paper).
