# `dataset/` — Annotated images and COCO annotations (download from Zenodo)

The image dataset (`dataset.zip`, 3.7 GB) **is not redistributed in this Git repository**
because its permanent archive is on Zenodo, which is the appropriate platform for research
data (versioned, citable, persistent DOI).

## Where to get the dataset

> **https://doi.org/10.5281/zenodo.21779854** — version DOI for **v2.0.0**, the version
> this study uses
> **https://doi.org/10.5281/zenodo.20298129** — concept DOI, always resolves to the
> latest version

**v1.0.0 (10.5281/zenodo.20298130) is superseded and should not be used.** Its partition
was drawn at the image level, so frames of the same acquisition field appeared on both
sides of the split. v2.0.0 rebuilds the dataset from the raw acquisitions and partitions
it over acquisition units; no count in v1.0.0 is comparable with the ones below.

## Download

```bash
wget https://zenodo.org/records/21779854/files/dataset.zip
unzip dataset.zip -d ./
```

Or with `zenodo_get`:

```bash
pip install zenodo-get
zenodo_get 10.5281/zenodo.21779854 --glob="dataset.zip"
unzip dataset.zip -d ./
```

## Contents (after unzipping)

```
dataset/
├── images/
│   ├── train/                      932 PNG images
│   ├── val/                        197 PNG images
│   └── test/                       234 PNG images
└── labels/
    ├── train/                      932 YOLO polygon .txt (one per image)
    ├── val/                        197
    └── test/                       234
coco_partitions/
├── instances_train.json            932 images, 928 annotations
├── instances_val.json              197 images, 193 annotations
└── instances_test.json             234 images, 229 annotations
data/
├── mapping_dataset_final_strat.csv the partition table: grouping keys, stratum, split
├── mapping_huvec_final.csv         HUVEC acquisition records
└── mapping_final_skov.csv          SKOV-3 acquisition records
data.yaml                           Ultralytics dataset descriptor
```

| Property | Value |
|---|---|
| Total images | 1,363 (1,030 HUVEC + 333 SKOV-3) |
| Format | PNG. 1,248 at the native acquisition resolution of 2,452 × 2,056 px; 115 available only as a 640 × 640 export |
| Train / val / test | 932 / 197 / 234 |
| — HUVEC | 681 / 157 / 192 |
| — SKOV-3 | 251 / 40 / 42 |
| Annotation format | COCO instance segmentation (polygonal), plus YOLO polygon labels |
| Total polygons | 1,350 across 1,213 images |
| Images without polygon | 150 (103 train, 18 val, 29 test) — wells with complete wound closure, kept as negative examples |
| Single class | the wound gap to be segmented |

## How the partition was drawn

The split is over **acquisition units**, not over images. Each well is imaged at several
post-scratch time points, so an image-level split puts highly correlated frames of the
same field on both sides of it.

`data/mapping_dataset_final_strat.csv` carries both keys, and they are not the same
number:

| Column | Distinct values | What it is |
|---|---|---|
| `group_key` | 265 | the physical acquisition field |
| `split_key` | 246 | the merged super-key the partition actually respects |

The two differ because of a conservative merge: for 19 wells that appear in two raw-export
batches whose field identity across batches could not be confirmed from the acquisition
records, all images were assigned to the same partition. That precludes leakage at the
cost of merging groups that may in fact be independent. Zero overlap was verified on both
keys and on the exported annotation files, not only on the partition table.

## Provenance of the annotations

The polygons were drawn on Roboflow, and the COCO metadata retains that origin. The
deposited partition, however, is **not** a Roboflow export: it is built by
`stage1/export_coco_per_partition.py`, which splits the *base* export according to
`mapping_dataset_final_strat.csv`. That script refuses the versioned Roboflow exports by
design, because those carry the superseded image-level split.

## Licensing

The image dataset and COCO annotations are released under **Creative Commons Attribution
4.0 International (CC BY 4.0)**. See [`LICENSE-CC-BY-4.0.txt`](../LICENSE-CC-BY-4.0.txt)
at the repository root.

You are free to share, adapt, and use the dataset commercially, provided you give
appropriate attribution (cite the Zenodo deposit and/or the companion paper).
