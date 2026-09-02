# `models/` — Trained weights (download from Zenodo)

The trained weights (30 `.pt` files, 2.0 GB combined) **are not redistributed in this Git
repository**. They are archived on Zenodo, inside two of the deposit's files.

## Where to get them

> **https://doi.org/10.5281/zenodo.21779854** — version DOI for **v2.0.0**, the version
> this study uses
> **https://doi.org/10.5281/zenodo.20298129** — concept DOI, always resolves to the
> latest version

**v1.0.0 (10.5281/zenodo.20298130) is superseded and should not be used.** Its models were
trained on a partition drawn at the image level, so frames of the same acquisition field
appeared on both sides of the split; none of its reported metrics is comparable with the
ones below.

| Archive | Contents | Size |
|---|---|---|
| `models_yolo.zip` | 25 YOLO11 runs — 5 configurations × 5 seeds | 1,404 MB |
| `models_unet.zip` | 5 U-Net comparator runs — 5 seeds | 621 MB |

```bash
wget https://zenodo.org/records/21779854/files/models_yolo.zip
unzip models_yolo.zip
```

Or with `zenodo_get`:

```bash
pip install zenodo-get
zenodo_get 10.5281/zenodo.21779854 --glob="models_*.zip"
```

## Layout inside the archive

```
runs/segment/runs_revision/<run_base>_seed<N>/weights/best.pt
```

with `<run_base>` one of the five below and `<N>` in 42–46. Each run directory also carries
its `provenance.json`, which records the architecture, padding, initialization and seed.

## The five configurations

They differ in **one variable at a time** from `M`: model scale (`S`, `X`), padding colour
(`M-white`), and initialization (`M-scratch`). Everything else — partition, resolution,
schedule, augmentation — is identical.

| Config | `run_base` | Params | Weight | mAP@50 | mAP@75 | Precision | Recall |
|---|---|---:|---:|---|---|---|---|
| **S** | `yolo11s-seg_black_coco` | 10.1 M | 20.5 MB | 93.98 ± 0.74 | 85.22 ± 1.17 | 98.97 ± 0.81 | 74.32 ± 2.30 |
| **M** | `yolo11m-seg_black_coco` | 22.4 M | 45.2 MB | 93.40 ± 1.08 | 85.47 ± 0.88 | 97.49 ± 2.31 | 78.34 ± 2.99 |
| **X** | `yolo11x-seg_black_coco` | 62.1 M | 124.8 MB | 93.32 ± 0.50 | 85.25 ± 0.59 | 97.52 ± 0.97 | 78.78 ± 0.79 |
| **M-white** | `yolo11m-seg_white_coco` | 22.4 M | 45.2 MB | 93.52 ± 0.57 | 85.51 ± 0.68 | 98.10 ± 0.79 | 76.16 ± 1.71 |
| **M-scratch** | `yolo11m-seg_black_scratch` | 22.4 M | 45.2 MB | 93.35 ± 0.11 | 83.84 ± 0.61 | 100.00 ± 0.00 | 63.58 ± 4.33 |

Percentages, mean ± standard deviation across the five seeds, on the held-out test
partition at a confidence threshold of 0.80. mAP@50 and mAP@75 integrate precision over
the full recall range and do not depend on that threshold; precision and recall do.

**None of these is the best.** The mAP@50 range spans 0.66 percentage points while the
spread across seeds *within* one configuration reaches 1.1, and no pairwise difference has
a confidence interval excluding zero. The `M-scratch` precision of 100.00 ± 0.00 is not an
achievement: it detects far less (recall 63.58%), and what it does detect it gets right.

Full per-configuration statistics, including mAP@50-95, F1 and bootstrap confidence
intervals, are in `stage3/table2.csv`.

## The U-Net comparator

`models_unet.zip` holds five seeds of an independent PyTorch implementation of the U-Net
architecture (Ronneberger, Fischer & Brox, 2015), 31.0 M parameters, 124.2 MB per seed. It
is a comparator, not a deployed model, and it produces a single semantic mask rather than
scored instances — so instance-level mAP does not apply to it.

## Which two are deployed

The web interface serves the seed-42 checkpoints of **M** (default) and **S** (fast mode),
from the model repository at
[`nmariotto/scratch-assay-segmentation`](https://huggingface.co/nmariotto/scratch-assay-segmentation),
where they are named `M.pt` and `S.pt`.

The choice is latency, not accuracy. Median inference on a CPU over 40 images, after three
discarded warm-up predictions:

| | CPU | GPU |
|---|---:|---:|
| S | 174 ms | 85 ms |
| M | 345 ms | 92 ms |
| X | 699 ms | 112 ms |
| U-Net | 1,604 ms | 165 ms |

`X` costs four times the CPU latency of `S` for no measurable gain in mAP, so it is not
deployed. The U-Net is impractical for an interactive interface on a CPU. Measurements are
in `stage3/latencia_inferencia.csv`.

## Loading

The 25 YOLO checkpoints load directly with Ultralytics:

```python
from ultralytics import YOLO
model = YOLO("runs/segment/runs_revision/yolo11m-seg_black_coco_seed42/weights/best.pt")
results = model.predict("image.png", conf=0.80)
```

The U-Net checkpoints are plain PyTorch `state_dict` files and need the architecture in
`unet_comparator/unet_model.py`; see `unet_comparator/README.md`.

## Licensing

The YOLO11 weights are **AGPL-3.0**, because Ultralytics treats models trained with its
software as derivative works and no commercial licence was obtained. The U-Net weights are
**MIT**, being an independent implementation that does not derive from Ultralytics. See
[`NOTICE`](../NOTICE) for how the three licences of this work divide.
