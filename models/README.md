# `models/` — Trained PyTorch weights (download from Zenodo)

The trained model weights (`.pt` files, ~410 MB combined) **are not redistributed in this Git repository** because their permanent archive is on Zenodo, which is the appropriate platform for large research data (versioned, citable, persistent DOI).

## Where to get the weights

All four deposited model weights are at:

> **https://doi.org/10.5281/zenodo.20298129** (concept DOI — always resolves to latest version)

The current version (v1.0.0, archived for the manuscript) is at:

> **https://doi.org/10.5281/zenodo.20298130** (version-locked DOI)

## Download options

### Option A — Browser

Go to https://zenodo.org/records/20298130 and click **Download** next to each `.pt` file you want.

### Option B — Command line (recommended)

Download a single model:

```bash
wget https://zenodo.org/records/20298130/files/Model_6_YOLOv11_Accurate.pt
wget https://zenodo.org/records/20298130/files/Model_2_Roboflow30_XL_black.pt
wget https://zenodo.org/records/20298130/files/Model_1_YOLOv11_XL_black.pt
wget https://zenodo.org/records/20298130/files/Model_5_Roboflow30_XL_white.pt
```

Download all four at once with the [`zenodo_get`](https://pypi.org/project/zenodo-get/) helper:

```bash
pip install zenodo-get
mkdir -p models && cd models
zenodo_get 10.5281/zenodo.20298130 --glob="Model_*.pt"
```

## Deposited models

| Model | File | Architecture | Padding | Init checkpoint | mAP@50 | mAP@75 | Precision | Recall | RF version | Size |
|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | `Model_1_YOLOv11_XL_black.pt` | YOLOv11 Instance Segmentation (Extra Large) | Black | COCOx-seg | 0.954 | — | — | — | v23 | 119 MB |
| Model 2 † | `Model_2_Roboflow30_XL_black.pt` | Roboflow 3.0 Instance Segmentation (Extra Large) | Black | COCOx-seg | 0.964 | 0.910 | 0.937 | 0.957 | v24 | 137 MB |
| Model 5 | `Model_5_Roboflow30_XL_white.pt` | Roboflow 3.0 Instance Segmentation (Extra Large) | White | Model 2 (transfer) | — | — | — | — | v36 | 137 MB |
| Model 6 † | `Model_6_YOLOv11_Accurate.pt` | YOLOv11 Instance Segmentation (Accurate variant) | White | COCOs-seg | 0.937 | 0.880 | 0.956 | 0.942 | v37 | 19.6 MB |

† served by the live Hugging Face web tool: https://huggingface.co/spaces/nmariotto/Scratch-assay-segmentation

## Models 3 and 4 (RF-DETR-Seg) are NOT deposited

The two RF-DETR-Seg configurations evaluated in the paper are not redistributed here. Reasons:

- The RF-DETR-Seg architecture was the slowest and lowest-recall variant (Recall 0.892, much lower than Model 2's 0.957).
- The architecture and training protocol are fully documented in Table 1 and Section 2.6 of the companion paper, so the weights can be regenerated from the deposited dataset.
- Checkpoint files are available from the corresponding author upon reasonable request (see contact in the main README).

## How to load

The four deposited models load directly via Ultralytics:

```python
from ultralytics import YOLO

# Lightweight option (20 MB, deployed in the web tool, white-edge padding)
model = YOLO('Model_6_YOLOv11_Accurate.pt')
results = model.predict('your_image.jpg', conf=0.8, imgsz=640)

# Best-overall option (137 MB, deployed in the web tool, black-edge padding)
model = YOLO('Model_2_Roboflow30_XL_black.pt')
results = model.predict('your_image.jpg', conf=0.8, imgsz=640)
```

Full training metadata (hyperparameters, epoch counts, augmentation settings) is in `models_metadata.json` within the Zenodo deposit.

## License

The trained model weights are released under the **MIT License** (see [`LICENSE`](../LICENSE) at repo root).
