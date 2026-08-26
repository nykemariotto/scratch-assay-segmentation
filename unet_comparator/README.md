# U-Net — the deep-learning arm of the benchmark

The **deep-learning** comparator that answers the request for *"benchmarking against at
least one classical automated tool and one deep-learning tool"*.

The three roles do not overlap:

| Role | Filled by |
| --- | --- |
| **Reference standard** (the ruler) | automated WHST **plus supervised manual correction** — Stage 4 |
| **Classical comparator** | WHST in **pure automatic mode** — `benchmark_classical.py`, already measured |
| **DL comparator** | **this U-Net** |

This U-Net **does not replace WHST** in either of the other two roles. It fills the row that
was empty.

---

## What this is, and what it is not

**It is** the canonical U-Net architecture (Ronneberger, Fischer & Brox, 2015) trained under
conditions identical to those of the YOLO11 grid, for an **architectural** comparison.

**It is not** a run of the tool published by Doğru, Ekinci & Akbulut (*BMC Med Imaging*
2024;24:15). We reimplemented the architecture rather than copying their hyperparameters,
for a methodological reason: copying their hyperparameters while swapping the dataset
neither reproduces their work nor produces a fair comparison — it produces a hybrid that is
neither.

**This has to appear in the Methods.** A reader who sees "we compared against Doğru et al."
and finds a reimplementation reacts badly; one who reads "we compared against the U-Net
architecture under identical conditions, and we state that this is not the published tool"
has nothing to object to.

---

## Fairness contract

What makes the comparison valid. If any row breaks, it becomes advocacy.

| Item | YOLO11 grid | U-Net |
| --- | --- | --- |
| Partition | `data.yaml` | **the same file**, not a copy |
| Resolution | 640 × 640 | 640 × 640 |
| Letterbox / padding | black (0) | black (0) |
| Epochs | 100 | 100 |
| Early stopping | disabled | disabled |
| Seeds | 42–46 | 42–46 |
| Batch | 4 | 4 |
| Determinism | seeds + deterministic cuDNN, autotuning off | same |
| HSV jitter | 0.015 / 0.7 / 0.4 | same |
| Translate / scale / fliplr | 0.1 / 0.5 / 0.5 | same |
| Augmentation per epoch | redrawn every epoch | same |
| Data order | **does not vary** with the seed | same (`SEED_AUG` fixed) |
| What the seed governs | weight initialisation | same |
| Masks | YOLO-seg polygons | **the same polygons**, rasterised |

> The three middle rows come from a defect fixed on 2026-07-28 — see "Defects found in
> review", below. They matter because Table 2 puts the standard deviations of the two arms
> side by side: if one of them carried an extra source of variance, the numbers would not be
> comparable.

### The one asymmetry, declared

The grid uses **mosaic** (1.0, switched off for the last 10 epochs). Mosaic is a detection
augmentation — it tiles four images and crops — and is not part of any standard U-Net
pipeline. Applying it here would be inventing a method; omitting it is the honest choice. It
is the only augmentation difference between the arms, and it has to be stated in the Methods.

Two minor choices, also declared:

- **BatchNorm** in the double block. It is not in the 2015 paper (it did not exist yet), but
  it is universal in modern implementations; without it training diverges at a comparable
  learning rate.
- **Loss = 0.5·BCE + 0.5·Dice.** Dice alone is unstable with an empty mask, and there are 150
  negatives in the dataset. The BCE term keeps the gradient defined.

---

## Files

| File | What it does |
| --- | --- |
| `unet_model.py` | architecture (31.0 M parameters) + `BCEDiceLoss` |
| `unet_data.py` | dataset, polygon rasterisation, reversible letterbox, augmentation |
| `train_unet.py` | trains one seed; writes `provenance.json`, `pip_freeze.txt`, `COMPLETED.json` |
| `run_unet_grid.py` | the 5 seeds in isolated processes, with a GPU guard |
| `predict_unet.py` | test-set areas → `unet_test_areas.csv` |
| `smoke_test.py` | validates everything **without training and without touching the GPU** |
| `estimate_cost.py` | parameters, MACs and CPU timings, on CPU only |

## How to run

Validation first — no GPU:

```bash
python unet_comparator/smoke_test.py
```

```bash
python unet_comparator/test_epoca_workers.py
```

Then the grid, and prediction from the resulting weights:

```bash
python unet_comparator/run_unet_grid.py --dry-run
```

```bash
python unet_comparator/run_unet_grid.py
```

```bash
python unet_comparator/predict_unet.py --weights runs/segment/unet_comparator/unet_black_seed42/best.pt
```

`run_unet_grid.py` **refuses to start** if it detects the YOLO grid running (any `results.csv`
without a sentinel written in the last 15 min). The guard is heuristic — `--force` overrides
it, but check first.

---

## Validation state (smoke test, CPU)

Everything verified without consuming GPU:

- partition matches: 932 / 197 / 234
- rasterisation: 205 of the 234 test images carry a mask; median area 14.0% (min 2.4 · max
  41.2); the remaining 29 are negatives and become empty masks
- **reversible letterbox**: round-trip area error 2452→640→2452 — worst 0.29%, median 0.04%.
  This is the check that matters most: area is the quantity the benchmark compares
- augmentation deterministic given (seed, epoch, index), and distinct samples do differ
- one CPU training iteration: finite loss, non-zero gradient, loss decreases
- an empty mask does not produce NaN

Two defects were found and fixed by the smoke test itself:

1. `CUDA_VISIBLE_DEVICES=""` does **not** hide the GPU on Windows — the value is `-1`.
   Without the assertion the test would have contended for the card with the grid.
2. `array_uint8.sum()` returns an **unsigned** integer; `a1 - a0` wrapped around when the
   area decreased, reporting a 10¹⁶ % error in the letterbox round-trip.

## Defects found in review (2026-07-28)

### The augmentation did not vary between epochs

The RNG was seeded with `(seed, index)` — **no epoch**. Each image got one fixed
transformation and repeated it for all 100 epochs. That is not stochastic augmentation: it is
a dataset transformed once. And it broke the fairness contract, because YOLO draws a new
transformation every epoch.

**The smoke test was certifying the defect.** It asserted *"determinism: same sample, same
output"* as though that were the desired property. Determinism has to be over **(seed, epoch,
index)**, not (seed, index) — the same class of error as `stage2/verify_padding.py`: a test
that cannot fail for the right reason.

Fix: `WoundDataset.set_epoca(e)`, called by the training loop before each epoch; RNG seeded
with `(SEED_AUG, epoch, index)`.

### Data order varied with the seed — and in YOLO it does not

The `DataLoader` used `generator.manual_seed(args.seed)`. Ultralytics uses a constant,
so there the order does **not** vary with the seed. Had the asymmetry been kept, the U-Net arm
would have had one more source of variance than the YOLO arm, and the standard deviations in
Table 2 would not be comparable.

Fix: the shuffle generator is seeded with `SEED_AUG`, a constant. In both arms the seed
governs **only** weight initialisation.

### Verification

`test_epoca_workers.py` confirms that `set_epoca` survives the spawn — the augmentation varies
by epoch **inside the workers too**, and remains reproducible given the epoch:

With 0 and with 2 workers, the same index gives a **different** sample in epoch 1 than in
epoch 2, and an **identical** sample when epoch 1 is repeated. Both pass. (The script's own
console output is still in Portuguese.)

It has to be a file, not a heredoc: on Windows the spawn re-imports `__main__`, and without
the guard the process hangs.

## Cost — predicted, then measured

`estimate_cost.py` (CPU, never touches the GPU) against `yolo11m-seg`, same 1×3×640×640 input:

| | U-Net | yolo11m-seg | ratio |
| --- | --- | --- | --- |
| parameters | 31.0 M | 22.4 M | 1.38× |
| **MACs / forward** | **301.8 G** | **56.9 G** | **5.30×** |
| forward (CPU) | 1.36 s | 0.35 s | 3.88× |
| fwd+bwd (CPU) | 4.31 s | 1.01 s | 4.27× |

**Parameters are not work.** The weight count suggests 1.4×; the CPU cost is ~4–5×. The U-Net
runs its first two convolutions at **full** 640 × 640 with 64 channels, while YOLO drops to /2
at the stem and never returns to full resolution. That is where the 300 G of MACs accumulate.

Anchored on the 78 min/seed that `yolo11m-seg` actually took on this GPU, the projection was
**4.9 to 6.7 h/seed**, i.e. **25 to 34 h** for five seeds — explicitly labelled a **ceiling**,
because MACs and CPU time overstate GPU cost for networks with heavy high-resolution
convolution, which is the pattern a GPU parallelises best.

**What it actually cost**, from `wall_seconds` in each `provenance.json`:

| seed | wall | best val IoU |
| --- | --- | --- |
| 42 | 3.01 h | 0.8684 |
| 43 | 2.99 h | 0.8649 |
| 44 | 3.02 h | 0.8658 |
| 45 | 2.98 h | 0.8731 |
| 46 | 2.98 h | 0.8653 |
| **5 seeds** | **15.0 h** | |

**3.00 h/seed against a 4.9 h floor** — the ceiling held, and with room to spare. Measured
against YOLO's 1.30 h/seed the real ratio is **2.30×**, not the 5.30× that MACs predict: the
GPU absorbs the high-resolution convolutions better than a MAC count can express. The lesson
is the one the estimate already stated — extrapolating GPU cost from MACs or CPU time gives a
bound, not a forecast.

The five seeds were run, so the comparator keeps full symmetry with the YOLO grid: mean ± SD
over the same five seeds on both arms. Reducing `base` to 32 or the resolution to 256 was
never on the table — either would have broken the fairness contract and voided the
comparison.
