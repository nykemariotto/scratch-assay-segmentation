# Web interface — Hugging Face Space source

The files deployed at
<https://huggingface.co/spaces/nmariotto/Scratch-assay-segmentation>, versioned here
so that the source of the tool described in the manuscript is auditable from one place.

| File | Where it goes |
|---|---|
| `app.py` | Space, root |
| `requirements.txt` | Space, root |
| `padding_patch.py` | Space, root — `app.py` imports it; without it the Space does not start |
| `MODEL_CARD.md` | model repository, as its `README.md` — on Hugging Face a model repository's README *is* its model card |
| `SPACE_README.md` | Space, as its `README.md` — the YAML front matter is what configures the Space |

### `sdk_version` tracks the Space, not the other way round

Hugging Face bumps `sdk_version` in the Space's own `README.md` when it rebuilds, and
it does so without telling anyone. The copy here has to be updated to match, because a
redeploy pushes this file over the Space's: on 2026-08-04 the Space was running
Streamlit 1.60.0 while this copy still declared 1.59.2, so redeploying would have
silently **downgraded** the runtime. Check the deployed value before pushing, not after.

The weights are **not** stored in the Space. They are downloaded at run time from the
public model repository <https://huggingface.co/nmariotto/scratch-assay-segmentation>
(`M.pt`, `S.pt`, AGPL-3.0). Earlier versions of the manuscript stated that the weights
were hosted inside the Space; they never were.

## Secrets the Space needs

`GOOGLE_DRIVE_REFRESH_TOKEN`, `GOOGLE_DRIVE_CLIENT_ID`, `GOOGLE_DRIVE_CLIENT_SECRET`
and `feedback_sheet_url`, all for the optional feedback log. The model download needs
no token: the repository is public, which is what makes the claim that any user can
retrieve and re-run the exact weights true.

## The feedback log keeps both model generations in one sheet

Decided rather than defaulted. Feedback rows written by app 3.0 carry the model keys
`Model_2` and `Model_6`; rows written by 4.0 carry `M` and `S`. The keys are distinct
strings, so filtering the model column alone separates the two generations — the
`APP_VERSION` column is a cross-check, not the discriminator.

The two generations are **not comparable**: 3.0 served weights trained on a dataset
partitioned at the image level, and reported a slightly different area for the same
image (see below). A rating of "Bad" under 3.0 and one under 4.0 are judgements about
different systems.

## The area is computed as in the manuscript

`mask_area_px()` counts the pixels of the union of every predicted mask, with
`retina_masks=True`, which is what `stage3/predict_areas.py` does. Version 3.0 of the
app took the polygon area of the single highest-confidence mask instead; measured over
14 test images that under-reported by 0.6% at the median and 2.3% at worst, with the
error growing as the wound shrank.
