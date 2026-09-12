import streamlit as st
import pandas as pd
import cv2
import hashlib
import os
import threading
import traceback
import zipfile
from PIL import Image
from io import BytesIO
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from huggingface_hub import hf_hub_download

# BEFORE ultralytics is used for anything. The models were trained with BLACK
# letterbox padding; Ultralytics 8.4.x hard-codes 114 (gray) as the fill value and
# does not expose it in the prediction API. Without this patch the app pads with
# gray and the tool silently stops being the configuration the manuscript
# evaluated: measured over six test images, that alone accounted for the whole
# residual disagreement with the published pipeline (-0.24% median, -0.52% worst),
# and with the patch the two agree to the pixel.
#
# Same module the evaluation pipeline uses, not a reimplementation.
import padding_patch

PADDING_FILL = padding_patch.apply("black")

from ultralytics import YOLO
import gspread
import numpy as np
import torch

st.set_page_config(page_title="Scratch Assay Segmentation", layout="wide")

APP_VERSION = "4.0"
DEFAULT_IMGSZ = 640

# Public model repository. The weights are AGPL-3.0, as they derive from
# Ultralytics YOLO11, and are downloadable without a token — the manuscript
# claims that the exact file behind any prediction can be inspected and
# redeployed independently, and a token-gated repository would make that false.
HF_MODEL_REPO = "nmariotto/scratch-assay-segmentation"

# Configurations M and S of the companion manuscript (Mariotto et al.,
# Cytometry Part A). They differ only in model scale; initialisation (COCO),
# padding colour (black) and training schedule are identical. The five
# configurations evaluated are not distinguishable in mean Average Precision,
# so neither of these is "the accurate one": the choice is latency against
# recall, and the labels say so.
MODEL_OPTIONS = {
    "M — default (22.4 M parameters)": "M.pt",
    "S — fast mode (10.1 M parameters)": "S.pt",
}

# Stable, filesystem-safe key for Drive folders and Sheet logging, decoupled
# from the user-visible label so relabeling does not fragment stored data.
MODEL_STORAGE_KEY = {
    "M — default (22.4 M parameters)": "M",
    "S — fast mode (10.1 M parameters)": "S",
}

# Measured on the held-out test set (n = 234), mean ± SD over five seeds;
# latency is the median model inference over 40 images on 16 CPU cores. Shown in
# the interface so the trade-off is stated rather than discovered.
MODEL_INFO = {
    "M": "mAP@50 93.4 ± 1.1% · recall 78.3 ± 3.0% · model inference ~345 ms per image "
         "on a 16-core CPU",
    "S": "mAP@50 94.0 ± 0.7% · recall 74.3 ± 2.3% · model inference ~174 ms per image "
         "on a 16-core CPU",
}
# The times above are model inference on a 16-core CPU. This hosted interface runs on
# the Space's own hardware, and it also receives the image and sends the result back,
# so a result takes a few seconds with either model. Said in the interface, so that the
# milliseconds are not read as what this page takes.
# Short enough to end before the open model list, which is drawn over the right-hand end
# of this line: the first wording ("…a few seconds per image with either model…") ran
# under it, and the Figure 2 capture showed the sentence cut off.
HOSTED_NOTE = ("In this hosted interface a result takes a few seconds with either model; "
               "the inference times above were measured on a 16-core CPU.")


# =========================
# Model init — public Hugging Face repository
# =========================
def _padding_efetivo():
    """Proves the padding reached LetterBox instead of assuming it did.

    A patch that is assumed to have applied, and has not, produces plausible
    numbers rather than an error, which is the hardest kind of defect to notice.
    The check costs one line, so it is made rather than assumed.
    """
    import ultralytics.data.augment as A
    return getattr(A.LetterBox(new_shape=(640, 640)), "padding_value", None)


@st.cache_resource
def load_model(model_filename):
    local_model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=model_filename,
        repo_type="model",
    )
    return YOLO(local_model_path)


# =========================
# Google Drive + Sheets (OAuth2), opened only to save feedback
# =========================
# Streamlit re-executes this whole file on every interaction. Connecting here, at
# the top, re-authenticated and reopened the feedback sheet on every click, for
# every user, including those who never save feedback. The connection is now made
# inside save_feedback, the only place that needs it.
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/drive",
                 "https://www.googleapis.com/auth/spreadsheets"]


def _google():
    """(drive, sheet) for one feedback save."""
    credentials = Credentials(
        token=None,
        refresh_token=st.secrets["GOOGLE_DRIVE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=st.secrets["GOOGLE_DRIVE_CLIENT_ID"],
        client_secret=st.secrets["GOOGLE_DRIVE_CLIENT_SECRET"],
        scopes=GOOGLE_SCOPES,
    )
    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    sheet = gspread.authorize(credentials).open_by_url(st.secrets["feedback_sheet_url"]).sheet1
    return drive, sheet


# =========================
# Helpers
# =========================
# Every user shares one model object per scale (load_model is a cached resource), and
# Ultralytics writes the settings of a call, the confidence threshold among them, into
# that shared object before taking its own lock (engine/model.py, predict). Two users
# predicting at once could each be served with the other's threshold. One lock around
# the whole call prevents it; on the Space's CPUs two predictions at once would not
# finish sooner anyway.
#
# The thread count. On the first prediction of each model Ultralytics sets PyTorch to
# min(8, os.cpu_count() - 1) threads (utils/torch_utils.py, select_device), and
# os.cpu_count() counts the host, not the container's share of it. Depending on what the
# host reports, a two-CPU Space then predicts on 1 thread or on up to 8. The count is set
# here, before every prediction, to the CPUs the container may use (at most 8), and
# printed once to the Space's log.
def _cgroup_cpu_limit(root="/sys/fs/cgroup"):
    """The container's CPU quota, in CPUs, or None when it has none."""
    try:  # cgroup v2: "<quota> <period>", or "max <period>" when unlimited
        with open(os.path.join(root, "cpu.max")) as f:
            quota, period = f.read().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open(os.path.join(root, "cpu", "cpu.cfs_quota_us")) as f:
            quota = int(f.read())
        with open(os.path.join(root, "cpu", "cpu.cfs_period_us")) as f:
            period = int(f.read())
        return quota / period if quota > 0 else None
    except (OSError, ValueError):
        return None


def usable_cpus():
    """Never more than the 8 that Ultralytics itself allows, so that a quota this cannot
    read (files hidden, or the limit set on a parent cgroup) gives no more threads than
    before, rather than one per core of the host."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    limit = _cgroup_cpu_limit()
    if limit:
        n = min(n, max(1, int(limit)))
    return min(n, 8)


@st.cache_resource
def inference_state():
    """Made once per process: Streamlit re-runs this file for every interaction of
    every user, and a lock created at the top would be a new lock on every run."""
    threads = usable_cpus()
    print(f"[inference] os.cpu_count() {os.cpu_count()}, cgroup CPU limit "
          f"{_cgroup_cpu_limit()}, PyTorch threads set to {threads}", flush=True)
    return threading.Lock(), threads


def predict(model, image_array, conf_threshold):
    """Same call as the evaluation pipeline: retina_masks gives masks at the
    original resolution instead of the model's internal 160 x 160 grid.

    A failure raises. It used to be retried three times, one second apart, and then
    reported as "No segmentation was detected", which turned a crash into the
    measurement of an image with no wound.
    """
    lock, threads = inference_state()
    with lock:
        torch.set_num_threads(threads)
        return model.predict(
            source=image_array,
            imgsz=DEFAULT_IMGSZ,
            conf=conf_threshold,
            retina_masks=True,
            verbose=False,
        )


def mask_area_px(result, height, width):
    """Wound area in pixels, computed exactly as in the manuscript.

    The published figures come from `stage3/predict_areas.py`, which counts the
    pixels of the UNION of every predicted mask. Two earlier choices in this app
    made it disagree with them:

      · it kept only the highest-confidence mask, so an image whose wound is
        split into two non-contiguous regions was under-reported. Rare (2 of the
        234 test images) but silent;
      · it took the shapely area of the mask POLYGON rather than counting mask
        pixels. Measured against the pipeline over 14 test images, that
        under-reported by 0.6% at the median and 2.3% at worst, and the error
        grew as the wound shrank — the polygon cuts corners, and the smaller the
        wound the larger the share of it that is boundary. It biased exactly the
        regime the manuscript already identifies as least reliable.

    Returns (area_px, n_masks, union_mask) with union_mask at (height, width).
    """
    if result.masks is None or len(result.masks) == 0:
        return 0, 0, None
    md = result.masks.data.cpu().numpy() > 0.5
    union = np.any(md, axis=0)
    if union.shape != (height, width):
        import cv2
        union = cv2.resize(
            union.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return int(union.sum()), int(md.shape[0]), union


def resize_image(image):
    return image.resize((640, 640))


def upload_to_drive(drive, image_bytes, filename, folder_id):
    media = MediaIoBaseUpload(image_bytes, mimetype="image/png")
    drive.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media,
        fields="id",
    ).execute()


def find_or_create_folder(drive, folder_name, parent=None):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent:
        query += f" and '{parent}' in parents"

    results = drive.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
    ).execute()

    folders = results.get("files", [])
    if folders:
        return folders[0]["id"]

    file_metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent:
        file_metadata["parents"] = [parent]

    file = drive.files().create(body=file_metadata, fields="id").execute()
    return file.get("id")


# The overlay is drawn at the size the former matplotlib figure produced (a 6 x 6 in
# canvas at 300 dpi, whose axes box is 1395 x 1386 px) with the same 2 pt red line,
# 8 px at 300 dpi. Drawn directly with OpenCV it takes ~0.06 s instead of the ~1.2 s
# matplotlib spent building the figure, and the two differ only at the edge of the line.
OVERLAY_BOX = (1395, 1386)
OVERLAY_LINE_PX = 8


def draw_overlay(image_np, contornos):
    h, w = image_np.shape[:2]
    scale = min(OVERLAY_BOX[0] / w, OVERLAY_BOX[1] / h)
    size = (max(1, int(w * scale + 1e-6)), max(1, int(h * scale + 1e-6)))
    canvas = cv2.resize(image_np, size, interpolation=cv2.INTER_AREA)
    pts = [np.round(np.column_stack([np.asarray(xs) * scale, np.asarray(ys) * scale]))
           .astype(np.int32) for xs, ys in contornos]
    if pts:
        cv2.polylines(canvas, pts, isClosed=True, color=(255, 0, 0),
                      thickness=OVERLAY_LINE_PX, lineType=cv2.LINE_AA)
    _, png = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return BytesIO(png.tobytes())


# Shown as PNG. On "auto", Streamlit may send an image with no alpha channel as JPEG
# (the original is served that way), which would blur the edge of the red line; the
# matplotlib overlay this replaces was an RGBA PNG and reached the page losslessly.
OVERLAY_FORMAT = "PNG"


# Streamlit re-runs this file on every interaction: picking a feedback rating, typing a
# comment, opening a panel. Each of those used to segment the image again, and in folder
# mode every image of the folder. What segmentation produces depends only on the image,
# the model and the confidence, so it is kept under those three. The calibration only
# rescales the area and the file name only labels it; both stay outside, so changing the
# scale costs nothing, and since the cache is shared by every session, two people who
# upload the same bytes under different names each see their own name. Entries expire
# after an hour, and at most 100 are kept across all sessions, the least recently used
# going first: a folder of more than 100 images, or several large folders at once, is
# segmented again on every run, as every folder was before. A failed prediction raises,
# so it is never cached and the next run tries again.
DISPLAY_WIDTH = 1460


def display_copy(image):
    """The copy of the original that the page shows, made once per image.

    Handed the decoded image, Streamlit encoded it as a full-resolution JPEG, re-opened
    it, reduced it to 1460 px wide and encoded it again, on every run and for every image
    on the page. These are the same steps, done once. On a 2452 x 2056 test image the
    bytes are identical to those the live app served before this change, and Streamlit
    passes a JPEG of this width through as it is.
    """
    full = BytesIO()
    image.save(full, format="JPEG", quality=100)
    if image.width <= DISPLAY_WIDTH:
        return full.getvalue()
    shown = Image.open(full)
    shown = shown.resize(
        (DISPLAY_WIDTH, int(1.0 * shown.height * DISPLAY_WIDTH / shown.width)),
        resample=Image.Resampling.BILINEAR,
    )
    out = BytesIO()
    shown.save(out, format="JPEG", quality=90)
    return out.getvalue()


@st.cache_data(max_entries=100, ttl=3600, show_spinner=False)
def segment(image_bytes, model_filename, confidence):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    width_px, height_px = image.size
    seg = {"width": width_px, "height": height_px, "display": display_copy(image),
           "area_px2": None, "n_regions": 0, "overlay_png": None, "contour_csv": None}

    # Ultralytics takes a colour numpy array as BGR, OpenCV's order, and flips it to RGB for
    # the model (data/loaders.py, engine/predictor.py); the evaluation pipeline passes the
    # file path, which OpenCV reads as BGR. This app used to pass the RGB array, so the model
    # saw red and blue swapped. Grey images (R = G = B) were unaffected, which is why the
    # comparison with the pipeline agreed; on the one colour image of the test set the app
    # found no wound where the pipeline measures 8.1% of the field with M and 14.8% with S.
    bgr = np.ascontiguousarray(image_np[..., ::-1])
    results = predict(load_model(model_filename), bgr, confidence / 100.0)
    if not results:
        return seg

    result = results[0]
    area_px2, n_masks, union = mask_area_px(result, height_px, width_px)
    if area_px2 == 0 or union is None:
        return seg

    # Overlay: every contour is drawn, so what the user sees is what was
    # counted. The contours are for display only; the number above comes
    # from the mask raster.
    contornos = []
    if result.masks is not None and result.masks.xyn is not None:
        for c in result.masks.xyn:
            if c is not None and len(c) >= 3:
                contornos.append(
                    [[float(x * width_px) for x, _ in c] + [float(c[0][0] * width_px)],
                     [float(y * height_px) for _, y in c] + [float(c[0][1] * height_px)]]
                )

    # Contour coordinates as data, in place of the polygon picture that used
    # to be shown. A CSV of vertices can be re-plotted or re-measured; a PNG
    # of the same polygon cannot.
    linhas = ["region,vertex,x_px,y_px"]
    for i, (xs, ys) in enumerate(contornos, start=1):
        for j, (x, y) in enumerate(zip(xs[:-1], ys[:-1]), start=1):
            linhas.append(f"{i},{j},{x:.2f},{y:.2f}")

    return {**seg, "area_px2": area_px2, "n_regions": n_masks,
            "overlay_png": draw_overlay(image_np, contornos).getvalue(),
            "contour_csv": "\n".join(linhas).encode("utf-8")}


def process_image(uploaded_file, model_filename, model_confidence, fov_um=None,
                  pixel_size_um=None):
    image_bytes = uploaded_file.getvalue()
    result = {
        "image": uploaded_file.name.replace(" ", "_"),
        "id": hashlib.md5(image_bytes).hexdigest()[:12],
        "error": None,
        "area_px2": None,
        "area_um2": None,
        "area_field_pct": None,
        "n_regions": 0,
        "no_segmentation": False,
        "Exibir": None,
        "Segmentada": None,
        "Contorno": None,
        "Upload": image_bytes,
    }
    try:
        seg = segment(image_bytes, model_filename, model_confidence)
    except Exception as e:
        # The traceback goes to the Space's log. The page says that the image failed,
        # which is not the same as an image with no wound; the previous version
        # returned nothing here, and in folder mode the image dropped out of the table.
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["Exibir"] = seg["display"]
    if seg["overlay_png"] is None:
        result["no_segmentation"] = True
        return result

    effective_pixel_size_um = None
    if pixel_size_um is not None and pixel_size_um > 0:
        effective_pixel_size_um = pixel_size_um
    elif fov_um is not None and fov_um > 0:
        effective_pixel_size_um = fov_um / float(seg["width"])

    area_px2 = seg["area_px2"]
    result.update({
        "area_px2": area_px2,
        "area_um2": (None if effective_pixel_size_um is None
                     else area_px2 * (effective_pixel_size_um ** 2)),
        "area_field_pct": 100.0 * area_px2 / float(seg["width"] * seg["height"]),
        "n_regions": seg["n_regions"],
        "Segmentada": BytesIO(seg["overlay_png"]),
        "Contorno": BytesIO(seg["contour_csv"]),
    })
    return result


def save_feedback(result, avaliacao, observacao, selected_model_label):
    image_name = result["image"]
    image_base_name = image_name.rsplit(".", 1)[0]
    storage_key = MODEL_STORAGE_KEY[selected_model_label]

    drive, sheet = _google()
    sheet.append_row([image_name, avaliacao, observacao, storage_key, APP_VERSION])

    if avaliacao in ["Acceptable", "Bad", "No segmentation"]:
        sufixo = (
            "aceitavel" if avaliacao == "Acceptable"
            else "ruim" if avaliacao == "Bad"
            else "sem_segmentacao"
        )

        parent_folder = find_or_create_folder(drive, "Feedback Segmentacoes")
        model_folder = find_or_create_folder(drive, storage_key, parent_folder)
        subfolder = find_or_create_folder(drive, image_base_name, model_folder)

        # From the uploaded file: "Exibir" holds the reduced copy the page shows.
        resized_original = resize_image(Image.open(BytesIO(result["Upload"])).convert("RGB"))
        buf = BytesIO()
        resized_original.save(buf, format="PNG")
        buf.seek(0)
        upload_to_drive(drive, buf, f"original_{storage_key}_v{APP_VERSION}_{sufixo}.png",
                        subfolder)

        if avaliacao != "No segmentation" and result.get("Segmentada"):
            resized_segmented = resize_image(Image.open(BytesIO(result["Segmentada"].getvalue())))
            buf = BytesIO()
            resized_segmented.save(buf, format="PNG")
            buf.seek(0)
            upload_to_drive(
                drive, buf, f"segmentada_{storage_key}_v{APP_VERSION}_{sufixo}.png", subfolder
            )


def render_metrics(result):
    area_px2 = result["area_px2"]
    area_um2 = result["area_um2"]
    area_pct = result["area_field_pct"]

    st.markdown("**Segmented area**")
    if area_px2 is not None:
        st.markdown(f"- {area_px2:,.0f} px²")
    if area_um2 is not None:
        st.markdown(f"- {area_um2:,.2f} µm²")
    if area_pct is not None:
        st.markdown(f"- {area_pct:.2f}% of the field")
        if area_pct < 5.0:
            st.warning(
                "The remaining gap is below 5% of the field. In the validation "
                "study, agreement with manual measurement degrades in this "
                "regime; treat this value as the least reliable point of a series."
            )
    if result.get("n_regions", 0) > 1:
        st.caption(f"{result['n_regions']} disconnected regions; the area is their union.")


def render_feedback_block(result, selected_model_label, prefix_key=""):
    st.markdown("#### Segmentation quality feedback")
    st.caption("Saving feedback sends this image, its file name, your rating and "
               "your comment to the authors, who keep them to improve the model. "
               "Do not submit images you are not allowed to share.")

    # Keyed by the image's content (and in folder mode its position), not by its name: two
    # files with the same name in one folder made Streamlit raise a duplicate-key error and
    # stop the page, and a new upload under an old name inherited the old rating and comment.
    avaliacao = st.radio(
        "Segmentation quality assessment:",
        ["Great", "Acceptable", "Bad", "No segmentation"],
        horizontal=True,
        key=f"{prefix_key}radio_{result['id']}",
    )
    observacao = st.text_area(
        "Observations (optional):",
        key=f"{prefix_key}obs_{result['id']}",
    )
    if st.button("Save feedback", key=f"{prefix_key}btn_{result['id']}"):
        save_feedback(result, avaliacao, observacao, selected_model_label)
        st.success("Feedback saved successfully.")


# =========================
# Layout / UI
# =========================
st.title("Scratch Assay Segmentation Tool")
st.caption(f"Platform version {APP_VERSION}")

st.markdown("---")

st.markdown("### Input")
col_input_1, col_input_2 = st.columns([2, 1])

with col_input_1:
    upload_option = st.radio("Choose upload type:", ["Single image", "Image folder"], horizontal=True)

with col_input_2:
    selected_model_label = st.selectbox("Segmentation model", list(MODEL_OPTIONS.keys()), index=0)

model_filename = MODEL_OPTIONS[selected_model_label]
load_model(model_filename)  # loaded when the model is chosen, not during the first upload
st.caption(MODEL_INFO[MODEL_STORAGE_KEY[selected_model_label]])
st.caption(HOSTED_NOTE)

_pad = _padding_efetivo()
if _pad != PADDING_FILL:
    st.error(
        f"Letterbox padding is {_pad}, not the {PADDING_FILL} the models were "
        "trained with. Areas would differ from the published pipeline. "
        "Do not use these results."
    )
    st.stop()

with st.expander("⚙️ Advanced Settings", expanded=False):
    model_confidence = st.slider("Model confidence (%)", 20, 100, 80)
    st.caption(
        "80% is the operating point at which the reported precision and recall "
        "were measured."
    )
    st.markdown(
        "### Physical calibration (optional)\n"
        "Provide the physical scale for conversion from pixel area to physical units (µm²). "
        "If left empty, results will be reported only in pixels²."
    )
    c1, c2 = st.columns(2)
    fov_um = c1.number_input(
        "Field of view width (µm)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Physical width of the image field, in micrometers.",
    )
    pixel_size_um = c2.number_input(
        "Pixel size (µm / pixel)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="If provided, this overrides the FOV-based calibration.",
    )

results = []

with st.sidebar:
    st.markdown("## Info")
    with st.expander("About / Citation", expanded=False):
        st.markdown(
            f"""
This tool was developed by the **Medical Physics Laboratory** of the Department of
**Biophysics and Pharmacology – IBB, UNESP**.
**FAPESP Process:** 2024/01849-4.
**Coordination:** Prof. Allan Alves.
**Development:** Nycolas Mariotto.

The two configurations offered here are those of the companion manuscript
(Mariotto et al., *Cytometry Part A*). They differ **only in model scale**:
initialisation, padding and training schedule are identical.

- **M**: default. {MODEL_INFO['M']}
- **S**: fast mode. {MODEL_INFO['S']}

Neither is the more accurate: across the five configurations evaluated, mean
mAP@50 spans 93.3–94.0% and no pairwise difference is distinguishable. The choice
is latency against recall.

**What this tool is for.** Comparing conditions across many wells. Agreement with
a careful manual measurement has 95% limits of agreement of about ±0.3 in closure
fraction, so a single automated measurement is **not** a substitute for a single
manual one.

Weights: [{HF_MODEL_REPO}](https://huggingface.co/{HF_MODEL_REPO}) — AGPL-3.0,
derived from Ultralytics YOLO11.
Companion archive: Zenodo DOI [10.5281/zenodo.20298129](https://doi.org/10.5281/zenodo.20298129).
            """
        )


# =========================
# Single image
# =========================
if upload_option == "Single image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tiff"])
    if uploaded_file:
        st.markdown("---")
        st.markdown("### Result")

        result = process_image(
            uploaded_file,
            model_filename=model_filename,
            model_confidence=model_confidence,
            fov_um=fov_um,
            pixel_size_um=pixel_size_um,
        )

        results.append(result)
        st.markdown(f"#### {result['image']}")

        if result["error"]:
            st.error(f"This image could not be processed ({result['error']}). "
                     "No area was measured.")
        elif result["no_segmentation"]:
            st.image(result["Exibir"], caption="Original", use_container_width=True)
            st.warning("No segmentation was detected for this image.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(result["Exibir"], caption="Original", use_container_width=True)
            with col2:
                st.image(result["Segmentada"], caption="Segmentation", use_container_width=True,
                         output_format=OVERLAY_FORMAT)

            render_metrics(result)

            st.markdown("### Export")
            e1, e2 = st.columns(2)
            with e1:
                st.download_button(
                    "Download segmented overlay (PNG)",
                    data=result["Segmentada"],
                    file_name=f"segmented_{result['image']}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with e2:
                st.download_button(
                    "Download contour coordinates (CSV)",
                    data=result["Contorno"],
                    file_name=f"contour_{result['image']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        if not result["error"]:
            st.markdown("---")
            render_feedback_block(result, selected_model_label, prefix_key="single_")


# =========================
# Folder
# =========================
elif upload_option == "Image folder":
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown("---")
        st.markdown("### Processing")

        processed = [
            process_image(
                f,
                model_filename=model_filename,
                model_confidence=model_confidence,
                fov_um=fov_um,
                pixel_size_um=pixel_size_um,
            )
            for f in uploaded_files
        ]

        erros = [r for r in processed if r["error"]]
        if erros:
            st.error(
                f"{len(erros)} image(s) could not be processed, and no area was measured "
                "for them:\n\n- " + "\n- ".join(f"{r['image']} ({r['error']})" for r in erros)
            )
        falhas = [f.name for f, r in zip(uploaded_files, processed) if r["no_segmentation"]]
        if falhas:
            st.warning(
                f"{len(falhas)} image(s) with no segmentation detected:\n\n- " + "\n- ".join(falhas)
            )

        zip_images_buffer = BytesIO()
        with zipfile.ZipFile(zip_images_buffer, "w") as zip_file:
            for idx, result in enumerate(processed, start=1):
                results.append(result)
                st.markdown("---")
                st.markdown(f"### Result {idx} · {result['image']}")

                if result["error"]:
                    st.error(f"This image could not be processed ({result['error']}). "
                             "No area was measured.")
                    continue
                if result["no_segmentation"]:
                    st.image(result["Exibir"], caption="Original", use_container_width=True)
                    st.warning("No segmentation was detected for this image.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(result["Exibir"], caption="Original", use_container_width=True)
                    with col2:
                        st.image(result["Segmentada"], caption="Segmentation",
                                 use_container_width=True, output_format=OVERLAY_FORMAT)

                    render_metrics(result)

                    zip_file.writestr(
                        f"segmentada_{result['image']}.png", result["Segmentada"].getvalue()
                    )
                    zip_file.writestr(
                        f"contorno_{result['image']}.csv", result["Contorno"].getvalue()
                    )

                render_feedback_block(result, selected_model_label, prefix_key=f"folder_{idx}_")

        zip_images_buffer.seek(0)

        if results:
            st.markdown("---")
            st.markdown("### Quantitative results")

            df = pd.DataFrame(
                [
                    {
                        "Image": r["image"],
                        "Segmented Area (px²)": (
                            "Error" if r["error"]
                            else f"{r['area_px2']:.0f}"
                            if (not r["no_segmentation"] and r["area_px2"] is not None)
                            else "No Segmentation"
                        ),
                        "Segmented Area (µm²)": (
                            f"{r['area_um2']:.2f}"
                            if (not r["no_segmentation"] and r["area_um2"] is not None)
                            else ""
                        ),
                        "Field (%)": (
                            f"{r['area_field_pct']:.2f}"
                            if (not r["no_segmentation"] and r["area_field_pct"] is not None)
                            else ""
                        ),
                        "Regions": "" if r["error"] else r.get("n_regions", 0),
                    }
                    for r in results
                ]
            )

            st.dataframe(df, use_container_width=True)

            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)

            st.markdown("### Export results")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download table (Excel)",
                    data=excel_buffer,
                    file_name="segmentation_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Download segmented images and contours (ZIP)",
                    data=zip_images_buffer,
                    file_name="segmented_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
