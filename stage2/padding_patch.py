# -*- coding: utf-8 -*-
"""
stage2/padding_patch.py — takes EXPLICIT control of Ultralytics' padding fill
colour, for the black-edge vs white-edge ablation.

Why it is needed: Ultralytics 8.4.x hard-codes 114 (grey) in three independent
places, none of them exposed in the training config:
  1. Mosaic._mosaic4/_mosaic9 -> np.full(..., 114)          [training]
  2. RandomPerspective        -> borderValue=(114,114,114)  [training]
  3. LetterBox                -> padding_value=114          [val/inference]

Patching LetterBox alone would contaminate the ablation: training would stay
grey. This module intercepts all three, auditably.

Usage:
    import padding_patch; padding_patch.apply("black")   # 0
    import padding_patch; padding_patch.apply("white")   # 255
    import padding_patch; padding_patch.apply("gray")    # 114 (upstream default)
"""
import os

import numpy as np

VALUES = {"black": 0, "white": 255, "gray": 114}
ENV = "WHST_PADDING_MODE"
_APPLIED = None


def worker_init(worker_id: int) -> None:
    """worker_init_fn that REAPPLIES the patch inside every worker process.

    THIS IS THE HEART OF THE FIX — without it the ablation does not happen.

    On Windows the DataLoader uses SPAWN: each worker is a fresh process and
    inherits no monkeypatch from the parent. Diagnosed on 2026-07-27: the workers
    reported `np.full(114) -> 114` while the parent reported 255, and 100 epochs
    of black and of white training came out with BIT-IDENTICAL loss.

    LetterBox escaped the problem by accident — padding_value becomes an instance
    attribute in __init__ (executed in the parent) and travels in the dataset
    pickle. That is why validation differed and training did not: exactly the
    pattern observed.

    This function is passed as `worker_init_fn`. It is pickled BY REFERENCE
    (`padding_patch.worker_init`), which forces the worker to import this module —
    and the patch is then applied in there, before the first batch. The mode comes
    from the environment variable, which child processes inherit.
    """
    mode = os.environ.get(ENV)
    if mode in VALUES:
        apply(mode, _in_worker=True)
    # preserve Ultralytics' original behaviour
    try:
        import random

        import torch
        s = torch.initial_seed() % 2 ** 32
        np.random.seed(s)
        random.seed(s)
    except Exception:
        pass


def apply(mode: str, _in_worker: bool = False):
    """Apply the padding value globally. Returns the numeric value used."""
    global _APPLIED
    if mode not in VALUES:
        raise ValueError(f"invalid mode: {mode} (use {list(VALUES)})")
    fill = VALUES[mode]
    os.environ[ENV] = mode          # inherited by child processes (spawn)

    import ultralytics.data.augment as A

    # ---- 1) LetterBox: force the padding_value default ----
    if not hasattr(A.LetterBox, "_orig_init"):
        A.LetterBox._orig_init = A.LetterBox.__init__

    def _lb_init(self, *args, **kw):
        kw["padding_value"] = fill
        return A.LetterBox._orig_init(self, *args, **kw)

    A.LetterBox.__init__ = _lb_init

    # ---- 2) RandomPerspective: the warp borderValue ----
    cv2 = A.cv2
    if not hasattr(cv2, "_orig_warpAffine"):
        cv2._orig_warpAffine = cv2.warpAffine
        cv2._orig_warpPerspective = cv2.warpPerspective

    # guard BY VALUE (==114), symmetric with _full: intercepts only the grey image
    # fill in RandomPerspective. Does not touch the borderValue=255 used by
    # apply_semantic for the semantic mask (dormant in this pipeline, but shielded).
    def _wa(src, M, dsize, *a, **kw):
        if kw.get("borderValue") == (114, 114, 114):
            kw["borderValue"] = (fill,) * 3
        return cv2._orig_warpAffine(src, M, dsize, *a, **kw)

    def _wp(src, M, dsize, *a, **kw):
        if kw.get("borderValue") == (114, 114, 114):
            kw["borderValue"] = (fill,) * 3
        return cv2._orig_warpPerspective(src, M, dsize, *a, **kw)

    cv2.warpAffine = _wa
    cv2.warpPerspective = _wp

    # ---- 3) Mosaic: the np.full(..., 114) canvas ----
    if not hasattr(A.np, "_orig_full"):
        A.np._orig_full = np.full

    def _full(shape, fill_value, *a, **kw):
        # intercept only the mosaic grey canvas; leave every other np.full alone
        if isinstance(fill_value, (int, np.integer)) and int(fill_value) == 114:
            fill_value = fill
        return A.np._orig_full(shape, fill_value, *a, **kw)

    A.np.full = _full

    # ---- 4) worker_init_fn: carry the patch into the workers ----
    # Without this, items 2 and 3 do not exist in the process that builds the batch.
    if not _in_worker:
        import ultralytics.data.build as B
        if getattr(B, "_padding_worker_init", None) is not worker_init:
            B.seed_worker = worker_init
            B._padding_worker_init = worker_init

    _APPLIED = mode
    return fill


def applied():
    return _APPLIED
