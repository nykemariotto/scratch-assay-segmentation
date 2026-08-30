# -*- coding: utf-8 -*-
"""
stage1/coco_to_yolo_seg.py — step 2.0a (no remapping).

Converts the per-partition COCO to YOLO-seg format and builds the tree
dataset/{images,labels}/{train,val,test}.

CORRECOES em relacao ao rascunho da spec:
  * the label stem comes from file_name (UNIQUE), NOT from extra.name — extra.name
    collides (934 train images -> only 738 distinct stems), which would overwrite
    ~200 labels per partition.
  * a categoria 0 'centers' e a supercategoria vazia do Roboflow; so a id 1
    'center' tem anotacoes. Classe unica -> indice 0.
  * images enter by hardlink (same NTFS volume), without duplicating 3.5 GB.
"""
import json, os, sys

SRC_IMG = (os.environ.get("ROBOFLOW_EXPORT", "<roboflow_export>")
           + r"\Pre Eclampsia.coco-segmentation\train")
ROOT = "dataset"
PARTS = ["train", "val", "test"]

# only the categories that actually have annotations
USED_CAT = {1: 0}          # category_id 1 ('center') -> classe 0
CLASS_NAMES = {0: "wound"}


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return "existente"
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        import shutil
        shutil.copy2(src, dst)
        return "copia"


def convert(part):
    coco = json.load(open(f"coco_partitions/instances_{part}.json", encoding="utf-8"))
    img_dir = os.path.join(ROOT, "images", part)
    lab_dir = os.path.join(ROOT, "labels", part)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    n_lab = n_inst = n_neg = 0
    n_clamp = 0
    modes = {}
    for im in coco["images"]:
        W, H = im["width"], im["height"]
        fn = im["file_name"]
        stem = os.path.splitext(fn)[0]

        # imagem
        src = os.path.join(SRC_IMG, fn)
        if not os.path.exists(src):
            sys.exit(f"ABORTADO: imagem ausente no export base: {src}")
        m = link_or_copy(src, os.path.join(img_dir, fn))
        modes[m] = modes.get(m, 0) + 1

        # label
        lines = []
        for a in by_img.get(im["id"], []):
            cid = a["category_id"]
            if cid not in USED_CAT:
                continue
            idx = USED_CAT[cid]
            for seg in a.get("segmentation", []):
                if len(seg) < 6:
                    continue
                norm = []
                for i, v in enumerate(seg):
                    x = v / (W if i % 2 == 0 else H)
                    if x < 0 or x > 1:
                        n_clamp += 1
                    norm.append(min(1.0, max(0.0, x)))
                lines.append(f"{idx} " + " ".join(f"{c:.6f}" for c in norm))
                n_inst += 1
        with open(os.path.join(lab_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        n_lab += 1
        if not lines:
            n_neg += 1

    print(f"{part:<6} imgs={len(coco['images']):>4}  labels={n_lab:>4}  "
          f"instancias={n_inst:>4}  negativos(vazios)={n_neg:>3}  "
          f"coords_clampadas={n_clamp}  {modes}")
    return len(coco["images"]), n_lab, n_inst, n_neg


if __name__ == "__main__":
    # IDEMPOTENCE (anti-leakage): clears the tree before reconverting. Without it,
    # a re-split that MOVES a field between partitions leaves the stale image+label
    # in the old partition (link_or_copy preserves what exists) AND adds the copy to
    # the new one -> the same physical field in 2 partitions = leakage. The
    # leakage-free partition flow is
    # exactly re-split -> reconvert, so this vector has a live trigger.
    import shutil
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
        print(f"cleared: {ROOT}/ (idempotent reconversion)")

    tot = [0, 0, 0, 0]
    for p in PARTS:
        r = convert(p)
        tot = [a + b for a, b in zip(tot, r)]
    print(f"\nTOTAL  imgs={tot[0]}  labels={tot[1]}  instancias={tot[2]}  negativos={tot[3]}")

    # data.yaml
    path_abs = os.path.abspath(ROOT).replace("\\", "/")
    with open("data.yaml", "w", encoding="utf-8") as f:
        f.write(f"path: {path_abs}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
        for i, n in CLASS_NAMES.items():
            f.write(f"  {i}: {n}\n")
    print(f"\nSalvo: data.yaml (path={path_abs})")
