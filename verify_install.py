# -*- coding: utf-8 -*-
"""
verify_install.py — checks that the environment is ready, so that the install
instructions are verifiable rather than merely described.

Checks, without downloading anything:
  - Python version
  - every dependency in requirements.txt: installed? version matches the pin?
  - does PyTorch see a GPU? (informational: only training needs one)
  - example set present and intact (MD5 from the MANIFEST)
  - full dataset present? (optional; only for training)

Exits 0 if inference and analysis can run, 1 if something essential is missing.
"""
import csv, hashlib, importlib.metadata as md, os, re, sys

ess_ok = True


def diz(ok, rot, det=""):
    print(f"  {'OK  ' if ok else 'MISSING'} {rot:<42}{det}")


print("=== Python ===")
v = sys.version_info
py_ok = (v.major, v.minor) >= (3, 11)
diz(py_ok, "version >= 3.11", f"{sys.version.split()[0]}")
ess_ok &= py_ok

print("\n=== dependencies (requirements.txt) ===")
req = "requirements.txt"
if not os.path.isfile(req):
    diz(False, "requirements.txt present")
    ess_ok = False
else:
    for ln in open(req, encoding="utf-8"):
        ln = ln.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(.+)$", ln)
        if not m:
            continue
        pkg, want = m.group(1), m.group(2)
        try:
            got = md.version(pkg)
        except Exception:
            diz(False, pkg, "not installed")
            ess_ok = False
            continue
        # torch carries a build suffix (2.6.0+cu124); compare the base part only
        base = got.split("+")[0]
        diz(True, pkg, f"{got}" + ("" if base == want else f"   [pinned: {want}]"))

print("\n=== GPU (needed only for training) ===")
try:
    import torch
    cuda = torch.cuda.is_available()
    diz(cuda, "CUDA available",
        torch.cuda.get_device_name(0) if cuda else "CPU only — inference and analysis still work")
except Exception as e:
    diz(False, "import torch", str(e)[:50])

print("\n=== example set ===")
manp = os.path.join("examples", "MANIFEST.csv")
if not os.path.isfile(manp):
    diz(False, "examples/MANIFEST.csv", "run: python stage1/prepare_examples.py")
    ess_ok = False
else:
    man = list(csv.DictReader(open(manp, encoding="utf-8-sig")))
    bad = 0
    for r in man:
        p = os.path.join("examples", "images", r["arquivo"])
        if not os.path.exists(p):
            bad += 1
            continue
        if hashlib.md5(open(p, "rb").read()).hexdigest() != r["md5"]:
            bad += 1
    diz(bad == 0, f"{len(man)} example images",
        "intact (MD5 matches)" if bad == 0 else f"{bad} missing or altered")
    ess_ok &= bad == 0

print("\n=== full dataset (optional; only for training) ===")
d = os.path.join("dataset", "images")
if os.path.isdir(d):
    n = {p: len(os.listdir(os.path.join(d, p)))
         for p in ("train", "val", "test") if os.path.isdir(os.path.join(d, p))}
    diz(True, "dataset/images", str(n))
else:
    diz(False, "dataset/images", "download from Zenodo (needed only for training)")

print("\n=== trained weights (optional; for inference) ===")
# separate THIS PROJECT's weights from the Ultralytics pretrained checkpoints
# (yolo11s-seg.pt and friends are COCO, and do not segment a wound)
BASE_ULTRA = re.compile(r"^yolo\d+[nsmlx]?-?(seg|cls|pose|obb)?\.pt$", re.I)
proj, base = [], []
for root in ("models", "."):
    if os.path.isdir(root):
        for f in sorted(os.listdir(root)):
            if f.endswith(".pt"):
                (base if BASE_ULTRA.match(f) else proj).append(f)
diz(bool(proj), "project weights (.pt)", ", ".join(proj[:3]) if proj else
    "download from Zenodo (DOI 10.5281/zenodo.20298129)")
if base:
    print(f"       (ignored: {', '.join(base[:4])} — Ultralytics COCO checkpoints)")

print("\n" + ("READY for inference and analysis." if ess_ok
              else "ESSENTIAL items missing — see above."))
sys.exit(0 if ess_ok else 1)
