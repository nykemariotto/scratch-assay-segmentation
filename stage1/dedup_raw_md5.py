# -*- coding: utf-8 -*-
"""
stage1/dedup_raw_md5.py — deduplication at the level of the RAW TIFF (bank A), not
of the Roboflow re-encoded PNGs. For each group of the same normalised identity
(well/snap, timepoint, campo) com >1 arquivo fisico, mostra MD5, dimensoes e
pixel-by-pixel comparison (mean absolute difference + correlation). Groups the test
images by the MD5 of the raw file to find real measurement duplicates.
"""
import csv, os, re, hashlib
from collections import defaultdict
import numpy as np
from PIL import Image

BA = os.environ.get("BANCO_A", "<banco_a>")
test = [x for x in csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8"))
        if x["partition"] == "test"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[\s_]+", " ", s.strip().lower())).strip()


def raw_path(x):
    a = x["arquivo_a"].strip()
    return os.path.join(BA, x["pasta_a"].replace("/", os.sep), a) if a else None


# ---- MD5 of each raw TIFF referenced by the test set (216) ----
print("MD5 of the test set raw TIFFs (216 files, ~7.6MB each)...")
recs = []  # dict por imagem do test com raw
md5cache = {}
for x in test:
    p = raw_path(x)
    if p and os.path.exists(p):
        if p not in md5cache:
            md5cache[p] = md5(p)
        recs.append({"arquivo_b": x["arquivo_b"], "group_key": x["group_key"],
                     "timepoint": x["timepoint_h"], "stem": norm(re.sub(r"\.(tiff?|png)$", "", x["arquivo_a"])),
                     "stem_id": x["stem_normalizado"], "path": p, "file": x["arquivo_a"],
                     "md5": md5cache[p]})
print(f"  raw files MD5-hashed: {len(md5cache)}  (test rows with a raw file: {len(recs)})")

# ---- grupos de mesma identidade normalizada (stem_id do dataset) com >1 arquivo fisico ----
by_id = defaultdict(list)
for r in recs:
    by_id[r["stem_id"]].append(r)
collisions = {k: v for k, v in by_id.items() if len({r["path"] for r in v}) > 1}
print(f"\nidentity groups with >1 distinct physical raw TIFF: {len(collisions)}")


def gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32)


print("\n=== EVIDENCE: pairs of the same identity: MD5, dims, pixel ===")
shown = 0
n_identico = n_distinto = 0
for stem_id, rs in sorted(collisions.items()):
    paths = sorted({r["path"] for r in rs})
    # only groups whose names look like a case/suffix collision
    files = [os.path.basename(p) for p in paths]
    md5s = [md5cache[p] for p in paths]
    arrs = [gray(p) for p in paths]
    dims = [a.shape for a in arrs]
    # pairwise comparison (the first against each other)
    same_dim = len(set(dims)) == 1
    if len(set(md5s)) == 1:
        verdict = "MD5 IDENTICO (duplicata real)"
        n_identico += 1
    else:
        # pixel: dif abs media + correlacao (se mesma dim)
        if same_dim:
            diffs = [float(np.mean(np.abs(arrs[0] - a))) for a in arrs[1:]]
            def corr(a, b):
                a = a.ravel(); b = b.ravel()
                a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
                return float(np.dot(a, b) / len(a))
            cors = [corr(arrs[0], a) for a in arrs[1:]]
            verdict = f"DISTINTO (dif_abs_media={max(diffs):.1f}/255, corr_min={min(cors):.3f})"
        else:
            verdict = f"DISTINTO (dimensoes diferentes: {dims})"
        n_distinto += 1
    if shown < 20:
        print(f"  [{stem_id}] {verdict}")
        for f, h, d in zip(files, md5s, dims):
            print(f"       {f:<26} md5={h[:12]}  dims={d[1]}x{d[0]}")
        shown += 1

print(f"\nidentity groups: {n_identico} IDENTICAL (MD5), {n_distinto} DISTINCT")

# ---- duplicatas de MEDICAO: >1 imagem do test com o MESMO md5 de cru ----
by_md5 = defaultdict(list)
for r in recs:
    by_md5[r["md5"]].append(r)
dup_md5 = {h: rs for h, rs in by_md5.items() if len(rs) > 1}
print(f"\n=== MEASUREMENT DUPLICATES (same raw MD5, >1 test image) ===")
print(f"  unique physical raw files in the test set: {len(by_md5)}")
print(f"  MD5s shared by >1 test image: {len(dup_md5)}")
for h, rs in list(dup_md5.items())[:10]:
    print(f"    md5={h[:12]}: {[r['file'] for r in rs]} | grupos={sorted(set(r['group_key'] for r in rs))}")

# ---- orfaos (19) ----
orphans = [x for x in test if not x["arquivo_a"].strip()]
print(f"\n=== ORPHANS (no 1:1 raw source): {len(orphans)} ===")

# ---- salvar resultado ----
import json
json.dump({
    "n_test": len(test), "n_com_cru": len(recs), "n_crus_unicos_md5": len(by_md5),
    "n_grupos_identidade_multiarquivo": len(collisions),
    "n_grupos_identicos_md5": n_identico, "n_grupos_distintos": n_distinto,
    "n_duplicatas_medicao": len(dup_md5), "n_orfaos": len(orphans),
}, open("stage1/dedup_raw_md5.json", "w"), indent=2)
print("\nSalvo: stage1/dedup_raw_md5.json")
