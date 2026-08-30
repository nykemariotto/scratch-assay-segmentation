# -*- coding: utf-8 -*-
"""
stage1/dedup_analysis.py — part 1: deduplication by PIXEL IDENTITY (not by name).

Motivo: rotulos de well se repetem entre lotes/experimentos; 'A1 0HR 1' de lotes
diferentes pode ser well FISICO diferente (a chave lote x well ja separa). Logo a
redundancia real so se determina comparando o conteudo decodificado.

Produz:
  - data/test_set_dedup.csv : one row per test image, with physical cluster, keep flag,
    motivo da remocao e o representante mantido.
  - a report: physical duplicates that cross partitions (train/val/test).
  - an MD5+pixel check of the type-(b) raw pairs [suffix (N)] in bank A.
"""
import csv, os, hashlib, re
from collections import defaultdict
import numpy as np
from PIL import Image

BANCO_A = os.environ.get("BANCO_A", "<banco_a>")
IMGROOT = os.path.join("dataset", "images")
rows = [r for r in csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8"))
        if r["partition"] != "EXCLUIDA"]


def decode(path):
    im = Image.open(path).convert("L")
    return np.asarray(im)


def pixel_hash(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sig(arr, s=256):
    a = np.asarray(Image.fromarray(arr).resize((s, s), Image.BILINEAR), dtype=np.float32).ravel()
    m, sd = a.mean(), a.std() + 1e-6
    return (a - m) / sd


print(f"Decoding {len(rows)} images (pixel hash + signature)...")
info = {}   # arquivo_b -> dict
sigs = {}
for i, r in enumerate(rows):
    p = os.path.join(IMGROOT, r["partition"], r["arquivo_b"])
    arr = decode(p)
    info[r["arquivo_b"]] = dict(row=r, shape=arr.shape, ph=pixel_hash(arr))
    sigs[r["arquivo_b"]] = sig(arr)
    if (i + 1) % 300 == 0:
        print(f"  {i+1}/{len(rows)}")

# -------- clusters por hash de pixel EXATO (byte-identico apos decodificacao) --------
by_hash = defaultdict(list)
for fb, d in info.items():
    by_hash[d["ph"]].append(fb)
exact_clusters = {h: fbs for h, fbs in by_hash.items() if len(fbs) > 1}
print(f"\nEXACT pixel clusters (>1 image): {len(exact_clusters)}")

# -------- near-dup cross-resolucao dentro do mesmo stem (640 vs nativo do mesmo campo) --------
# only between images with the same normalised stem that do not share an exact hash
by_stem = defaultdict(list)
for fb, d in info.items():
    by_stem[d["row"]["stem_normalizado"]].append(fb)

# union-find to merge exact + near-duplicates
parent = {fb: fb for fb in info}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    parent[find(a)] = find(b)

for fbs in exact_clusters.values():
    for x in fbs[1:]:
        union(x, fbs[0])

near_pairs = []
for stem, fbs in by_stem.items():
    for i in range(len(fbs)):
        for j in range(i + 1, len(fbs)):
            a, b = fbs[i], fbs[j]
            if find(a) == find(b):
                continue
            c = float(np.dot(sigs[a], sigs[b]) / len(sigs[a]))
            if c >= 0.999:
                union(a, b); near_pairs.append((a, b, round(c, 5)))

clusters = defaultdict(list)
for fb in info:
    clusters[find(fb)].append(fb)
multi = {k: v for k, v in clusters.items() if len(v) > 1}
print(f"Physical clusters with >1 image (exact + cross-resolution near-dup): {len(multi)}")
print(f"  cross-resolution near-dup pairs (0.999<=corr<1, different size): {len(near_pairs)}")

# -------- cross-partition: does a physical cluster appear in >1 partition? --------
print("\n=== PHYSICAL DUPLICATES THAT CROSS PARTITIONS ===")
cross = []
for rep, fbs in multi.items():
    parts = {info[fb]["row"]["partition"] for fb in fbs}
    grps = {info[fb]["row"]["group_key"] for fb in fbs}
    if len(parts) > 1:
        cross.append((fbs, parts, grps))
print(f"physical clusters in >1 partition: {len(cross)}")
for fbs, parts, grps in cross[:12]:
    print(f"  {sorted(parts)} | groups={len(grps)} | {[f[:28] for f in fbs]}")
same_grp_cross = sum(1 for _, p, g in cross if len(g) == 1)
print(f"  of those, with the SAME group_key (intra-group redundancy, not leakage): {same_grp_cross}")
print(f"  with a DIFFERENT group_key (investigate!): {len(cross)-same_grp_cross}")

# =========================================================================
# data/test_set_dedup.csv : one row per test image, keep/removed
# =========================================================================
test = [r for r in rows if r["partition"] == "test"]
# each test cluster's representative = the highest resolution, then the name with a raw source
def rank(fb):
    d = info[fb]
    has_raw = 1 if d["row"]["arquivo_a"].strip() else 0
    return (d["shape"][0] * d["shape"][1], has_raw)  # maior + com fonte crua primeiro

out = []
seen_rep = {}
for r in sorted(test, key=lambda r: (r["group_key"], int(r["timepoint_h"]))):
    fb = r["arquivo_b"]
    cl = find(fb)
    members_test = [m for m in clusters[cl] if info[m]["row"]["partition"] == "test"]
    rep = max(members_test, key=rank)
    keep = (fb == rep)
    reason = ""
    if not keep:
        # motivo: mesmo hash exato -> identico; near-dup -> resolucao diferente
        if info[fb]["ph"] == info[rep]["ph"]:
            reason = "pixel_identico"
        else:
            reason = "mesmo_campo_resolucao_diferente"
    out.append({
        "filename": fb,
        "cell_line": "SKOV-3" if r["linha_celular"] == "SKOV" else r["linha_celular"],
        "group_key": r["group_key"],
        "timepoint_h": int(r["timepoint_h"]),
        "raw_path_banco_a": (os.path.join(BANCO_A, r["pasta_a"].replace("/", os.sep), r["arquivo_a"])
                             if r["arquivo_a"].strip() else ""),
        "physical_id": f"phys_{sorted(clusters[cl])[0][:16]}",
        "keep": 1 if keep else 0,
        "dup_reason": reason,
        "dup_of": "" if keep else rep,
        "n_no_cluster_test": len(members_test),
    })

cols = ["filename", "cell_line", "group_key", "timepoint_h", "raw_path_banco_a",
        "physical_id", "keep", "dup_reason", "dup_of", "n_no_cluster_test"]
with open("data/test_set_dedup.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(out)

n_keep = sum(o["keep"] for o in out)
print(f"\n=== data/test_set_dedup.csv ===")
print(f"  rows (test): {len(out)}")
print(f"  UNIQUE physical images to measure (keep=1): {n_keep}")
print(f"  removidas como duplicata: {len(out)-n_keep}")
from collections import Counter
print(f"  motivos: {dict(Counter(o['dup_reason'] for o in out if not o['keep']))}")

# =========================================================================
# MD5 + pixel check of the type-(b) RAW pairs [suffix (N)] in bank A
# =========================================================================
print("\n=== RAW PAIRS OF TYPE (b), suffix (N): MD5 + pixel ===")
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# groups, by (folder, stem without suffix/case), the raw files with and without (N)
raw_by_key = defaultdict(set)
for r in rows:
    a = r["arquivo_a"].strip()
    if not a:
        continue
    base = re.sub(r"\s*\(\d+\)", "", a)              # remove ' (2)'
    key = (r["pasta_a"], re.sub(r"[\s_]+", " ", os.path.splitext(base)[0]).lower())
    raw_by_key[key].add(a)

type_b = {k: sorted(v) for k, v in raw_by_key.items() if len(v) > 1}
print(f"raw groups with >1 physical file (suffix): {len(type_b)}")
checked = idic = idif = 0
examples = []
for (pasta, stem), files in type_b.items():
    paths = [os.path.join(BANCO_A, pasta.replace("/", os.sep), f) for f in files]
    if not all(os.path.exists(p) for p in paths):
        continue
    hs = [md5(p) for p in paths]
    checked += 1
    if len(set(hs)) == 1:
        idic += 1
        if len(examples) < 6: examples.append((stem, files, "MD5 IDENTICO"))
    else:
        # md5 difere -> compara pixel
        arrs = [decode(p) for p in paths]
        same_shape = len({a.shape for a in arrs}) == 1
        pix_id = same_shape and all(np.array_equal(arrs[0], a) for a in arrs[1:])
        if pix_id:
            idic += 1
            if len(examples) < 6: examples.append((stem, files, "pixel identico (md5 difere)"))
        else:
            idif += 1
            if len(examples) < 6: examples.append((stem, files, "CONTEUDO DISTINTO (campos diferentes)"))

print(f"  pairs checked: {checked} | identical content: {idic} | distinct content: {idif}")
for stem, files, verdict in examples:
    print(f"    [{verdict}] {stem}: {files}")

print("\nSalvo: data/test_set_dedup.csv")
