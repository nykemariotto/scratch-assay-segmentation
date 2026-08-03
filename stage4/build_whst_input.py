# -*- coding: utf-8 -*-
"""
stage4/build_whst_input.py — Constroi whst_input/ com os TIFFs crus FISICOS UNICOS do
test set (dedup por MD5 do cru), com CSV de correspondencia, corrige o
data/test_set_dedup.csv (now based on the MD5 of the raw TIFF) and lists the 19 orphans.

Copies (not a hardlink: the raw is on one volume and the destination on another).
Medicao WHST/inferencia AI rodam nas CRUAS (2452x2056, FOV completo); as metricas
de segmentacao (mAP) seguem nas PNGs anotadas.
"""
import csv, os, re, hashlib, shutil
from collections import defaultdict

BA = os.environ.get("BANCO_A", "<banco_a>")
OUT = "whst_input"
os.makedirs(OUT, exist_ok=True)

test = [x for x in csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8"))
        if x["partition"] == "test"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe(s):
    return re.sub(r"[^A-Za-z0-9+.-]+", "_", s).strip("_")


def raw_path(x):
    a = x["arquivo_a"].strip()
    return os.path.join(BA, x["pasta_a"].replace("/", os.sep), a) if a else None


with_raw = [x for x in test if x["arquivo_a"].strip()]
orphans = [x for x in test if not x["arquivo_a"].strip()]

# ---- MD5 of each raw file; copied once per unique MD5 ----
print(f"Processing {len(with_raw)} images with a raw file (MD5 + copy)...")
md5_to_dest = {}       # md5 -> nome no whst_input
md5_to_first = {}      # md5 -> (test row, raw path)
corr_rows = []
copied = 0
for x in with_raw:
    p = raw_path(x)
    h = md5(p)
    cell = "SKOV-3" if x["linha_celular"] == "SKOV" else x["linha_celular"]
    if h not in md5_to_dest:
        # prefixo MD5 garante nome UNICO em FS case-insensitive (Windows):
        # without it, '..A1_12HR_1' and '..A1_12hr_1' (DISTINCT images) would collide.
        dest_name = f"{h[:10]}__{safe(cell)}__{safe(x['group_key'])}__tp{x['timepoint_h']}h__{safe(os.path.splitext(x['arquivo_a'])[0])}.tiff"
        dest = os.path.join(OUT, dest_name)
        if not os.path.exists(dest):
            shutil.copy2(p, dest)
            copied += 1
        md5_to_dest[h] = dest_name
        md5_to_first[h] = x
    corr_rows.append({
        "test_image": x["arquivo_b"],
        "cell_line": cell,
        "group_key": x["group_key"],
        "timepoint_h": int(x["timepoint_h"]),
        "raw_file_original": x["arquivo_a"],
        "raw_md5": h,
        "whst_input_file": md5_to_dest[h],
        "is_duplicate_of": "" if md5_to_first[h]["arquivo_b"] == x["arquivo_b"]
                           else md5_to_first[h]["arquivo_b"],
    })
    if (len(corr_rows)) % 50 == 0:
        print(f"  {len(corr_rows)}/{len(with_raw)}  (copiados {copied})")

n_unique = len(md5_to_dest)
n_dup = len(with_raw) - n_unique
print(f"\narquivos crus fisicos unicos copiados: {n_unique}")
print(f"test images that are duplicates (MD5 already seen): {n_dup}")

# ASSERCAO: nomes unicos em FS case-insensitive + 1 arquivo por MD5
lower_names = [n.lower() for n in md5_to_dest.values()]
assert len(set(lower_names)) == len(md5_to_dest), "COLISAO de nome case-insensitive!"
n_disk = len([f for f in os.listdir(OUT) if f.lower().endswith(".tiff")])
assert n_disk == n_unique, f"disco={n_disk} != md5 unicos={n_unique} (arquivo perdido!)"
print(f"asserts OK: {n_disk} arquivos no disco == {n_unique} MD5 unicos, 0 colisao de nome")

# ---- correspondencia CSV ----
with open(os.path.join(OUT, "correspondencia.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["test_image", "cell_line", "group_key", "timepoint_h",
                                      "raw_file_original", "raw_md5", "whst_input_file", "is_duplicate_of"])
    w.writeheader()
    w.writerows(sorted(corr_rows, key=lambda r: (r["cell_line"], r["group_key"], r["timepoint_h"])))

# ---- orphans: a separate list, noting whether the group has measured siblings ----
grp_has_raw = defaultdict(int)
for r in corr_rows:
    grp_has_raw[r["group_key"]] += 1
with open(os.path.join(OUT, "orfaos_sem_cru.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test_image", "cell_line", "group_key", "timepoint_h", "well", "grupo_tem_outras_medidas"])
    for x in sorted(orphans, key=lambda x: (x["group_key"], int(x["timepoint_h"]))):
        cell = "SKOV-3" if x["linha_celular"] == "SKOV" else x["linha_celular"]
        w.writerow([x["arquivo_b"], cell, x["group_key"], int(x["timepoint_h"]), x["well"],
                    grp_has_raw.get(x["group_key"], 0)])

# ---- data/test_set_dedup.csv corrigido (baseado em MD5 do cru) ----
dedup_rows = []
for r in sorted(corr_rows, key=lambda r: (r["cell_line"], r["group_key"], r["timepoint_h"])):
    dedup_rows.append({
        "filename": r["test_image"], "cell_line": r["cell_line"], "group_key": r["group_key"],
        "timepoint_h": r["timepoint_h"], "raw_md5": r["raw_md5"],
        "physical_status": "duplicata" if r["is_duplicate_of"] else "unico",
        "dup_of": r["is_duplicate_of"], "raw_source": "banco_a",
        "whst_input_file": r["whst_input_file"],
    })
for x in sorted(orphans, key=lambda x: (x["group_key"], int(x["timepoint_h"]))):
    cell = "SKOV-3" if x["linha_celular"] == "SKOV" else x["linha_celular"]
    dedup_rows.append({
        "filename": x["arquivo_b"], "cell_line": cell, "group_key": x["group_key"],
        "timepoint_h": int(x["timepoint_h"]), "raw_md5": "", "physical_status": "sem_fonte_crua",
        "dup_of": "", "raw_source": "nenhuma", "whst_input_file": "",
    })
with open("data/test_set_dedup.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filename", "cell_line", "group_key", "timepoint_h",
                                      "raw_md5", "physical_status", "dup_of", "raw_source", "whst_input_file"])
    w.writeheader(); w.writerows(dedup_rows)

print(f"\n=== RESUMO ===")
print(f"  test total: {len(test)}")
print(f"  com cru: {len(with_raw)}  ->  fisicos unicos a medir: {n_unique}  (duplicatas: {n_dup})")
print(f"  orphans (no raw file, outside the paired analysis): {len(orphans)}")
print(f"  whst_input/: {n_unique} TIFFs + correspondencia.csv + orfaos_sem_cru.csv")
print(f"  data/test_set_dedup.csv atualizado ({len(dedup_rows)} linhas)")
