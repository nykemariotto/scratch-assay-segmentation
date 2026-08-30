# -*- coding: utf-8 -*-
"""
stage4/add_baselines_to_whst.py — copies the 7 recovered 0h baselines into whst_input/
(the MAIN folder, so the macro's getFileList() covers them in one run) with
nome prefixado por MD5, e registra-os no correspondencia.csv marcados como
recovered baseline (they do not belong to the annotated test set).
"""
import csv, os, re, hashlib, shutil

OUT = "whst_input"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe(s):
    return re.sub(r"[^A-Za-z0-9+.-]+", "_", s).strip("_")


bl = [r for r in csv.DictReader(open("stage4/baselines_recuperados.csv", encoding="utf-8-sig"))
      if r["status"] == "recuperavel"]
print(f"baselines recuperaveis a copiar: {len(bl)}")

# existing names (case-insensitive) to guarantee uniqueness
existing = {f.lower() for f in os.listdir(OUT) if f.lower().endswith(".tiff")}
existing_md5 = {r["raw_md5"] for r in csv.DictReader(open(os.path.join(OUT, "correspondencia.csv"), encoding="utf-8-sig"))}

new_rows = []
for r in bl:
    src = r["baseline_0h_path"]
    if not os.path.exists(src):
        print(f"  MISSING: {src}")
        continue
    h = md5(src)
    cell = r["cell_line"]
    note = r["confianca"]
    if r.get("arquivo_alt"):
        note += f"; campo/candidato alternativo disponivel: {r['arquivo_alt']}"
    dest_name = f"{h[:10]}__BASELINE__{safe(cell)}__{safe(r['group_key'])}__tp0h__{safe(os.path.splitext(r['arquivo'])[0])}.tiff"
    dest = os.path.join(OUT, dest_name)
    already = h in existing_md5 or dest_name.lower() in existing
    if not os.path.exists(dest):
        shutil.copy2(src, dest)
    print(f"  {r['group_key']:<28} -> {dest_name}  {'(ja no test!)' if already else ''}")
    new_rows.append({
        "test_image": "(baseline_recuperado)", "cell_line": cell,
        "group_key": r["group_key"], "timepoint_h": 0,
        "raw_file_original": r["arquivo"], "raw_md5": h,
        "whst_input_file": dest_name, "is_duplicate_of": "",
        "is_baseline": "yes", "baseline_note": note,
    })

# ---- reescreve correspondencia.csv com colunas is_baseline/baseline_note ----
cpath = os.path.join(OUT, "correspondencia.csv")
old = list(csv.DictReader(open(cpath, encoding="utf-8-sig")))
for r in old:
    r["is_baseline"] = "no"
    r["baseline_note"] = ""
allrows = old + new_rows
cols = ["test_image", "cell_line", "group_key", "timepoint_h", "raw_file_original",
        "raw_md5", "whst_input_file", "is_duplicate_of", "is_baseline", "baseline_note"]
with open(cpath, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(sorted(allrows, key=lambda r: (r["cell_line"], r["group_key"], int(r["timepoint_h"]))))

# ---- verificacao ----
n_tiff = len([f for f in os.listdir(OUT) if f.lower().endswith(".tiff")])
n_ci = len({f.lower() for f in os.listdir(OUT) if f.lower().endswith(".tiff")})
print(f"\n=== whst_input/ after the baselines ===")
print(f"  TIFFs on disk: {n_tiff}  (case-insensitive unique: {n_ci})")
print(f"  correspondencia.csv: {len(allrows)} rows ({len(new_rows)} baselines + {len(old)} test)")
assert n_tiff == n_ci, "COLISAO de nome!"
print(f"  ASSERT ok: no name collision")
