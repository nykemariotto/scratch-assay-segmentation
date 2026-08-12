# -*- coding: utf-8 -*-
"""
stage4/integrate_recovered_baselines.py — ETAPA 2 de 2: integra os t0 recuperados ao
pipeline, depois de medidos pelo stage4/whst_batch.ijm em whst_output_novos/.

What it does:
  1. moves image, ROI, mask, polygon and overlay to the official directories;
  2. acrescenta linha em whst_input/correspondencia.csv, data/whst_pass1_qc.csv e
     data/visual_triage.csv (empty category: not yet triaged — validity will be
     decided during the manual correction itself, as was done with the 4 cases
     'ruidosa_recuperavel');
  3. produces stage4/baseline_worklist.csv for the manual correction (pass 5).

A .bak_prebase backup of every altered CSV. Atomic writes.
It overwrites nothing: it aborts if the destination already has the file.
"""
import csv, os, shutil, sys
from collections import Counter

STAGE_IN = "whst_input_novos"
STAGE_OUT = "whst_output_novos"
MANIF = "stage4/baselines_recuperados.csv"
CORR = "whst_input/correspondencia.csv"
QC = "data/whst_pass1_qc.csv"
HUM = "data/visual_triage.csv"
WL = "stage4/baseline_worklist.csv"

for p in (MANIF, STAGE_OUT):
    if not os.path.exists(p):
        sys.exit(f"could not find {p} — run stage4/prepare_recovered_baselines.py and the "
                 f"stage4/whst_batch.ijm sobre {STAGE_IN}/ antes.")

man = list(csv.DictReader(open(MANIF, encoding="utf-8-sig")))
res_p = os.path.join(STAGE_OUT, "data/whst_batch_results.csv")
if not os.path.isfile(res_p):
    sys.exit(f"could not find {res_p} (output of stage4/whst_batch.ijm)")
res = {r["filename"]: r for r in csv.DictReader(open(res_p, encoding="utf-8-sig"))}
print(f"manifesto: {len(man)} | medidas do WHST: {len(res)}")


def base(f):
    for e in (".tiff", ".tif"):
        if f.lower().endswith(e):
            return f[: -len(e)]
    return os.path.splitext(f)[0]


def mv(src, dst):
    if not os.path.exists(src):
        return False
    if os.path.exists(dst):
        sys.exit(f"ABORTADO: destino ja existe: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return True


novos = []
for m in man:
    fn = m["whst_input_file"]
    r = res.get(fn)
    if not r:
        print(f"  [SKIP] no WHST measurement: {fn[:56]}"); continue
    if r.get("status") != "ok":
        print(f"  [AVISO] status={r.get('status')} em {fn[:48]} — area indisponivel")
    b = base(fn)
    mv(os.path.join(STAGE_IN, fn), os.path.join("whst_input", fn))
    for sub, suf in (("rois", ".roi"), ("masks", "_mask.png"),
                     ("polygons", "_polygon.csv"), ("overlays", "_overlay.jpg")):
        mv(os.path.join(STAGE_OUT, sub, b + suf),
           os.path.join("whst_output", sub, b + suf))
    m["area_pct"] = r.get("area_pct", "")
    novos.append(m)

if not novos:
    sys.exit("nada integrado.")


def append_csv(path, novas_linhas, chave):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    campos = list(rows[0].keys())
    exist = {r[chave] for r in rows}
    add = [r for r in novas_linhas if r[chave] not in exist]
    if not add:
        print(f"  {path}: nada novo"); return 0
    if not os.path.isfile(path + ".bak_prebase"):
        shutil.copy2(path, path + ".bak_prebase")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(rows)
        for a in add:
            w.writerow({c: a.get(c, "") for c in campos})
    os.replace(tmp, path)
    print(f"  {path}: +{len(add)} (agora {len(rows)+len(add)})")
    return len(add)


print("\nCSVs:")
append_csv(CORR, [{"test_image": "(baseline_recuperado_posteriori)",
                   "cell_line": m["cell_line"], "group_key": m["analysis_unit"],
                   "timepoint_h": 0, "raw_file_original": m["raw_file_original"],
                   "raw_md5": m["raw_md5"], "whst_input_file": m["whst_input_file"],
                   "is_duplicate_of": "", "is_baseline": "yes",
                   "baseline_note": "recuperado do banco cru a posteriori"}
                  for m in novos], "whst_input_file")
append_csv(QC, [{"whst_input_file": m["whst_input_file"], "cell_line": m["cell_line"],
                 "timepoint_h": 0, "analysis_unit": m["analysis_unit"],
                 "series_key": m["series_key"], "campo": m["campo"],
                 "grupo_split": "", "is_baseline": "yes", "area_pct": m["area_pct"],
                 "categoria": "NAO_TRIADA", "needs_correction": 1,
                 "raw_file": m["raw_file_original"], "test_image": ""}
                for m in novos], "whst_input_file")
append_csv(HUM, [{"whst_input_file": m["whst_input_file"], "cell_line": m["cell_line"],
                  "group_key": m["analysis_unit"], "timepoint_h": 0,
                  "campo": m["campo"], "raw_file_original": m["raw_file_original"],
                  "raw_md5": m["raw_md5"], "categoria": "NAO_TRIADA", "subtipo": "",
                  "origem_categoria": "baseline_recuperado_posteriori",
                  "na_lista_correcao": "sim", "na_lista_revisao": "nao"}
                 for m in novos], "whst_input_file")

rows = []
for i, m in enumerate(sorted(novos, key=lambda x: (x["analysis_unit"], x["campo"])), 1):
    rows.append({"ordem": i, "whst_input_file": m["whst_input_file"],
                 "cell_line": m["cell_line"], "analysis_unit": m["analysis_unit"],
                 "timepoint_h": 0, "campo": m["campo"], "eh_baseline": "SIM",
                 "tarefa": "corrigir contorno (baseline recuperado; se invalida, "
                           "Select None -> IMAGEM INVALIDA)",
                 "roi_auto": base(m["whst_input_file"]) + ".roi"})
with open(WL, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"\nintegrados: {len(novos)}  |  worklist: {WL} ({len(rows)} frames)")
print("area_pct medida:", {m["whst_input_file"][:10]: m["area_pct"] for m in novos})
print("\nNEXT: Fiji -> stage4/whst_manual_correction.ijm -> pass '5 - baselines recuperados'")
print("Depois:  python stage4/apply_corrections.py && python stage4/whst_series_analysis.py "
      "&& python stage4/final_closure_table.py")
for d in (STAGE_IN, STAGE_OUT):
    rest = sum(len(fs) for _, _, fs in os.walk(d)) if os.path.isdir(d) else 0
    print(f"  {d}/: {rest} arquivo(s) restante(s)")
