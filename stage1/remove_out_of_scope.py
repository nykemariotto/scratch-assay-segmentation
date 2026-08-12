# -*- coding: utf-8 -*-
"""
stage1/remove_out_of_scope.py — removes from the pipeline the images the operator
marked as 'out of analysis scope' (cross-shaped images, used as an algorithm test
rather than as experimental data).

VERIFIED BEFORE RUNNING: those images do not appear in
data/mapping_dataset_final_strat.csv (0 rows), that is, they were never part of the
model's train/val/test partitions. They are baselines recovered from the raw TIFFs
for the WHST paired analysis only. The removal therefore does not affect the
treino, o split leakage-free, nem exige re-treino.

IT DOES NOT DELETE: it moves them to whst_output/_removed_out_of_scope/ with a
manifest, and backs up every altered CSV (.bak_prefora). Reversible.

Arquivos alterados:
  whst_input/<img>.tiff                    -> quarentena
  whst_input/correspondencia.csv           -> remove linha
  data/whst_pass1_qc.csv                        -> remove linha
  data/visual_triage.csv                      -> remove linha
  data/whst_batch_results.csv                   -> remove linha (se existir)
  whst_output/{overlays,masks,rois,polygons}/<img>*  -> quarentena
  whst_output/overlays_sorted/**/<overlay>          -> quarentena
"""
import csv, os, shutil, sys

REPORT = "data/annotation_report.csv"
QUAR = os.path.join("whst_output", "_removed_out_of_scope")
MARCA = "fora_do_escopo"


def base(f):
    for e in (".tiff", ".tif"):
        if f.lower().endswith(e):
            return f[: -len(e)]
    return os.path.splitext(f)[0]


rep = list(csv.DictReader(open(REPORT, encoding="utf-8-sig")))
alvo = {r["whst_input_file"]: r for r in rep if r["motivo_classificado"] == MARCA}
if not alvo:
    sys.exit("nenhuma imagem marcada como fora_do_escopo")
print(f"images to remove: {len(alvo)}")
for k, r in alvo.items():
    print(f"  {r['analysis_unit']:<22} tp{r['timepoint_h']:<3} {k[:56]}")

# guard: confirm they are not in the training dataset
MAPF = "data/mapping_dataset_final_strat.csv"
if os.path.isfile(MAPF):
    corr = {r["whst_input_file"]: r for r in
            csv.DictReader(open("whst_input/correspondencia.csv", encoding="utf-8-sig"))}
    tb = {corr[k]["test_image"] for k in alvo if k in corr}
    M = list(csv.DictReader(open(MAPF, encoding="utf-8")))
    hit = [r for r in M if r["arquivo_b"] in tb]
    if hit:
        sys.exit(f"ABORTED: {len(hit)} of these images ARE in the training dataset "
                 f"({MAPF}). Remocao exigiria re-split/re-treino — revise antes.")
    print(f"  guard OK: 0 of these images in {MAPF} (train/val/test unaffected)")

os.makedirs(QUAR, exist_ok=True)
movidos = []

# ---- 1. arquivos ----
for k in alvo:
    b = base(k)
    cands = [os.path.join("whst_input", k)]
    for sub, suf in (("overlays", "_overlay.jpg"), ("masks", "_mask.png"),
                     ("rois", ".roi"), ("polygons", "_polygon.csv")):
        d = os.path.join("whst_output", sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith(b):
                    cands.append(os.path.join(d, f))
    # overlay ordenado (nome diferente)
    ov = alvo[k].get("arquivo_overlay", "")
    if ov:
        for dp, _, fs in os.walk(os.path.join("whst_output", "overlays_sorted")):
            if ov in fs:
                cands.append(os.path.join(dp, ov))
    for p in cands:
        if os.path.exists(p):
            dest = os.path.join(QUAR, os.path.basename(p))
            if os.path.exists(dest):
                dest = os.path.join(QUAR, f"{os.path.basename(os.path.dirname(p))}__{os.path.basename(p)}")
            shutil.move(p, dest)
            movidos.append((p, dest))
print(f"\nfiles moved to quarantine: {len(movidos)}")

# ---- 2. CSVs ----
def filtra(path, col, chaves, enc="utf-8-sig"):
    if not os.path.isfile(path):
        return None
    rows = list(csv.DictReader(open(path, encoding=enc)))
    if col not in rows[0]:
        return None
    keep = [r for r in rows if r[col] not in chaves]
    n = len(rows) - len(keep)
    if n:
        shutil.copy2(path, path + ".bak_prefora")
        with open(path, "w", encoding=enc, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(keep)
    return len(rows), len(keep), n


for path, col in (("whst_input/correspondencia.csv", "whst_input_file"),
                  ("data/whst_pass1_qc.csv", "whst_input_file"),
                  ("data/visual_triage.csv", "whst_input_file")):
    r = filtra(path, col, set(alvo))
    if r:
        print(f"  {path:<38} {r[0]} -> {r[1]}  (removidas {r[2]})")

# data/whst_batch_results.csv usa o nome do arquivo na 1a coluna
p = "data/whst_batch_results.csv"
if os.path.isfile(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    c0 = list(rows[0].keys())[0]
    keep = [r for r in rows if r[c0] not in alvo]
    if len(keep) != len(rows):
        shutil.copy2(p, p + ".bak_prefora")
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(keep)
        print(f"  {p:<38} {len(rows)} -> {len(keep)}  (removidas {len(rows)-len(keep)})")

# ---- 3. manifesto ----
man = os.path.join(QUAR, "MANIFESTO.csv")
with open(man, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["whst_input_file", "analysis_unit", "timepoint_h", "motivo_operador",
                "justificativa", "arquivos_movidos"])
    for k, r in alvo.items():
        n = sum(1 for a, b in movidos if base(os.path.basename(a)).startswith(base(k))
                or os.path.basename(a) == r.get("arquivo_overlay"))
        w.writerow([k, r["analysis_unit"], r["timepoint_h"], r["motivo"],
                    "algorithm test image; not experimental data. "
                    "Verificado ausente de data/mapping_dataset_final_strat.csv.", n])
print(f"\nmanifesto: {man}")
print("Reversible: the files are in quarantine and each CSV has a .bak_prefora")
