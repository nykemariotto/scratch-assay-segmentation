# -*- coding: utf-8 -*-
"""
stage4/prepare_recovered_baselines.py — prepares the 5 t0 recovered from the raw
bank for measurement, under the same convention as every other image in the pipeline.

CONTEXT: 14 series are outside the paired analysis for lack of a baseline. For 5
delas existe t0 no banco cru (`Banco de dados/HUVEC-RAW`), no lote, tratamento,
the correct well and field, with a previously unseen MD5 (they are not in the pipeline).

PROVENANCE DECLARED: these 5 enter by a different route from the rest — they went
through neither the blind visual triage nor the automatic QC, because they were
localizadas depois. Ficam marcadas com `BASELINE_REC` no nome e
`origem='baseline_recuperado_posteriori'` in the correspondence, so the analysis
can be redone without them as a robustness test. The border criterion of the manual
correction is the same (PROTOCOL section 1).

STEP 1 of 2: copies to whst_input_novos/ (staging) and writes the manifest.
Depois: rodar stage4/whst_batch.ijm sobre essa pasta e, em seguida,
stage4/integrate_recovered_baselines.py.
"""
import csv, hashlib, os, re, shutil, sys

SEP = os.sep
BANCO = os.environ.get("BANCO_A", "<banco_a>")
STAGE = "whst_input_novos"
MANIF = "stage4/baselines_recuperados.csv"

# (analysis_unit, campo, caminho relativo no HUVEC-RAW, series_key)
CAND = [
    ("LUIS RAW||PET||C5", "2", ("LUIS RAW", "PET", "0h", "C5 0H 2.tiff"),
     "LUIS RAW||PET||C5||c2"),
    ("LUIS RAW||PET||F2", "2", ("LUIS RAW", "PET", "0h", "F2 0H 2.tiff"),
     "LUIS RAW||PET||F2||c2"),
    ("Migração (PET e PEP) n2 - RAW||PEP||A2", "1",
     ("Migração (PET e PEP) n2 - RAW", "PEP", "0h", "A2 0H 1.tiff"),
     "Migração (PET e PEP) n2 - RAW||PEP||A2||c1"),
    ("originais||None||A5", "2", ("originais", "0hr", "A5 0HR 2.tiff"),
     "originais||None||A5||c2"),
    ("originais||None||B5", "2", ("originais", "0hr", "B5 0HR 2.tiff"),
     "originais||None||B5||c2"),
]

if not os.path.isdir(BANCO):
    sys.exit(f"banco cru inacessivel: {BANCO}")

corr = list(csv.DictReader(open("whst_input/correspondencia.csv", encoding="utf-8-sig")))
ja_md5 = {r["raw_md5"] for r in corr}
ser_exist = {r["series_key"] for r in
             csv.DictReader(open("data/whst_pass1_qc.csv", encoding="utf-8-sig"))}

os.makedirs(STAGE, exist_ok=True)


def san(s):
    s = re.sub(r"\.(tif|tiff)$", "", s, flags=re.I)
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


rows = []
for au, campo, rel, sk in CAND:
    src = os.path.join(BANCO, *rel)
    if not os.path.exists(src):
        print(f"  [FALTA] {src}"); continue
    md5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    if md5 in ja_md5:
        print(f"  [PULA] ja no pipeline (md5 {md5[:10]}): {rel[-1]}"); continue
    if sk not in ser_exist:
        print(f"  [AVISO] series_key inexistente no QC: {sk}")
    novo = (f"{md5[:10]}__BASELINE_REC__HUVEC__{san(au)}__tp0h__{san(rel[-1])}.tiff")
    dst = os.path.join(STAGE, novo)
    shutil.copy2(src, dst)
    rows.append({"whst_input_file": novo, "cell_line": "HUVEC",
                 "analysis_unit": au, "series_key": sk, "campo": campo,
                 "timepoint_h": 0, "raw_file_original": rel[-1],
                 "raw_rel": SEP.join(rel), "raw_md5": md5,
                 "origem": "baseline_recuperado_posteriori",
                 "tamanho_mb": round(os.path.getsize(src) / 1e6, 2)})
    print(f"  OK  {au[:36]:<38} c{campo}  {rel[-1]:<16} md5={md5[:10]}")

if not rows:
    sys.exit("\nnada a preparar.")
with open(MANIF, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"\n{len(rows)} images in {STAGE}/  | manifest: {MANIF}")
print("\nPROXIMO PASSO — medir com o MESMO pipeline automatico:")
print("  Fiji: Plugins > Macros > Run... -> stage4/whst_batch.ijm")
print(f"        image folder     : <project>{SEP}{STAGE}")
print(f"        pasta de saida   : <projeto>{SEP}whst_output_novos")
print("  Depois: python stage4/integrate_recovered_baselines.py")
