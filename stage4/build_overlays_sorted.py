# -*- coding: utf-8 -*-
"""
stage4/build_overlays_sorted.py — creates whst_output/overlays_sorted/ with copies
of the 223 overlays, renamed so that ALPHABETICAL order groups by:
    cell line  ->  series (group)  ->  timepoint  ->  field

Name pattern:
    {cell_line}__{group}__tp{NN}h__{campo}__{orig}__{md5_10}.jpg
  - tp with 2 digits (tp00h, tp08h, tp12h, tp24h, tp48h, tp72h): otherwise
    'tp8h' would sort after 'tp24h'.
  - the md5 (10 chars) goes LAST: it preserves traceability without affecting order.

BLINDING (critical): this script reads ONLY whst_input/correspondencia.csv. It
NEVER opens data/whst_pass1_qc.csv. No QC flag, category or numeric value enters
the names or the folders -> the manual triage stays blind and can later be crossed
with the automatic QC to measure sensitivity and specificity.

It also creates the EMPTY triage subfolders:
    _SEG_RUIM/_super/  _SEG_RUIM/_sub/  _IMG_INVALIDA/  _AMBIGUO/
Whatever is left in the root = OK.

It writes whst_output/overlays_sorted_map.csv (used by stage4/read_triage.py).

Safety: if overlays_sorted/ already exists and is not empty, it ABORTS (it may hold
a manual triage). Run with the environment variable FORCE=1 to recreate.
"""
import csv, os, re, shutil, sys

CORRESP = "whst_input/correspondencia.csv"
OVERLAY_DIR = "whst_output/overlays"
OUT_DIR = "whst_output/overlays_sorted"
MAP_CSV = "whst_output/overlays_sorted_map.csv"
TRIAGE = ["_SEG_RUIM/_super", "_SEG_RUIM/_sub", "_IMG_INVALIDA", "_AMBIGUO"]


def san(s):
    """alnum only; runs of non-alnum -> a single '_'. Guarantees that '__' appears
    only as a field separator, never inside a field."""
    s = re.sub(r"\.(tif|tiff|png|jpe?g)$", "", s, flags=re.I)
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def campo_of(r):
    """field derived ONLY from the correspondence (without touching the QC)."""
    if r["cell_line"].startswith("SKOV"):
        return r["group_key"].split("|")[2]          # F-field (P1|CT|F10 -> F10)
    s = re.sub(r"\.(tif|tiff|png)$", "", r["raw_file_original"], flags=re.I).strip()
    tail = [t for t in re.split(r"[ _]+", s) if t in ("1", "2")]
    return tail[-1] if tail else "1"                 # imagem de campo unico


def main():
    co = list(csv.DictReader(open(CORRESP, encoding="utf-8-sig")))
    assert len(co) == 223, f"correspondencia tem {len(co)} linhas, esperava 223"

    # index md5_prefix -> the real overlay (a join robust to sanitisation differences)
    ov = [f for f in os.listdir(OVERLAY_DIR) if f.lower().endswith(".jpg")]
    assert len(ov) == 223, f"{len(ov)} overlays, esperava 223"
    ov_by_pref = {}
    for f in ov:
        pref = f.split("__", 1)[0]
        assert pref not in ov_by_pref, f"prefixo overlay duplicado: {pref}"
        ov_by_pref[pref] = f

    # guard against clobbering a manual triage
    if os.path.isdir(OUT_DIR) and any(os.scandir(OUT_DIR)):
        if os.environ.get("FORCE") != "1":
            sys.exit(f"ABORTED: {OUT_DIR}/ already exists and is not empty "
                     f"(it may hold a triage). Use FORCE=1 to recreate.")
        shutil.rmtree(OUT_DIR)
        print(f"FORCE=1: {OUT_DIR}/ removed and recreated.")

    for t in TRIAGE:
        os.makedirs(os.path.join(OUT_DIR, t), exist_ok=True)

    rows, names = [], set()
    for r in co:
        pref = r["raw_md5"][:10]
        assert pref == r["whst_input_file"].split("__", 1)[0]
        assert pref in ov_by_pref, f"no overlay for {pref} ({r['raw_file_original']})"
        tp = int(r["timepoint_h"])
        campo = campo_of(r)
        orig = san(r["raw_file_original"])
        new = (f"{r['cell_line']}__{san(r['group_key'])}__tp{tp:02d}h__"
               f"{campo}__{orig}__{pref}.jpg")
        assert new not in names, f"nome duplicado: {new}"
        names.add(new)
        shutil.copy2(os.path.join(OVERLAY_DIR, ov_by_pref[pref]),
                     os.path.join(OUT_DIR, new))
        rows.append({"sorted_basename": new, "cell_line": r["cell_line"],
                     "group_key": r["group_key"], "timepoint_h": tp, "campo": campo,
                     "raw_file_original": r["raw_file_original"], "raw_md5": r["raw_md5"],
                     "whst_input_file": r["whst_input_file"], "test_image": r["test_image"],
                     "overlay_source": ov_by_pref[pref]})

    rows.sort(key=lambda x: x["sorted_basename"])
    with open(MAP_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # instructions inside the folder itself (no QC data -> the blinding holds)
    with open(os.path.join(OUT_DIR, "_INSTRUCOES.txt"), "w", encoding="utf-8") as f:
        f.write(
            "BLIND VISUAL TRIAGE — move ONLY the problems into the subfolders.\n"
            "Whatever stays in the ROOT = OK.\n\n"
            "  _SEG_RUIM/_super/  -> segmentacao pegou area demais (super-segmentou)\n"
            "  _SEG_RUIM/_sub/    -> segmentacao pegou area de menos (sub-segmentou)\n"
            "  _IMG_INVALIDA/     -> image unsuitable for the assay (no wound, out of focus, ...)\n"
            "  _AMBIGUO/          -> cannot be decided with confidence\n\n"
            "Ordem alfabetica ja agrupa: linha celular -> serie -> timepoint -> campo.\n"
            "Depois rode: python stage4/read_triage.py\n")

    assert len(rows) == 223
    print(f"OK: {len(rows)} copies in {OUT_DIR}/")
    print(f"    empty triage subfolders: {', '.join(TRIAGE)}")
    print(f"    map: {MAP_CSV}")
    # a sample of ONE series' ordering, to check the zero-padding
    ex = [x["sorted_basename"] for x in rows if "Controle_Saudavel_None_A1" in x["sorted_basename"]]
    print("\n  example of an ordered series (A1):")
    for n in ex[:8]:
        print("   ", n)


if __name__ == "__main__":
    main()
