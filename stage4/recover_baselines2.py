# -*- coding: utf-8 -*-
"""
stage4/recover_baselines2.py — part 2, CORRECTED. Resolves the 0h of the missing
groups by the same field_id definition as the dataset
(stage1/final_grouping_skov.py), not by the rank of
timestamp. Valida a regra contra os 0h JA anotados e atribui nivel de confianca
conforme a estabilidade do field_id do proprio grupo entre timepoints.
"""
import os, re, csv
from collections import defaultdict

P1 = os.environ.get("RAW_ARCHIVE_P1", "<raw_archive_p1>")
P2 = os.environ.get("RAW_ARCHIVE_P2", "<raw_archive_p2>")
HRAW = os.environ.get("BANCO_A", "<banco_a>") + r"\HUVEC-RAW"
IMG = (".tif", ".tiff", ".bmp", ".png")

TCANON = {"ct": "CT", "ptx": "PTX", "75": "GEO", "75ug": "GEO", "geo": "GEO",
          "75ptx": "GEO+PTX", "75PTX": "GEO+PTX", "75ug+PTX": "GEO+PTX",
          "carbo": "CARBO", "carbo_geo": "CARBO+GEO"}

sk = list(csv.DictReader(open("data/mapping_final_skov.csv", encoding="utf-8")))


def snap_num(a):
    m = re.match(r"snap[-_ ]?(\d+)", a.lower())
    return int(m.group(1)) if m else None


# block_min por (exp, treatment, tp) = min snap ANOTADO (como no dataset)
block_min = {}
tmp = defaultdict(list)
for r in sk:
    n = snap_num(r["arquivo_a"])
    if n is not None:
        tmp[(r["experimento"], r["tratamento"], r["timepoint_h"])].append(n)
for k, v in tmp.items():
    block_min[k] = min(v)

# ---- FORMULA validation (not rank): does the annotated field_id match the rule? ----
print("=== validacao da regra field_id (formula do dataset) nos 0h anotados ===")
okc = badc = 0
for r in sk:
    if r["timepoint_h"] != "0":
        continue
    fid = int(r["field_id"])
    n = snap_num(r["arquivo_a"])
    if n is not None:  # P2: snap-block
        bm = block_min.get((r["experimento"], r["tratamento"], "0"))
        exp = n - bm + 1 if bm is not None else None
    else:  # P1: nomeado ct1_2 / 75geo1_2 / 75ug+ptx1_2
        m = re.match(r"^(75ug\+ptx|75geo|ptx|ct)(\d+)_(\d+)$",
                     re.sub(r"\.(tiff?|png)$", "", r["arquivo_a"].strip(), flags=re.I).lower())
        exp = (int(m.group(2)) - 1) * 5 + int(m.group(3)) if m else None
    if exp == fid:
        okc += 1
    else:
        badc += 1
print(f"  regra confere: {okc} ok, {badc} divergentes (de {okc+badc} anotados em 0h)")

# ---- stability of each target group's field_id (same snap between 24h/48h?) ----
def grp_stability(grp):
    rows = [r for r in sk if r["grp_field"] == grp]
    snaps = {}
    for r in rows:
        if r["timepoint_h"] in ("24", "48"):
            snaps[r["timepoint_h"]] = snap_num(r["arquivo_a"])
    if "24" in snaps and "48" in snaps:
        return snaps["24"] == snaps["48"], snaps
    return None, snaps


# ---- resolve the 0h of each missing SKOV group ----
def p1_named_0h(tr, fid):
    rep, field = (fid - 1) // 5 + 1, (fid - 1) % 5 + 1
    prefix = {"GEO": "75geo", "GEO+PTX": "75ug+ptx", "CT": "ct", "PTX": "ptx"}[tr]
    fname = f"{prefix}{rep}_{field}.tiff"
    folder = {"GEO": "75ug", "GEO+PTX": "75ug+PTX", "CT": "CT", "PTX": "PTX"}[tr]
    return os.path.join(P1, "0h", folder, fname), fname


P2_FOLDER = {"CT": "ct", "CARBO": "carbo", "GEO": "geo", "CARBO+GEO": "carbo_geo"}


def p2_snap_file(tr, snap):
    """Blocos P2 sao identicos entre timepoints, entao o snap do grupo (24h/48h)
    identifies the 0h of the same physical field. Direct method (not the annotated
    block_min)."""
    fname = f"Snap-{snap:02d}.tiff"
    return os.path.join(P2, "0h", P2_FOLDER[tr], fname), fname


targets = [("P1", "GEO+PTX", 1), ("P1", "GEO+PTX", 6), ("P1", "GEO", 11),
           ("P1", "GEO", 6), ("P2", "CARBO", 5), ("P2", "CT", 15)]

print("\n=== BASELINES SKOV (formula + confianca) ===")
results = []
for exp, tr, fid in targets:
    grp = f"{exp}|{tr}|F{fid}"
    stable, snaps = grp_stability(grp)
    alt = ""
    if exp == "P1":
        path, fname = p1_named_0h(tr, fid)
        conf = "media (0h nomeado; ordem de aquisicao inferida)" if stable else "baixa (field_id instavel)"
    else:
        # usa o snap do PROPRIO grupo (blocos P2 identicos entre timepoints)
        snap = snaps.get("24", snaps.get("48"))
        path, fname = p2_snap_file(tr, snap)
        if stable:
            conf = "alta (P2 bloco estavel; snap do grupo)"
        else:
            conf = f"baixa (field_id instavel: candidatos Snap-{snaps['24']:02d}/Snap-{snaps['48']:02d})"
            ap, af = p2_snap_file(tr, snaps["48"])
            alt = af if os.path.exists(ap) else ""
    exists = os.path.exists(path) if path else False
    results.append((grp, path if exists else "", fname, exists, conf, stable, snaps, alt))
    st = "estavel" if stable else ("INSTAVEL" if stable is False else "?")
    print(f"  {grp:<16} -> {fname:<16} {'EXISTE' if exists else 'AUSENTE':<7} "
          f"[{st} 24/48h={snaps}] {'alt=' + alt if alt else ''} conf={conf}")

# ---- HUVEC ----
def well_of(fn):
    m = re.match(r"^\s*([A-Fa-f])\s*(\d{1,2})\b", os.path.splitext(fn)[0])
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else None

def huvec_0h(lotes, well):
    hits = []
    for lote in lotes:
        for sub in ("0h", "0hr"):
            d = os.path.join(HRAW, lote, sub)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(IMG) and "scale" not in f.lower() and well_of(f) == well:
                        hits.append(os.path.join(d, f))
    return hits

print("\n=== BASELINES HUVEC ===")
huvec = [("Controle+Saudavel||None||A1", ["Controle", "Saudável"], "A1"),
         ("originais||None||A5", ["originais"], "A5")]
huvec_res = []
for grp, lotes, well in huvec:
    hits = huvec_0h(lotes, well)
    huvec_res.append((grp, hits, well))
    print(f"  {grp}: {len(hits)} img 0h de {well}  conf={'alta (well direto)' if hits else 'N/A'}")
    for h in hits:
        print(f"      {h}")

# ---- CSV final ----
with open("stage4/baselines_recuperados.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group_key", "cell_line", "status", "baseline_0h_path", "arquivo",
                "arquivo_alt", "n_candidatos", "confianca"])
    for grp, path, fname, exists, conf, stable, snaps, alt in results:
        w.writerow([grp, "SKOV-3", "recuperavel" if exists else "sem_0h", path,
                    fname if exists else "", alt, 1 if exists else 0, conf if exists else ""])
    for grp, hits, well in huvec_res:
        w.writerow([grp, "HUVEC", "recuperavel" if hits else "sem_0h",
                    hits[0] if hits else "",
                    os.path.basename(hits[0]) if hits else "",
                    os.path.basename(hits[1]) if len(hits) > 1 else "",
                    len(hits), "alta (well direto)" if hits else ""])

rec = sum(1 for r in results if r[3]) + sum(1 for _, h, _ in huvec_res if h)
print(f"\n=== SUMMARY: {rec}/8 recoverable ===")
norec = [r[0] for r in results if not r[3]] + [g for g, h, w in huvec_res if not h]
print(f"  without 0h (excluded from the paired analysis): {norec}")
print("\nSalvo: stage4/baselines_recuperados.csv")
