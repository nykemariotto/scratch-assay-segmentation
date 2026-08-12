# -*- coding: utf-8 -*-
"""
stage4/recover_baselines.py — part 2: recovers the missing 0h (baseline) images of
the test groups that have no 0h. WHST measures the raw image (no annotation needed);
since the whole group is in test, measuring the raw 0h introduces no leakage.

Estrategia:
  SKOV: field_id = posicao (rank por CreationDateTime) dentro de (exp, tratamento, 0h).
        Validado contra os 0h JA anotados em data/mapping_final_skov.csv antes de usar.
  HUVEC: procura arquivos 0h do well nas pastas cruas do(s) lote(s) do grupo.
"""
import os, re, csv
from collections import defaultdict

P1 = os.environ.get("RAW_ARCHIVE_P1", "<raw_archive_p1>")
P2 = os.environ.get("RAW_ARCHIVE_P2", "<raw_archive_p2>")
HRAW = os.environ.get("BANCO_A", "<banco_a>") + r"\HUVEC-RAW"
IMG = (".tif", ".tiff", ".bmp", ".png")

TREAT_CANON = {"ct": "CT", "CT": "CT", "ptx": "PTX", "PTX": "PTX",
               "75": "GEO", "75ug": "GEO", "75geo": "GEO", "geo": "GEO",
               "75ptx": "GEO+PTX", "75PTX": "GEO+PTX", "75ug+PTX": "GEO+PTX",
               "carbo": "CARBO", "carbo_geo": "CARBO+GEO", "geo_carbo": "CARBO+GEO"}


def dt_of(xml):
    try:
        x = open(xml, encoding="utf-8", errors="ignore").read()
    except Exception:
        return None
    m = re.search(r"<CreationDateTime>([^<]+)<", x)
    if not m:
        return None
    mm = re.match(r"(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", m.group(1).strip())
    return tuple(map(int, mm.groups())) if mm else None


def skov_0h_fields(root, exp):
    """(exp,treatment) -> list ordered by time -> field_id 1..N -> filepath."""
    out = defaultdict(list)
    base = os.path.join(root, "0h")
    if not os.path.isdir(base):
        return out
    for tr in os.listdir(base):
        trd = os.path.join(base, tr)
        if not os.path.isdir(trd):
            continue
        canon = TREAT_CANON.get(tr, tr)
        files = []
        for f in os.listdir(trd):
            if f.lower().endswith(IMG) and "scale" not in f.lower():
                files.append((dt_of(os.path.join(trd, f + "_metadata.xml")), f, os.path.join(trd, f)))
        files.sort(key=lambda t: (t[0] or (9999,), t[1]))
        out[(exp, canon)] = files  # index i -> field_id i+1
    return out


print("Indexando 0h de P1 e P2...")
sk0 = {}
sk0.update(skov_0h_fields(P1, "P1"))
sk0.update(skov_0h_fields(P2, "P2"))
for k, v in sorted(sk0.items()):
    print(f"  {k[0]}/{k[1]:<10}: {len(v)} campos 0h")

# ---- validacao: field_id anotado em 0h bate com o rank? ----
print("\n=== validacao do rank field_id vs 0h anotado (data/mapping_final_skov.csv) ===")
skrows = [r for r in csv.DictReader(open("data/mapping_final_skov.csv", encoding="utf-8"))
          if r["timepoint_h"] == "0"]
ok = bad = 0
for r in skrows:
    key = (r["experimento"], r["tratamento"])
    fid = int(r["field_id"])
    lst = sk0.get(key, [])
    if fid <= len(lst):
        # arquivo esperado pelo rank
        exp_file = lst[fid - 1][1]
        # compara com o arquivo cru anotado (arquivo_a)
        if exp_file.lower() == r["arquivo_a"].strip().lower():
            ok += 1
        else:
            bad += 1
            if bad <= 6:
                print(f"  DIVERGE {key} F{fid}: rank->{exp_file}  anotado->{r['arquivo_a']}")
print(f"  concordancia: {ok} ok, {bad} divergentes de {len(skrows)} anotados em 0h")

# ---- resolver os grupos SKOV faltantes ----
missing_skov = [("P1", "GEO+PTX", 1), ("P1", "GEO+PTX", 6), ("P1", "GEO", 11),
                ("P1", "GEO", 6), ("P2", "CARBO", 5), ("P2", "CT", 15)]
print("\n=== BASELINES SKOV RECUPERAVEIS ===")
skov_res = []
for exp, tr, fid in missing_skov:
    lst = sk0.get((exp, tr), [])
    if fid <= len(lst):
        path = lst[fid - 1][2]
        exists = os.path.exists(path)
        skov_res.append((f"{exp}|{tr}|F{fid}", path, exists))
        print(f"  {exp}|{tr}|F{fid} -> {os.path.basename(path)}  ({'EXISTE' if exists else 'AUSENTE'})")
    else:
        skov_res.append((f"{exp}|{tr}|F{fid}", "", False))
        print(f"  {exp}|{tr}|F{fid} -> NO 0h (field_id {fid} > {len(lst)} fields)")

# ---- resolver os grupos HUVEC faltantes ----
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

print("\n=== BASELINES HUVEC RECUPERAVEIS ===")
huvec_missing = [("Controle+Saudavel", ["Controle", "Saudável", "Saudavel"], "A1"),
                 ("originais", ["originais"], "A5")]
huvec_res = []
for grp, lotes, well in huvec_missing:
    hits = huvec_0h(lotes, well)
    huvec_res.append((f"{grp}||None||{well}", hits))
    print(f"  {grp}||None||{well}: {len(hits)} imagem(ns) 0h de {well}")
    for h in hits:
        print(f"      {h}")

# ---- resumo ----
print("\n" + "=" * 60)
print("RESUMO BASELINES")
print("=" * 60)
rec = [(g, p) for g, p, e in skov_res if e] + [(g, h[0]) for g, h in huvec_res if h]
norec = [g for g, p, e in skov_res if not e] + [g for g, h in huvec_res if not h]
print(f"  recuperaveis: {len(rec)}/8")
print(f"  without 0h (excluded from the paired analysis): {len(norec)} {norec}")

with open("stage4/baselines_recuperados.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group_key", "status", "baseline_0h_path", "n_candidatos"])
    for g, p, e in skov_res:
        w.writerow([g, "recuperavel" if e else "sem_0h", p if e else "", 1 if e else 0])
    for g, h in huvec_res:
        w.writerow([g, "recuperavel" if h else "sem_0h", h[0] if h else "", len(h)])
print("\nSalvo: stage4/baselines_recuperados.csv")
