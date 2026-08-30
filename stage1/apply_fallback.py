# -*- coding: utf-8 -*-
"""
stage1/apply_fallback.py — M0 + M1.

M0: unifica HUVEC (data/mapping_huvec_final.csv) e SKOV (data/mapping_final_skov.csv) num
    unico mapa por imagem, com group_key homogenea.
M1: applies the fallback of the 19 shared wells before the split, via a super-key
    SHARED__<well>, fundindo 'Originais (1)' e 'originais' do mesmo well local.
"""
import csv
from collections import Counter, defaultdict

HUVEC_CSV = "data/mapping_huvec_final.csv"   # 1369 linhas (HUVEC com grp_huvec)
SKOV_CSV = "data/mapping_final_skov.csv"     # 333 linhas (com grp_field)
OUT = "stage1/mapping_with_splitkey.csv"

FALLBACK_LOTES = {"originais", "originais (1)"}
# SHARED_WELLS is not hard-coded: it is computed as the real intersection of the two
# batches' wells (the block after M0). Hard-coding would be silent leakage if a
# revision of the dataset created a new well in both batches or changed a batch's
# spelling.
SHARED_WELLS = set()

# ---------------------------------------------------------------- M0
base = list(csv.DictReader(open(HUVEC_CSV, encoding="utf-8")))
skov = {r["arquivo_b"]: r for r in csv.DictReader(open(SKOV_CSV, encoding="utf-8"))}

rows = []
for r in base:
    cl = r["linha_celular"]
    o = {
        "arquivo_b": r["arquivo_b"],
        "stem_normalizado": r["stem_normalizado"],
        "linha_celular": cl,
        "timepoint_h": r["timepoint_h"],
        "arquivo_a": r["arquivo_a"],
        "pasta_a": r["pasta_a"],
    }
    if cl == "HUVEC":
        o["group_key"] = r["grp_huvec"]
        o["lote"] = r["lote_canon"]
        o["well"] = r["well_campo"]
        o["tratamento"] = r["trat_huvec"] or "None"
        o["estrato"] = f'HUVEC|{r["trat_huvec"] or "None"}'
        o["excluida"] = "1" if r.get("split_huvec") == "EXCLUIDA" else ""
        o["resolucao"] = r.get("resolucao", "")
    elif cl == "SKOV":
        s = skov.get(r["arquivo_b"])
        o["group_key"] = s["grp_field"] if s else ""
        o["lote"] = s["experimento"] if s else ""
        o["well"] = f'F{s["field_id"]}' if s else ""
        o["tratamento"] = s["tratamento"] if s else ""
        o["estrato"] = s["grp_treat"] if s else ""
        o["excluida"] = ""
        o["resolucao"] = "ok_hash"
    else:  # barras de escala
        o["group_key"] = ""
        o["lote"] = o["well"] = o["tratamento"] = o["estrato"] = ""
        o["excluida"] = "1"
        o["resolucao"] = "scale"
    rows.append(o)

print(f"M0 — unified map: {len(rows)} rows")
print(f"   by cell line: {dict(Counter(r['linha_celular'] or '(scale)' for r in rows))}")
print(f"   excluded: {sum(1 for r in rows if r['excluida'])} "
      f"(1 ambiguous HUVEC + 2 scale bars)")
sem_grupo = [r for r in rows if not r["group_key"] and not r["excluida"]]
print(f"   without group_key and not excluded: {len(sem_grupo)}")

# SHARED_WELLS dinamico: wells presentes nos DOIS lotes de fallback (HUVEC ativo).
_wells_por_lote = defaultdict(set)
for r in rows:
    if r["linha_celular"] == "HUVEC" and not r["excluida"]:
        lk = r["lote"].strip().lower()
        if lk in FALLBACK_LOTES:
            _wells_por_lote[lk].add(r["well"])
SHARED_WELLS = _wells_por_lote["originais (1)"] & _wells_por_lote["originais"]
print(f"   SHARED_WELLS (intersecao dinamica): {len(SHARED_WELLS)} -> {sorted(SHARED_WELLS)}")


# ---------------------------------------------------------------- M1
def split_key_of(r):
    if r["excluida"]:
        return ""
    if (r["linha_celular"] == "HUVEC"
            and r["lote"].strip().lower() in FALLBACK_LOTES
            and r["well"] in SHARED_WELLS):
        return f'SHARED__{r["well"]}'
    return r["group_key"]


for r in rows:
    r["split_key"] = split_key_of(r)

act = [r for r in rows if not r["excluida"]]
n_before = len({r["group_key"] for r in act})
n_after = len({r["split_key"] for r in act})
fused = [r for r in act if r["split_key"].startswith("SHARED__")]

print(f"\nM1 — fallback via the super-key")
print(f"   groups before: {n_before}")
print(f"   groups after : {n_after}   (reduction {n_before - n_after})")
print(f"   images under SHARED__: {len(fused)}")
print(f"   wells fundidos ({len({r['well'] for r in fused})}): "
      f"{sorted({r['well'] for r in fused})}")

# sanity: does each SHARED contain exactly the two batches?
chk = defaultdict(set)
for r in fused:
    chk[r["split_key"]].add(r["lote"])
so_um = {k: v for k, v in chk.items() if len(v) < 2}
print(f"   super-keys with only 1 batch: {len(so_um)} {sorted(so_um) if so_um else ''}")

print(f"\n   groups by cell line (after the fallback):")
for cl in ("HUVEC", "SKOV"):
    print(f"      {cl}: {len({r['split_key'] for r in act if r['linha_celular']==cl})}")

# ---- FAIL-FAST: do not write a potentially leaky map in a non-interactive pipeline.
# 'excluida' e STRING ('1'/'') -> contar por comprehension, nunca sum(r['excluida']).
assert len(rows) == 1369, f"esperado 1369 linhas, obtido {len(rows)}"
assert len(sem_grupo) == 0, f"rows without group_key: {[r['arquivo_b'] for r in sem_grupo]}"
assert sum(1 for r in rows if r["excluida"]) == 3, \
    "esperado 3 excluidas (1 HUVEC ambigua + 2 barras de escala)"
assert len(SHARED_WELLS) == 19, \
    (f"intersecao de wells compartilhados mudou (esperado 19): {sorted(SHARED_WELLS)}"
     " -- revisao de dataset? revalidar o fallback antes de prosseguir")
assert n_before - n_after == 19, f"reducao de grupos esperada 19, obtida {n_before - n_after}"
assert len(so_um) == 0, f"super-chaves com fusao incompleta (1 so lote): {sorted(so_um)}"
print("\n   fail-fast: every anti-leakage invariant holds")

cols = list(rows[0].keys())
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"\nSalvo: {OUT}")
