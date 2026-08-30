# -*- coding: utf-8 -*-
"""Fecha as pontas soltas de contabilidade do HUVEC."""
import json, csv, re, os
from collections import defaultdict, Counter

BASE = os.environ.get("SCRATCH_ASSAY_ROOT", ".")
C = json.load(open(os.path.join(BASE, "data/hash_cache_huvec.json"), encoding="utf-8"))
raw, bd = C["raw"], C["bd"]
OUT = {}

# ---------------- 1. duplicados de hash dentro de RAW ----------------
byh = defaultdict(list)
for r in raw:
    byh[r["md5"]].append(r)
dups = {h: v for h, v in byh.items() if len(v) > 1}
print(f"[1] RAW: {len(raw)} files, {len(byh)} distinct hashes, {len(dups)} duplicated hashes")
dup_report = []
for h, v in dups.items():
    lotes = sorted({x["batch"] for x in v})
    print(f"  hash {h[:12]}  n={len(v)}  batches={lotes}")
    for x in v:
        print(f"      {x['rel']}  |  {x['name']}  ({x['size']} bytes)")
    dup_report.append(dict(md5=h, n=len(v), lotes=lotes,
                           files=[f"{x['rel']}/{x['name']}" for x in v],
                           cross_lote=len(lotes) > 1))
OUT["dups_raw"] = dup_report
OUT["dups_cross_lote"] = sum(1 for d in dup_report if d["cross_lote"])

# ---------------- 4. RAW vs BD ----------------
raw_h = set(byh)
bd_byh = defaultdict(list)
for r in bd:
    bd_byh[r["md5"]].append(r)
bd_h = set(bd_byh)
only_raw = raw_h - bd_h
only_bd = bd_h - raw_h
print(f"\n[4] BD: {len(bd)} files, {len(bd_h)} distinct hashes")
print(f"    hashes so em RAW (crua nunca no BD): {len(only_raw)} "
      f"-> {sum(len(byh[h]) for h in only_raw)} arquivos")
print(f"    hashes only in the bank (no matching raw): {len(only_bd)}")
cnt = Counter()
for h in only_raw:
    for x in byh[h]:
        cnt[(x["batch"], x["treatment"])] += 1
for k, n in cnt.most_common():
    print(f"      so-RAW: {k[0]} / {k[1]} -> {n}")
OUT["only_raw_hashes"] = len(only_raw)
OUT["only_raw_files"] = sum(len(byh[h]) for h in only_raw)
OUT["only_bd_hashes"] = len(only_bd)
OUT["only_raw_by_ctx"] = {f"{k[0]}/{k[1]}": n for k, n in cnt.items()}
OUT["only_bd_files"] = [f"{bd_byh[h][0]['rel']}/{bd_byh[h][0]['name']}" for h in only_bd]

# BD duplicados
bd_dups = {h: v for h, v in bd_byh.items() if len(v) > 1}
print(f"    BD duplicated hashes: {len(bd_dups)}")
OUT["bd_dups"] = [dict(md5=h, files=[f"{x['rel']}/{x['name']}" for x in v]) for h, v in bd_dups.items()]

# ---------------- contexto por hash ----------------
h2ctx = {h: sorted({(x["batch"], x["treatment"]) for x in v}) for h, v in byh.items()}

# ---------------- 2/3. mapping ----------------
rows = list(csv.DictReader(open(os.path.join(BASE, "stage1/mapping_huvec_treatment.csv"), encoding="utf-8")))
hv = [r for r in rows if r["linha_celular"] == "HUVEC"]
print(f"\n[2/3] HUVEC rows in the mapping: {len(hv)}")
semfonte = [r for r in hv if not r["lote"].strip()]
print(f"      no source (empty batch): {len(semfonte)}")

# BD lookup por (pasta,arquivo) e por nome normalizado
def norm(s):
    s = re.sub(r"\.(tiff?|png|bmp)$", "", s.strip(), flags=re.I)
    s = s.lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

bd_key = {}
for r in bd:
    pasta = f"HUVEC/{r['rel']}" if r["rel"] != "." else "HUVEC"
    bd_key[(pasta, r["name"])] = r

# indice por nome normalizado dentro do RAW
raw_byname = defaultdict(list)
for r in raw:
    raw_byname[norm(r["name"])].append(r)
# index by normalised name within the bank (to find the physical file under another case)
bd_byname = defaultdict(list)
for r in bd:
    bd_byname[norm(r["name"])].append(r)

def well_of(fn):
    stem = re.sub(r"\.(tiff?|png|bmp)$", "", fn, flags=re.I)
    m = re.match(r"^\s*([A-Fa-f])\s*(\d{1,2})\b", stem)
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else None

res = Counter()
detalhe = []
for r in semfonte:
    arq = r["arquivo_a"].strip()
    pasta = r["pasta_a"].strip()
    nm = norm(arq) if arq else norm(r["stem_normalizado"])
    w = well_of(arq) if arq else well_of(r["stem_normalizado"])
    tp = r["timepoint_h"]
    # via 1: nome normalizado casa BD (case-insensitive) -> hash -> contexto
    ctxs = set()
    via = None
    cands = bd_byname.get(nm, [])
    # restringe a mesma pasta se possivel
    if pasta:
        same = [c for c in cands if (f"HUVEC/{c['rel']}" if c["rel"] != "." else "HUVEC") == pasta]
        if same:
            cands = same
    if cands:
        for c in cands:
            ctxs |= set(h2ctx.get(c["md5"], []))
        via = "bd_nome_norm"
    if not ctxs:
        rc = raw_byname.get(nm, [])
        if rc:
            ctxs = {(x["batch"], x["treatment"]) for x in rc}
            via = "raw_nome_norm"
    ctxs = sorted(ctxs)
    if len(ctxs) == 1:
        res["recuperado_unico"] += 1
        st = "unico"
    elif len(ctxs) > 1:
        res["ambiguo_multi_lote"] += 1
        st = "ambiguo"
    else:
        res["irrecuperavel"] += 1
        st = "nenhum"
    detalhe.append(dict(arquivo_b=r["arquivo_b"], arquivo_a=arq, pasta_a=pasta,
                        stem=r["stem_normalizado"], well=w, timepoint=tp,
                        n_cands=len(cands), via=via, status=st,
                        ctxs=[f"{b}/{t}" for b, t in ctxs],
                        arquivo_fisico=(f"{cands[0]['rel']}/{cands[0]['name']}" if cands else "")))

print("      recovery of the source-less ones:", dict(res))
n_well = sum(1 for d in detalhe if d["well"])
n_tp = sum(1 for d in detalhe if d["timepoint"])
print(f"      well recoverable by name: {n_well}/{len(detalhe)}; timepoint: {n_tp}/{len(detalhe)}")
for d in detalhe[:15]:
    print(f"        {d['status']:9s} well={d['well']} tp={d['timepoint']} "
          f"via={d['via']} ctxs={d['ctxs'][:3]} arq={d['arquivo_a'] or d['stem']}")
OUT["semfonte_total"] = len(semfonte)
OUT["semfonte_res"] = dict(res)
OUT["semfonte_well_ok"] = n_well
OUT["semfonte_tp_ok"] = n_tp
OUT["semfonte_detalhe"] = detalhe

# contabilidade final
com_lote = len(hv) - len(semfonte)
final_ok = com_lote + res["recuperado_unico"]
amb = sum(1 for r in hv if r["ctx_ambiguo"].strip())
well_ok = sum(1 for r in hv if r["well_campo"].strip())
print(f"\n[3] HUVEC anotadas = {len(hv)}")
print(f"    with (batch,treat) by hash          : {com_lote}")
print(f"    + recovered by normalised name      : {res['recuperado_unico']}")
print(f"    = with (batch,treat)                : {final_ok}")
print(f"    ambiguous across batches            : {res['ambiguo_multi_lote']}")
print(f"    with no source at all               : {res['irrecuperavel']}")
print(f"    ambiguous ctx in the hash join      : {amb}")
print(f"    well_campo filled in                : {well_ok}/{len(hv)}")
# completo = lote + trat + well
completo = 0
for r in hv:
    if r["lote"].strip() and r["well_campo"].strip():
        completo += 1
print(f"    (batch & well) complete by hash     : {completo}")
comp_final = completo + sum(1 for d in detalhe if d["status"] == "unico" and d["well"])
print(f"    (batch & well) complete after recov.: {comp_final}")
OUT["hv_total"] = len(hv)
OUT["com_lote_hash"] = com_lote
OUT["com_lote_final"] = final_ok
OUT["ctx_ambiguo_hash"] = amb
OUT["well_ok"] = well_ok
OUT["completo_hash"] = completo
OUT["completo_final"] = comp_final

json.dump(OUT, open(os.path.join(BASE, "stage1/close_huvec.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("\nsalvo stage1/close_huvec.json")
