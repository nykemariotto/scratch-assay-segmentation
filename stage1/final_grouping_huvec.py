# -*- coding: utf-8 -*-
"""The FINAL HUVEC grouping key + feasibility of the 70/15/15 split.

⚠️ RUN ONCE, NOT REGENERABLE — IT DRAWS. Re-running it rebuilds the leakage-free
partition, which every trained model and every reported statistic depends on."""
import csv, re, os, json, random, collections

BASE = os.environ.get("SCRATCH_ASSAY_ROOT", ".")
SRC = os.path.join(BASE, "data/mapping_huvec_final.csv")
OUTCSV = os.path.join(BASE, "data/mapping_huvec_final.csv")   # reescreve com grp_huvec
OUTJSON = os.path.join(BASE, "stage1/grouping_huvec.json")

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
hv = [r for r in rows if r["linha_celular"] == "HUVEC"]
OUT = {"n_huvec": len(hv)}
print(f"HUVEC anotadas: {len(hv)}")


def campo_of(s):
    st = re.sub(r"\.(tiff?|png|bmp)$", "", s or "", flags=re.I)
    st = re.sub(r"\s*\(\d+\)\s*$", "", st)          # remove sufixo de colisao "(2)"
    m = re.search(r"(\d)\s*$", st)
    return m.group(1) if m else ""


for r in hv:
    r["campo"] = campo_of(r["arquivo_a"] or r["stem_normalizado"])

# =========================================================================
# POINT 1 - within a batch, is the treatment always determined by the subfolder?
#           e nos lotes achatados, (lote, well) ja identifica o well fisico?
# =========================================================================
print("\n=== [1] lote -> tratamentos ===")
lote_trats = collections.defaultdict(collections.Counter)
for r in hv:
    lote_trats[r["lote"]][r["trat_huvec"]] += 1
achatados, subpastas = [], []
for lo, c in sorted(lote_trats.items(), key=lambda x: -sum(x[1].values())):
    tipo = "ACHATADO" if list(c) == [""] else "COM SUBPASTA"
    (achatados if tipo == "ACHATADO" else subpastas).append(lo)
    print(f"  {lo!r:45s} n={sum(c.values()):4d} {tipo}  trats={dict(c)}")
OUT["lotes_achatados"] = [l for l in achatados if l]
OUT["lotes_com_subpasta"] = subpastas
OUT["lote_trats"] = {lo: dict(c) for lo, c in lote_trats.items()}

# in a flattened batch, does one well appear in >1 context? (0 by definition)
print("\n  well com >1 tratamento dentro do mesmo lote:")
viol = 0
wt = collections.defaultdict(set)
for r in hv:
    wt[(r["lote"], r["well_campo"])].add(r["trat_huvec"])
for (lo, w), ts in sorted(wt.items()):
    if len(ts) > 1:
        viol += 1
        print(f"    {lo!r} well={w} -> {sorted(ts)}")
print(f"    total violacoes: {viol}")
OUT["wells_com_multi_trat_no_lote"] = viol

# how many wells are reused across batches (shows why 'well' alone is invalid)
w_lotes = collections.defaultdict(set)
for r in hv:
    w_lotes[r["well_campo"]].add((r["lote"], r["trat_huvec"]))
cross = sum(1 for w, s in w_lotes.items() if len(s) > 1)
print(f"\n  distinct wells: {len(w_lotes)}; crossing >1 (batch,treatment): {cross}")
OUT["wells_distintos"] = len(w_lotes)
OUT["wells_cruzando_contexto"] = cross

# =========================================================================
# PONTO 3 - fusao de lotes canonicos
# Fase 1 concluiu: 'Originais (1)', 'originais', 'Migracao n3' sao experimentos
# DISTINCT -> do NOT merge. The only merge applied: Controle + Saudavel (conservative,
# resolve as 11 anotadas ambiguas entre esses dois lotes).
# =========================================================================
CONTROLE = [l for l in lote_trats if l.startswith("Controle")]
SAUDAVEL = [l for l in lote_trats if l.startswith("Saud")]
FUSAO = {l: "Controle+Saudavel" for l in CONTROLE + SAUDAVEL}
print(f"\n=== [3] fusao canonica aplicada: {list(FUSAO)} -> 'Controle+Saudavel' ===")
OUT["fusao_aplicada"] = {"lotes": list(FUSAO), "canonico": "Controle+Saudavel"}

# =========================================================================
# POINT 4 - the 71 with no physical source (origem_lote != 'hash')
# =========================================================================
print("\n=== [4] procedencia do rotulo de lote ===")
proc = collections.Counter(r["origem_lote"] for r in hv)
print("  origem_lote:", dict(proc))
amb = [r for r in hv if r["lote_ambiguo"].strip()]
print(f"  ambiguas entre lotes: {len(amb)}")
print("  ", collections.Counter(r["lote_ambiguo"] for r in amb))
OUT["origem_lote"] = dict(proc)
OUT["ambiguas"] = collections.Counter(r["lote_ambiguo"] for r in amb)

# resolucao: as 11 Controle-vs-Saudavel viram o lote fundido; a 1 restante e excluida
resolvidas, excluidas = 0, []
for r in hv:
    lo = r["lote"].strip()
    amb_s = r["lote_ambiguo"].strip()
    if amb_s:
        parts = [p.split("/")[0] for p in amb_s.split("|")]
        if all(p in FUSAO for p in parts):
            r["lote_canon"] = "Controle+Saudavel"
            r["resolucao"] = "ambigua_resolvida_por_fusao"
            resolvidas += 1
            continue
        r["lote_canon"] = ""
        r["resolucao"] = "excluida_ambigua"
        excluidas.append(r)
        continue
    r["lote_canon"] = FUSAO.get(lo, lo)
    r["resolucao"] = "ok_" + (r["origem_lote"] or "sem_origem")
print(f"  ambiguas resolvidas pela fusao : {resolvidas}")
print(f"  excluidas (ambiguidade real)   : {len(excluidas)}")
for r in excluidas:
    print(f"     -> {r['stem_normalizado']!r} amb={r['lote_ambiguo']!r}")
OUT["ambiguas_resolvidas_por_fusao"] = resolvidas
OUT["excluidas"] = [r["stem_normalizado"] for r in excluidas]

usable = [r for r in hv if r["lote_canon"] and r["well_campo"].strip()]
print(f"  utilizaveis (lote_canon & well): {len(usable)} / {len(hv)}")
OUT["utilizaveis"] = len(usable)

# =========================================================================
# PONTO 2 - timepoints por (lote, trat, well)
# =========================================================================
print("\n=== [2] timepoints por well fisico (lote_canon, trat, well) ===")
tps = collections.defaultdict(set)
imgs = collections.Counter()
for r in usable:
    k = (r["lote_canon"], r["trat_huvec"], r["well_campo"])
    tps[k].add(r["timepoint_h"])
    imgs[k] += 1
dist = collections.Counter(len(v) for v in tps.values())
print("  n_timepoints -> n_wells:", dict(sorted(dist.items())))
multi = sum(1 for v in tps.values() if len(v) > 1)
print(f"  wells com >1 timepoint: {multi}/{len(tps)} "
      f"({100*multi/len(tps):.1f}%)  -> exactly what the grouping keeps together")
OUT["timepoints_por_well"] = {str(k): v for k, v in sorted(dist.items())}
OUT["wells_multi_timepoint"] = multi
OUT["wells_total_C"] = len(tps)

# =========================================================================
# MEASUREMENT OF THE CANDIDATE KEYS
# =========================================================================
KEYS = {
    "A_well":              lambda r: (r["well_campo"],),
    "B_lote_well":         lambda r: (r["lote_canon"], r["well_campo"]),
    "C_lote_trat_well":    lambda r: (r["lote_canon"], r["trat_huvec"], r["well_campo"]),
    "D_lote_trat":         lambda r: (r["lote_canon"], r["trat_huvec"]),
}


def measure(name, fn, data):
    g = collections.defaultdict(list)
    for r in data:
        g[fn(r)].append(r)
    sizes = sorted(len(v) for v in g.values())
    ntp = [len({x["timepoint_h"] for x in v}) for v in g.values()]
    # a group containing >1 distinct physical (batch,treatment,well) is conservative;
    # a physical well split across groups is LEAKAGE
    phys = collections.defaultdict(set)
    for r in data:
        phys[(r["lote_canon"], r["trat_huvec"], r["well_campo"])].add(fn(r))
    split_phys = sum(1 for v in phys.values() if len(v) > 1)
    m = dict(chave=name, n_grupos=len(g), n_imagens=len(data),
             img_min=sizes[0], img_mediana=sizes[len(sizes)//2], img_max=sizes[-1],
             img_media=round(len(data)/len(g), 2),
             grupos_com_1_img=sum(1 for s in sizes if s == 1),
             tp_por_grupo_dist=dict(sorted(collections.Counter(ntp).items())),
             wells_fisicos_partidos=split_phys,
             grupos_multi_well_fisico=sum(1 for v in g.values()
                 if len({(x["lote_canon"], x["trat_huvec"], x["well_campo"]) for x in v}) > 1))
    return m, g


print("\n=== KEY MEASUREMENT ===")
measures, groups = {}, {}
for nm, fn in KEYS.items():
    m, g = measure(nm, fn, usable)
    measures[nm], groups[nm] = m, g
    print(f"\n  {nm}")
    for k, v in m.items():
        if k != "chave":
            print(f"      {k:26s} {v}")
OUT["medicao_chaves"] = measures

# baseline A: the error that would be made
mA = measures["A_well"]
print(f"\n  >>> chave A (well sozinho) juntaria {mA['grupos_multi_well_fisico']} grupos "
      f"containing physical wells from different contexts -- it does not leak, but it "
      f"destroys granularity, and above all mixes unrelated experiments.")

# =========================================================================
# POINT 5 - simulation of the 70/15/15 split with the chosen key (C)
# =========================================================================
CHOSEN = "C_lote_trat_well"
gC = groups[CHOSEN]
print(f"\n=== [5] 70/15/15 SPLIT by group, key = {CHOSEN} ===")


# Estrato = TRATAMENTO quando existe (pooling entre lotes n1/n2, senao os bracos
# NEB ficam com 2 grupos por lote e nunca alcancam val); senao o proprio lote.
def stratum(k):
    lo, tr, w = k
    return f"trat:{tr}" if tr else f"lote:{lo}"


TARGET = {"train": .70, "val": .15, "test": .15}


STRATA = collections.defaultdict(list)
for k in gC:
    STRATA[stratum(k)].append(k)


def do_split(seed):
    """LPT per stratum: groups ordered by decreasing size (ties broken at random)
    go to the partition with the largest image deficit against its quota. Then a
    coverage repair: if a partition ended up without a treatment, a timepoint or a
    batch, the smallest eligible group is moved there."""
    rnd = random.Random(seed)
    assign = {}
    for s, ks in STRATA.items():
        ks = sorted(ks, key=lambda k: (-len(gC[k]), rnd.random()))
        tot_s = sum(len(gC[k]) for k in ks)
        quota = {p: TARGET[p] * tot_s for p in TARGET}
        got = {p: 0.0 for p in TARGET}
        for k in ks:
            p = max(TARGET, key=lambda p: (quota[p] - got[p], rnd.random()))
            assign[k] = p
            got[p] += len(gC[k])
    # ---- reparo de cobertura ----
    def facets(k):
        v = gC[k]
        return (["trat:" + r["trat_huvec"] for r in v]
                + ["tp:" + r["timepoint_h"] for r in v]
                + ["lote:" + r["lote_canon"] for r in v])

    todos = set()
    for k in gC:
        todos |= set(facets(k))
    for _ in range(30):
        pres = {p: set() for p in TARGET}
        for k in gC:
            pres[assign[k]] |= set(facets(k))
        falta = [(p, f) for p in TARGET for f in todos if f not in pres[p]]
        if not falta:
            break
        p, f = falta[0]
        cand = [k for k in gC if f in facets(k) and assign[k] != p]
        if not cand:
            break
        # move the smallest group, taking it from the partition richest in that facet
        cont = collections.Counter(assign[k] for k in cand)
        doador = cont.most_common(1)[0][0]
        cand = [k for k in cand if assign[k] == doador] or cand
        assign[min(cand, key=lambda k: len(gC[k]))] = p
    return assign


best = None
for seed in range(400):
    assign = do_split(seed)
    cnt = collections.Counter()
    for k, v in gC.items():
        cnt[assign[k]] += len(v)
    tot = sum(cnt.values())
    err = sum(abs(cnt[p]/tot - TARGET[p]) for p in TARGET)
    tpset = collections.defaultdict(set)
    trset = collections.defaultdict(set)
    loset = collections.defaultdict(set)
    for k, v in gC.items():
        for r in v:
            tpset[assign[k]].add(r["timepoint_h"])
            trset[assign[k]].add(r["trat_huvec"])
            loset[assign[k]].add(r["lote_canon"])
    ntr_tot = len({r["trat_huvec"] for v in gC.values() for r in v})
    nlo_tot = len({r["lote_canon"] for v in gC.values() for r in v})
    ok = all(len(tpset[p]) == 4 and len(trset[p]) == ntr_tot and len(loset[p]) == nlo_tot
             for p in TARGET)
    score = (0 if ok else 1, err)
    if best is None or score < best[0]:
        best = (score, seed, dict(assign), ok)
(_sc, seed, assign, cobertura_total) = best
err = _sc[1]
print(f"  melhor seed={seed} erro_abs_total={err:.4f} "
      f"cobertura_total(tp+trat+lote em train/val/test)={cobertura_total}")
OUT["cobertura_total"] = bool(cobertura_total)

cnt = collections.Counter()
for k, v in gC.items():
    cnt[assign[k]] += len(v)
tot = sum(cnt.values())
grp_cnt = collections.Counter(assign.values())
print(f"  imagens: train {cnt['train']} ({100*cnt['train']/tot:.1f}%)  "
      f"val {cnt['val']} ({100*cnt['val']/tot:.1f}%)  "
      f"test {cnt['test']} ({100*cnt['test']/tot:.1f}%)   total {tot}")
print(f"  grupos : train {grp_cnt['train']}  val {grp_cnt['val']}  test {grp_cnt['test']}"
      f"   total {len(gC)}")

# distribuicao por timepoint
print("\n  timepoint × partition (images):")
tp_tab = collections.defaultdict(collections.Counter)
for k, v in gC.items():
    for r in v:
        tp_tab[r["timepoint_h"]][assign[k]] += 1
for tp in sorted(tp_tab, key=lambda x: int(x)):
    c = tp_tab[tp]
    t = sum(c.values())
    print(f"    {tp:>3}h  train {c['train']:4d} ({100*c['train']/t:4.1f}%)  "
          f"val {c['val']:3d} ({100*c['val']/t:4.1f}%)  test {c['test']:3d} ({100*c['test']/t:4.1f}%)  tot {t}")

# distribuicao por lote
print("\n  batch × partition (images):")
lo_tab = collections.defaultdict(collections.Counter)
for k, v in gC.items():
    for r in v:
        lo_tab[r["lote_canon"]][assign[k]] += 1
for lo in sorted(lo_tab):
    c = lo_tab[lo]
    t = sum(c.values())
    print(f"    {lo!r:40s} train {c['train']:4d}  val {c['val']:3d}  test {c['test']:3d}  tot {t}")

# distribuicao por tratamento
print("\n  treatment × partition (images):")
tr_tab = collections.defaultdict(collections.Counter)
for k, v in gC.items():
    for r in v:
        tr_tab[r["trat_huvec"] or "(no label)"][assign[k]] += 1
for tr in sorted(tr_tab):
    c = tr_tab[tr]
    t = sum(c.values())
    print(f"    {tr:16s} train {c['train']:4d}  val {c['val']:3d}  test {c['test']:3d}  tot {t}")

OUT["split"] = dict(seed=seed, chave=CHOSEN,
                    imagens=dict(cnt), grupos=dict(grp_cnt), total=tot,
                    pct={k: round(100*v/tot, 2) for k, v in cnt.items()},
                    por_timepoint={tp: dict(c) for tp, c in tp_tab.items()},
                    por_lote={lo: dict(c) for lo, c in lo_tab.items()},
                    por_tratamento={tr: dict(c) for tr, c in tr_tab.items()})

# verificacao final de leakage
print("\n  VERIFICACAO: nenhum well fisico partido entre particoes?")
phys = collections.defaultdict(set)
for k, v in gC.items():
    for r in v:
        phys[(r["lote_canon"], r["trat_huvec"], r["well_campo"])].add(assign[k])
bad = sum(1 for v in phys.values() if len(v) > 1)
print(f"    physical wells in >1 partition: {bad}  (0 = no leakage)")
# tambem: mesmo nome de arquivo (identidade) em duas particoes?
nm_part = collections.defaultdict(set)
for k, v in gC.items():
    for r in v:
        nm_part[r["stem_normalizado"]].add(assign[k])
bad2 = sum(1 for v in nm_part.values() if len(v) > 1)
print(f"    identical stems in >1 partition: {bad2}")
OUT["leakage_wells_partidos"] = bad
OUT["leakage_stems_partidos"] = bad2

# =========================================================================
# CSV FINAL
# =========================================================================
for r in hv:
    if r["lote_canon"] and r["well_campo"].strip():
        k = (r["lote_canon"], r["trat_huvec"], r["well_campo"])
        r["grp_huvec"] = f"{k[0]}||{k[1] or 'None'}||{k[2]}"
        r["split_huvec"] = assign[k]
    else:
        r["grp_huvec"] = ""
        r["split_huvec"] = "EXCLUIDA"

cols = list(rows[0].keys())
for c in ["campo", "lote_canon", "resolucao", "grp_huvec", "split_huvec"]:
    if c not in cols:
        cols.append(c)
with open(OUTCSV, "w", newline="", encoding="utf-8") as f:
    wcsv = csv.DictWriter(f, fieldnames=cols)
    wcsv.writeheader()
    for r in rows:
        wcsv.writerow({c: r.get(c, "") for c in cols})
print(f"\nCSV escrito: {OUTCSV}  ({len(rows)} linhas, {len(cols)} colunas)")

json.dump(OUT, open(OUTJSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(f"JSON: {OUTJSON}")
