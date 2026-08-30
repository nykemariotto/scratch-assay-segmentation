import csv, json, collections, os
B = os.environ.get("SCRATCH_ASSAY_ROOT", ".")
def load(p):
    with open(os.path.join(B,p), encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
fin = load("data/mapping_huvec_final.csv")
wt  = load("stage1/mapping_with_treatment.csv")
sk  = load("data/mapping_final_skov.csv")
print("rows final:", len(fin), "rows with_treat:", len(wt), "rows skov:", len(sk))

# arquivo_b uniqueness
c = collections.Counter(r['arquivo_b'] for r in fin)
dups = {k:v for k,v in c.items() if v>1}
print("dup arquivo_b in final:", len(dups), list(dups.items())[:5])

# status distribution
print("status final:", collections.Counter(r['status'] for r in fin))
print("linha_celular final:", collections.Counter(r['linha_celular'] for r in fin))

# HUVEC set
hu = [r for r in fin if r['linha_celular']=='HUVEC']
print("HUVEC rows:", len(hu))
sv = [r for r in fin if r['linha_celular']=='SKOV']
print("SKOV rows:", len(sv))
other = [r for r in fin if r['linha_celular'] not in ('HUVEC','SKOV')]
print("other rows:", len(other), collections.Counter(r['status'] for r in other))

# grp_huvec fill
g_filled = [r for r in hu if r['grp_huvec'].strip()]
g_empty  = [r for r in hu if not r['grp_huvec'].strip()]
print("HUVEC with grp:", len(g_filled), "without:", len(g_empty))
for r in g_empty:
    print("  EMPTY:", r['arquivo_b'], r['stem_normalizado'], r['lote_canon'], r['resolucao'])

# any non-HUVEC got grp?
nh = [r for r in fin if r['linha_celular']!='HUVEC' and r['grp_huvec'].strip()]
print("non-HUVEC rows with grp_huvec:", len(nh))

# groups
groups = collections.Counter(r['grp_huvec'] for r in g_filled)
print("n_grupos:", len(groups))
sizes = sorted(groups.values())
import statistics
print("min/med/max/mean:", sizes[0], statistics.median(sizes), sizes[-1], round(sum(sizes)/len(sizes),2))
print("groups with 1 image:", sum(1 for s in sizes if s==1))

# split
sp = collections.Counter(r['split_huvec'] for r in g_filled)
print("split images:", sp, "sum:", sum(sp.values()))
print("pct:", {k: round(100*v/len(g_filled),1) for k,v in sp.items()})
gsplit = {}
for r in g_filled:
    gsplit.setdefault(r['grp_huvec'], set()).add(r['split_huvec'])
multi = {k:v for k,v in gsplit.items() if len(v)>1}
print("GROUPS IN >1 SPLIT:", len(multi), list(multi.items())[:5])
print("split groups:", collections.Counter(next(iter(v)) for v in gsplit.values()))

# timepoints per group
tpg = {}
for r in g_filled:
    tpg.setdefault(r['grp_huvec'], set()).add(r['timepoint_h'])
print("timepoints per group:", collections.Counter(len(v) for v in tpg.values()))
print("wells with >1 tp:", sum(1 for v in tpg.values() if len(v)>1), "/", len(tpg))
print("timepoint split:", collections.Counter((r['timepoint_h'], r['split_huvec']) for r in g_filled))

# lote_canon
print("canonical batches:", collections.Counter(r['lote_canon'] for r in g_filled))
print("trat:", collections.Counter(r['trat_huvec'] for r in g_filled))
print("resolucao:", collections.Counter(r['resolucao'] for r in hu))
