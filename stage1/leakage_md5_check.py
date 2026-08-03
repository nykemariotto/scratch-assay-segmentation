# -*- coding: utf-8 -*-
"""
stage1/leakage_md5_check.py — Checagem de leakage por DUPLICATA DE CONTEUDO (MD5) no
the WHOLE dataset (train+val+test). For each raw file of bank A that feeds
the dataset, it computes the MD5, groups by MD5, and for groups with >1 image checks whether
caem em particoes diferentes. Arquivo fisico identico em particoes diferentes =
LEAKAGE.
"""
import csv, os, hashlib, json
from collections import defaultdict

BA = os.environ.get("BANCO_A", "<banco_a>")
rows = [x for x in csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8"))
        if x["partition"] != "EXCLUIDA" and x["arquivo_a"].strip()]
print(f"active images with a raw source: {len(rows)} (those without arquivo_a have no raw to hash)")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# hash 1x por caminho fisico
path_md5 = {}
n = 0
for x in rows:
    p = os.path.join(BA, x["pasta_a"].replace("/", os.sep), x["arquivo_a"])
    if p not in path_md5:
        path_md5[p] = md5(p) if os.path.exists(p) else None
    n += 1
    if n % 200 == 0:
        print(f"  {n}/{len(rows)}  (caminhos unicos hasheados: {len(path_md5)})")

miss = [p for p, h in path_md5.items() if h is None]
print(f"\ncaminhos crus unicos: {len(path_md5)} | inexistentes: {len(miss)}")

# group by MD5 -> list of images (partition, group_key, arquivo_b, arquivo_a, path)
by_md5 = defaultdict(list)
for x in rows:
    p = os.path.join(BA, x["pasta_a"].replace("/", os.sep), x["arquivo_a"])
    h = path_md5.get(p)
    if h:
        by_md5[h].append({"part": x["partition"], "grp": x["group_key"],
                          "b": x["arquivo_b"], "a": x["arquivo_a"], "path": p})

multi = {h: v for h, v in by_md5.items() if len(v) > 1}
print(f"\n=== GRUPOS DE MD5 IDENTICO (conteudo cru duplicado) ===")
print(f"  MD5 distintos no dataset: {len(by_md5)}")
print(f"  grupos de MD5 com >1 imagem: {len(multi)}")

cross = []
for h, v in multi.items():
    parts = {r["part"] for r in v}
    if len(parts) > 1:
        cross.append((h, v, parts))

print(f"\n=== LEAKAGE: identical MD5 in >1 PARTITION ===")
print(f"  MD5 groups crossing a partition: {len(cross)}")
if cross:
    print("  !!! LEAKAGE DETECTADO !!!")
    for h, v, parts in cross:
        print(f"    md5={h[:12]} particoes={sorted(parts)}")
        for r in v:
            print(f"        [{r['part']}] {r['a']}  grupo={r['grp']}  ({r['b'][:30]})")
else:
    print("  none -> NO LEAKAGE from content duplication")

# detail of the MD5 groups (all of them, even intra-partition) for the report
print(f"\n=== detalhe: grupos de MD5 duplicado (todos) ===")
for h, v in multi.items():
    parts = sorted({r["part"] for r in v})
    grps = sorted({r["grp"] for r in v})
    same = "MESMA particao/grupo" if len(parts) == 1 and len(grps) == 1 else \
           ("same partition, different groups" if len(parts) == 1 else "CROSSES PARTITION")
    print(f"  md5={h[:12]} x{len(v)} particoes={parts} grupos={len(grps)} -> {same}")
    for r in v:
        print(f"      [{r['part']}] {r['a']}")

json.dump({
    "n_ativas_com_cru": len(rows), "n_caminhos_unicos": len(path_md5),
    "n_inexistentes": len(miss), "n_md5_distintos": len(by_md5),
    "n_grupos_md5_duplicado": len(multi), "n_cruzam_particao": len(cross),
    "cross_particao": [{"md5": h, "particoes": sorted(parts),
                        "arquivos": [r["a"] for r in v], "grupos": sorted({r["grp"] for r in v})}
                       for h, v, parts in cross],
    "duplicados_intra": [{"md5": h, "particao": sorted({r["part"] for r in v})[0],
                          "arquivos": [r["a"] for r in v], "grupos": sorted({r["grp"] for r in v})}
                         for h, v in multi.items() if len({r["part"] for r in v}) == 1],
}, open("stage1/leakage_md5_check.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nSalvo: stage1/leakage_md5_check.json")
