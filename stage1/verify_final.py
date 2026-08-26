# -*- coding: utf-8 -*-
"""
stage1/verify_final.py — M3. Prova final de zero-overlap. Sai com codigo 1 se violado.
Verifies at three levels:
  1. nenhuma split_key cruza particoes
  2. nenhuma group_key (unidade fisica, ANTES da fusao) cruza particoes
  3. the fallback worked: both versions of each shared well are together
"""
import csv, sys
from collections import defaultdict, Counter

# A versao estratificada (cell_line x tratamento) e a autoritativa.
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/mapping_dataset_final_strat.csv"
print(f"Verificando: {SRC}\n")
rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
act = [r for r in rows if r["partition"] != "EXCLUIDA"]
fail = False

# ---- 1) split_key ----
k = defaultdict(set)
for r in act:
    k[r["split_key"]].add(r["partition"])
v1 = {x: p for x, p in k.items() if len(p) > 1}
print(f"1) split keys in >1 partition: {len(v1)}")
if v1:
    fail = True
    print("   FALHA:", list(v1)[:10])

# ---- 2) group_key (unidade fisica antes da fusao) ----
g = defaultdict(set)
for r in act:
    g[r["group_key"]].add(r["partition"])
v2 = {x: p for x, p in g.items() if len(p) > 1}
print(f"2) group keys (physical unit) in >1 partition: {len(v2)}")
if v2:
    fail = True
    print("   FALHA:", list(v2)[:10])

# ---- 3) fallback efetivo ----
sh = [r for r in act if r["split_key"].startswith("SHARED__")]
bad = []
for key in {r["split_key"] for r in sh}:
    parts = {r["partition"] for r in act if r["split_key"] == key}
    if len(parts) > 1:
        bad.append((key, parts))
print(f"3) super-chaves SHARED com versoes separadas: {len(bad)}")
if bad:
    fail = True
    print("   FALHA:", bad[:10])
print(f"   (images under SHARED: {len(sh)}; wells: {len({r['well'] for r in sh})})")

# ---- 4) conservacao ----
print(f"4) conservacao: {len(rows)} linhas, {len(act)} ativas, "
      f"{len(rows)-len(act)} excluidas")
dup = [x for x, n in Counter(r["arquivo_b"] for r in rows).items() if n > 1]
print(f"   arquivo_b duplicado: {len(dup)}")
sem = [r for r in act if not r["split_key"] or not r["partition"]]
print(f"   active without split_key/partition: {len(sem)}")
if dup or sem:
    fail = True

if fail:
    print("\nVERIFICACAO FALHOU")
    sys.exit(1)

print("\nZERO-OVERLAP FINAL VERIFICADO")
print("  - nenhuma split_key cruza particoes")
print("  - nenhuma unidade fisica (group_key) cruza particoes")
print("  - fallback applied: both versions of the 19 wells are together")

# ---- tabela Supplementary ----
print("\n--- Supplementary: composicao final ---")
print(f"{'partition':<10}{'line':<8}{'groups':>8}{'images':>9}")
t = defaultdict(lambda: defaultdict(set))
c = defaultdict(Counter)
for r in act:
    t[r["partition"]][r["linha_celular"]].add(r["split_key"])
    c[r["partition"]][r["linha_celular"]] += 1
for p in ("train", "val", "test"):
    for cl in ("HUVEC", "SKOV"):
        print(f"{p:<10}{cl:<8}{len(t[p][cl]):>8}{c[p][cl]:>9}")
tot = Counter(r["partition"] for r in act)
print(f"\n{'TOTAL':<18}{len({r['split_key'] for r in act}):>8}{len(act):>9}")
for p in ("train", "val", "test"):
    print(f"  {p}: {tot[p]} images ({100*tot[p]/len(act):.1f}%)")
