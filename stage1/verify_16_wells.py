# -*- coding: utf-8 -*-
"""
stage1/verify_16_wells.py — Decisao (b): a chave composta ja separa 'Originais (1)' de
'originais' in the shared wells? And does the 0h count prove they are the same
fisicamente distintos?

Adaptado ao CSV real (data/mapping_huvec_final.csv):
  lote_canon | trat_huvec | well_campo | timepoint_h | campo | grp_huvec | split_huvec
"""
import csv
from collections import defaultdict, Counter

CSV = "data/mapping_huvec_final.csv"
LAB_A, LAB_B = "Originais (1)", "originais"

rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r.get("linha_celular") == "HUVEC"]
sub = [r for r in rows if r.get("lote_canon") in (LAB_A, LAB_B)]

print(f"Images from the two batches: {len(sub)}")
print(f"  {LAB_A}: {sum(1 for r in sub if r['lote_canon']==LAB_A)}")
print(f"  {LAB_B}: {sum(1 for r in sub if r['lote_canon']==LAB_B)}")

wells = defaultdict(lambda: defaultdict(list))
for r in sub:
    wells[r["well_campo"]][r["lote_canon"]].append(r)

shared = sorted(w for w, d in wells.items() if len(d) > 1)
print(f"\nWells present in BOTH batches: {len(shared)} -> {shared}")

# ---------------------------------------------------------------- TESTE 1
print("\n" + "=" * 72)
print("TEST 1: does the composite key separate the two batches?")
print("=" * 72)
all_separated = True
for w in shared:
    ga = {r["grp_huvec"] for r in wells[w][LAB_A]}
    gb = {r["grp_huvec"] for r in wells[w][LAB_B]}
    ov = ga & gb
    if ov:
        all_separated = False
        print(f"  Well {w}: SAME GROUP (X)  overlap={ov}")
    else:
        print(f"  Well {w}: SEPARADOS (OK)  {sorted(ga)} | {sorted(gb)}")

# ---------------------------------------------------------------- TESTE 2
print("\n" + "=" * 72)
print("TEST 2: field count at 0h (physically distinct wells?)")
print("=" * 72)
n_distinct = 0
for w in shared:
    ca = sum(1 for r in wells[w][LAB_A] if r["timepoint_h"] == "0")
    cb = sum(1 for r in wells[w][LAB_B] if r["timepoint_h"] == "0")
    diff = ca != cb
    n_distinct += diff
    tag = "DISTINTOS (0h difere)" if diff else "mesma contagem no 0h"
    print(f"  Well {w}: {{{LAB_A}: {ca}, {LAB_B}: {cb}}} -> {tag}")

# global count per timepoint (batch evidence, more robust than per well)
print("\n  TOTAL count by timepoint (evidence of the batch):")
for lab in (LAB_A, LAB_B):
    c = Counter(r["timepoint_h"] for r in sub if r["lote_canon"] == lab)
    print(f"    {lab:<16}: {dict(sorted(c.items(), key=lambda x: int(x[0])))}"
          f"  total={sum(c.values())}")

# the set of wells in each batch (plate layout)
sa = {r["well_campo"] for r in sub if r["lote_canon"] == LAB_A}
sb = {r["well_campo"] for r in sub if r["lote_canon"] == LAB_B}
print(f"\n  Wells {LAB_A} ({len(sa)}): {sorted(sa)}")
print(f"  Wells {LAB_B} ({len(sb)}): {sorted(sb)}")
print(f"  So em {LAB_A}: {sorted(sa-sb)}")
print(f"  So em {LAB_B}: {sorted(sb-sa)}")

# ---------------------------------------------------------------- particoes
print("\n" + "=" * 72)
print("CURRENT PARTITION SITUATION (what the fallback would change)")
print("=" * 72)
n_diff_part = 0
for w in shared:
    pa = {r["split_huvec"] for r in wells[w][LAB_A]}
    pb = {r["split_huvec"] for r in wells[w][LAB_B]}
    if pa != pb or (pa | pb) != (pa & pb):
        n_diff_part += 1
        print(f"  Well {w}: {LAB_A}->{sorted(pa)}  {LAB_B}->{sorted(pb)}")
print(f"\n  Wells whose two batches fall in different partitions: {n_diff_part}/{len(shared)}")
print("  (the fallback would force these into the same partition; only sensible if they are the SAME well)")

# ---------------------------------------------------------------- conclusao
print("\n" + "=" * 72)
print("CONCLUSAO")
print("=" * 72)
if all_separated:
    print("TEST 1: the composite key ALREADY separates the two batches in EVERY")
    print("        shared well (the batch is the first component of grp_huvec).")
    print(f"TEST 2: {n_distinct}/{len(shared)} wells have a different field count at 0h.")
    print("\n-> THE FALLBACK IS REDUNDANT for the goal of 'not splitting a well':")
    print("   the two batches already form naturally distinct groups.")
    print("   Applying it would merely MERGE distinct experiments, reducing effective")
    print("   independence with no gain in safety.")
else:
    print("The key does NOT separate on some well — see the overlaps in Test 1.")
