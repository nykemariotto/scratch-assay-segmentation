# -*- coding: utf-8 -*-
"""Testa estabilidade da ordem de aquisicao dentro de (experimento, tratamento)
usando CreationDateTime do metadata Zeiss. Le o XML inteiro (bug anterior lia so
4000 chars). Isso avalia se o offset do snap dentro do bloco de tratamento
identifies the same field across timepoints (via metadata, not pixels)."""
import os, re, json
from collections import defaultdict

P1 = os.environ.get("RAW_ARCHIVE_P1", "<raw_archive_p1>")
P2 = os.environ.get("RAW_ARCHIVE_P2", "<raw_archive_p2>")
TPS = ["0h", "24h", "48h", "72h"]


def dt_of(xml):
    try:
        x = open(xml, encoding="utf-8", errors="ignore").read()
    except Exception:
        return None
    m = re.search(r"<CreationDateTime>([^<]+)<", x)
    if not m:
        return None
    return m.group(1).strip()


def parse_dt(s):
    # formatos: '08/10/2022 15:57:57' (dd/mm/yyyy) — retorna tupla ordenavel
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", s)
    if m:
        d, mo, y, H, M, S = map(int, m.groups())
        return (y, mo, d, H, M, S)
    return None


def main():
    rows = []
    for exp, root in [("P1", P1), ("P2", P2)]:
        for tp in TPS:
            base = os.path.join(root, tp)
            if not os.path.isdir(base):
                continue
            for tr in os.listdir(base):
                trd = os.path.join(base, tr)
                if not os.path.isdir(trd):
                    continue
                for f in os.listdir(trd):
                    if f.lower().endswith((".tif", ".tiff")) and "scale" not in f.lower():
                        xml = os.path.join(trd, f + "_metadata.xml")
                        dt = dt_of(xml)
                        m = re.search(r"(\d+)", f)
                        snap = int(m.group(1)) if m else -1
                        rows.append(dict(exp=exp, tr=tr, tp=tp, f=f, snap=snap,
                                         dt=dt, dtk=parse_dt(dt) if dt else None))
    got = sum(1 for r in rows if r["dt"])
    print(f"files: {len(rows)}  with CreationDateTime: {got}")

    # date per (exp, timepoint): confirms each timepoint is a distinct session
    print("\n=== date/time by session (exp, timepoint) ===")
    sess = defaultdict(list)
    for r in rows:
        if r["dtk"]:
            sess[(r["exp"], r["tp"])].append(r["dtk"])
    for k in sorted(sess):
        v = sorted(sess[k])
        print(f"  {k[0]} {k[1]:>3}: {len(v)} imgs  de {v[0]} a {v[-1]}")

    # dentro de (exp, tr, tp): snap na ordem temporal
    print("\n=== temporal order of the snaps within (exp, treatment, tp) ===")
    order = defaultdict(list)
    for r in rows:
        if r["dtk"]:
            order[(r["exp"], r["tr"], r["tp"])].append((r["dtk"], r["snap"]))
    # normaliza rotulos p/ agrupar CT variantes etc.
    canon = {"ct": "CT", "CT": "CT", "ptx": "PTX", "PTX": "PTX",
             "75": "GEO", "75ug": "GEO", "geo": "GEO",
             "75ptx": "GEO+PTX", "75PTX": "GEO+PTX", "75ug+PTX": "GEO+PTX",
             "carbo": "CARBO", "carbo_geo": "CARBO+GEO", "geo_carbo": "CARBO+GEO"}
    # for each (exp, canon_tr) show the snap sequence per timepoint
    by_ct = defaultdict(dict)
    for (exp, tr, tp), lst in order.items():
        seq = [s for _, s in sorted(lst)]
        by_ct[(exp, canon.get(tr, tr))][tp] = seq
    stable_report = {}
    for key in sorted(by_ct):
        print(f"\n  {key[0]}/{key[1]}:")
        seqs = by_ct[key]
        for tp in TPS:
            if tp in seqs:
                print(f"    {tp:>3}: {seqs[tp]}")
        # teste: a sequencia temporal de snaps e a mesma (ordenada) entre timepoints?
        common_tps = [tp for tp in TPS if tp in seqs]
        # check whether the snap at position i is the same across timepoints
        if len(common_tps) >= 2:
            minlen = min(len(seqs[tp]) for tp in common_tps)
            pos_consistent = all(
                len(set(seqs[tp][i] for tp in common_tps)) == 1
                for i in range(minlen)
            )
            # e se ordenar por snap crescente, a ordem temporal e crescente?
            temporal_eq_numeric = all(seqs[tp] == sorted(seqs[tp]) for tp in common_tps)
            stable_report[f"{key[0]}/{key[1]}"] = {
                "pos_consistent_across_tp": pos_consistent,
                "temporal_order_eq_numeric": temporal_eq_numeric,
                "timepoints": common_tps,
            }
            print(f"    -> same position = same snap across timepoints: {pos_consistent}")
            print(f"    -> ordem temporal == ordem numerica do snap: {temporal_eq_numeric}")

    with open("stage1/order_stability.json", "w", encoding="utf-8") as f:
        json.dump({"n": len(rows), "with_dt": got, "stability": stable_report},
                  f, indent=2, ensure_ascii=False)
    print("\nSalvo: stage1/order_stability.json")


if __name__ == "__main__":
    main()
