# -*- coding: utf-8 -*-
"""
stage1/assign_treatment.py — assigns (experiment, treatment) to each annotated SKOV image,
por identidade EXATA de arquivo (MD5), e junta ao stage1/mapping_b_to_a.csv.
It also extracts CreationDateTime from the metadata.xml files to test the stability
of the acquisition order (via metadata, not pixels).
"""
import hashlib, os, json, csv, re
from collections import defaultdict, Counter

BD = os.environ.get("BANCO_A", "<banco_a>") + r"\SKOV"
P1 = os.environ.get("RAW_ARCHIVE_P1", "<raw_archive_p1>")
P2 = os.environ.get("RAW_ARCHIVE_P2", "<raw_archive_p2>")
TPS = ["0h", "24h", "48h", "72h"]

# normalisation of the treatment labels across P1/P2 and timepoints
TREAT_CANON = {
    "ct": "CT", "CT": "CT",
    "ptx": "PTX", "PTX": "PTX",
    "75": "GEO", "75ug": "GEO", "75geo": "GEO", "geo": "GEO", "GEO": "GEO",
    "75ptx": "GEO+PTX", "75PTX": "GEO+PTX", "75ug+PTX": "GEO+PTX", "75ug+ptx": "GEO+PTX",
    "carbo": "CARBO",
    "carbo_geo": "CARBO+GEO", "geo_carbo": "CARBO+GEO",
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def creation_dt(xml_path):
    try:
        x = open(xml_path, encoding="utf-8", errors="ignore").read(4000)
        m = re.search(r"<CreationDateTime>([^<]+)<", x)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def build_source_index():
    """hash -> dict(experiment, treatment_canon, treatment_raw, timepoint, filename, dt)."""
    idx = {}
    for exp, root in [("P1", P1), ("P2", P2)]:
        for tp in TPS:
            base = os.path.join(root, tp)
            if not os.path.isdir(base):
                continue
            for tr in os.listdir(base):
                trd = os.path.join(base, tr)
                if not os.path.isdir(trd):
                    continue
                canon = TREAT_CANON.get(tr, tr)
                for f in os.listdir(trd):
                    if f.lower().endswith((".tif", ".tiff")) and "scale" not in f.lower():
                        p = os.path.join(trd, f)
                        try:
                            h = md5(p)
                        except Exception:
                            continue
                        dt = creation_dt(p + "_metadata.xml")
                        idx[h] = dict(experiment=exp, treatment=canon, treatment_raw=tr,
                                      timepoint=tp, filename=f, dt=dt)
    return idx


def main():
    print("Indexing the P1/P2 sources (hash + metadata)...")
    src = build_source_index()
    print(f"  source files indexed: {len(src)}")

    # hash of each raw file of the SKOV bank -> source
    print("Matching Banco de dados/SKOV by hash...")
    bd_map = {}  # (pasta_rel, filename) -> source dict
    bd_counts = Counter()
    for tp in TPS:
        d = os.path.join(BD, tp)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".tif", ".tiff")) and "scale" not in f.lower():
                try:
                    h = md5(os.path.join(d, f))
                except Exception:
                    continue
                s = src.get(h)
                bd_map[(f"SKOV/{tp}", f)] = s
                bd_counts[(tp, s["experiment"], s["treatment"]) if s else (tp, "SEM_MATCH", "")] += 1

    matched = sum(1 for v in bd_map.values() if v)
    print(f"  BD files matched: {matched}/{len(bd_map)}")

    # junta ao stage1/mapping_b_to_a.csv
    rows = list(csv.DictReader(open("stage1/mapping_b_to_a.csv", encoding="utf-8")))
    out_rows = []
    join_stats = Counter()
    treat_by_line = defaultdict(Counter)
    for r in rows:
        cl = r["linha_celular"]
        pasta = r["pasta_a"].strip()
        arq = r["arquivo_a"].strip()
        exp = treat = dt = ""
        if cl == "SKOV" and pasta.startswith("SKOV/") and arq:
            s = bd_map.get((pasta, arq))
            if s:
                exp, treat, dt = s["experiment"], s["treatment"], s["dt"] or ""
                join_stats["skov_ok"] += 1
                treat_by_line["SKOV"][treat] += 1
            else:
                # named 0h treatment (75geo1_2 etc.) OR no match
                if arq and not arq.lower().startswith("snap"):
                    # file named after a treatment; the hash should match too, but
                    # if pasta_a="SKOV/0h" the bd_map already tried. Flag for review.
                    join_stats["skov_named_nohit"] += 1
                else:
                    join_stats["skov_nohit"] += 1
        elif cl == "SKOV":
            join_stats["skov_no_arquivo_fisico"] += 1
        r2 = dict(r)
        r2["experimento"] = exp
        r2["tratamento"] = treat
        r2["creation_dt"] = dt
        out_rows.append(r2)

    with open("stage1/mapping_with_treatment.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("\n=== Join ao mapping ===")
    for k, v in join_stats.items():
        print(f"  {k}: {v}")
    print("\n=== SKOV annotated by treatment (canonical) ===")
    for t, n in treat_by_line["SKOV"].most_common():
        print(f"  {t}: {n}")

    # ---- teste de estabilidade de ordem via CreationDateTime ----
    print("\n=== Estabilidade da ordem de aquisicao (CreationDateTime) ===")
    # for each (experiment, treatment, timepoint) sort by dt and list the snap ids
    order = defaultdict(list)
    for s in src.values():
        if s["dt"]:
            m = re.search(r"(\d+)", s["filename"])
            snap = int(m.group(1)) if m else -1
            order[(s["experiment"], s["treatment"], s["timepoint"])].append((s["dt"], snap, s["filename"]))
    # example: P1 CT at each timepoint, ordered by time
    for exp, tr in [("P1", "CT"), ("P1", "PTX"), ("P2", "CT")]:
        print(f"\n  {exp}/{tr} — snap ids in temporal order, by timepoint:")
        for tp in TPS:
            lst = sorted(order.get((exp, tr, tp), []))
            if lst:
                snaps = [s for _, s, _ in lst]
                print(f"    {tp}: {snaps}")

    with open("stage1/assign_treatment.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_source": len(src), "n_bd": len(bd_map), "n_matched": matched,
            "bd_counts": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in bd_counts.items()},
            "join_stats": dict(join_stats),
            "skov_treatment_dist": dict(treat_by_line["SKOV"]),
        }, f, indent=2, ensure_ascii=False)
    print("\nSalvos: stage1/mapping_with_treatment.csv, stage1/assign_treatment.json")


if __name__ == "__main__":
    main()
