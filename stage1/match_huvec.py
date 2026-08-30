# -*- coding: utf-8 -*-
"""
stage1/match_huvec.py — Atribui (lote/experimento, tratamento) a cada imagem HUVEC anotada,
por identidade EXATA de arquivo (MD5) contra HUVEC-RAW.
It also answers the DECISIVE TEST: does the same well appear in >1 treatment/batch?
"""
import hashlib, os, json, csv, re
from collections import defaultdict, Counter

BD = os.environ.get("BANCO_A", "<banco_a>") + r"\HUVEC"
RAW = os.environ.get("BANCO_A", "<banco_a>") + r"\HUVEC-RAW"
IMG = (".tif", ".tiff", ".bmp", ".png")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def well_of(fn):
    stem = re.sub(r"\.(tiff?|png|bmp)$", "", fn, flags=re.IGNORECASE)
    m = re.match(r"^\s*([A-Fa-f])\s*(\d{1,2})\b", stem)
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else None


def index_raw():
    """hash -> list of (batch, treatment, timepoint, filename)."""
    idx = defaultdict(list)
    meta = []
    for dirpath, _, files in os.walk(RAW):
        rel = os.path.relpath(dirpath, RAW)
        if rel == ".":
            continue
        parts = rel.split(os.sep)
        batch = parts[0]
        # ultimo componente costuma ser o timepoint (0h/8h/12h/24h/0hr...)
        tp = parts[-1] if re.match(r"^\d+\s*hr?$", parts[-1], re.I) else None
        if tp and len(parts) >= 3:
            treatment = parts[-2] if len(parts) > 2 else None
        elif tp and len(parts) == 2:
            treatment = None  # lote achatado: batch/timepoint
        else:
            treatment = None
        for f in files:
            if f.lower().endswith(IMG) and "scale" not in f.lower():
                p = os.path.join(dirpath, f)
                try:
                    h = md5(p)
                except Exception:
                    continue
                rec = (batch, treatment, tp, f)
                idx[h].append(rec)
                meta.append(dict(batch=batch, treatment=treatment, tp=tp, f=f,
                                 well=well_of(f), h=h))
    return idx, meta


def main():
    print("Indexando HUVEC-RAW por hash...")
    idx, meta = index_raw()
    print(f"  arquivos RAW: {len(meta)}   hashes distintos: {len(idx)}")

    # ---------- DECISIVE TEST: does a well cross treatment/batch? ----------
    well_ctx = defaultdict(set)
    for m in meta:
        if m["well"]:
            well_ctx[m["well"]].add((m["batch"], m["treatment"]))
    multi = {w: v for w, v in well_ctx.items() if len(v) > 1}
    print(f"\n>>> DECISIVE TEST: wells appearing in >1 (batch,treatment): "
          f"{len(multi)}/{len(well_ctx)}")
    for w in sorted(multi)[:6]:
        ctxs = sorted(f"{b}/{t}" for b, t in multi[w])
        print(f"    {w}: {len(ctxs)} contextos -> {ctxs[:5]}")

    # combinacoes lote/tratamento existentes
    combos = sorted({(m["batch"], m["treatment"]) for m in meta})
    print(f"\nCombinations (batch, treatment): {len(combos)}")
    for b, t in combos:
        n = sum(1 for m in meta if m["batch"] == b and m["treatment"] == t)
        print(f"    {b} / {t}  -> {n} imgs")

    # ---------- casar Banco de dados/HUVEC ----------
    print("\nCasando Banco de dados/HUVEC por hash...")
    bd_map = {}
    stats = Counter()
    for dirpath, _, files in os.walk(BD):
        rel = os.path.relpath(dirpath, BD)
        for f in files:
            if f.lower().endswith(IMG) and "scale" not in f.lower():
                try:
                    h = md5(os.path.join(dirpath, f))
                except Exception:
                    continue
                hits = idx.get(h, [])
                key = (f"HUVEC/{rel}" if rel != "." else "HUVEC", f)
                if not hits:
                    stats["sem_match"] += 1
                    bd_map[key] = None
                else:
                    treats = {(b, t) for b, t, _, _ in hits}
                    if len(treats) == 1:
                        stats["match_unico"] += 1
                    else:
                        stats["match_multiplo_contexto"] += 1
                    bd_map[key] = hits
    print(f"  BD arquivos: {sum(stats.values())}  {dict(stats)}")

    # ---------- join ao mapping ----------
    rows = list(csv.DictReader(open("stage1/mapping_with_treatment.csv", encoding="utf-8")))
    join = Counter()
    out = []
    for r in rows:
        r2 = dict(r)
        r2["lote"] = ""
        r2["trat_huvec"] = ""
        r2["ctx_ambiguo"] = ""
        if r["linha_celular"] == "HUVEC":
            pasta = r["pasta_a"].strip()
            arq = r["arquivo_a"].strip()
            hits = bd_map.get((pasta, arq)) if pasta and arq else None
            if hits:
                ctxs = sorted({(b, t) for b, t, _, _ in hits})
                r2["lote"] = ctxs[0][0]
                r2["trat_huvec"] = ctxs[0][1] or ""
                if len(ctxs) > 1:
                    r2["ctx_ambiguo"] = "|".join(f"{b}/{t}" for b, t in ctxs)
                    join["ambiguo"] += 1
                else:
                    join["ok"] += 1
            else:
                join["sem_fonte"] += 1
        out.append(r2)

    with open("stage1/mapping_huvec_treatment.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"\n=== Join HUVEC anotado ===  {dict(join)}")
    hv = [r for r in out if r["linha_celular"] == "HUVEC" and r["lote"]]
    print("\nAnnotated by (batch, treatment):")
    for k, n in Counter((r["lote"], r["trat_huvec"]) for r in hv).most_common():
        print(f"    {k[0]} / {k[1] or '(sem trat)'}  -> {n}")

    with open("stage1/match_huvec.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_raw": len(meta), "n_raw_hashes": len(idx),
            "wells_multi_context": len(multi), "wells_total": len(well_ctx),
            "combos": [[b, t] for b, t in combos],
            "bd_stats": dict(stats), "join": dict(join),
        }, f, indent=2, ensure_ascii=False)
    print("\nSalvos: stage1/mapping_huvec_treatment.csv, stage1/match_huvec.json")


if __name__ == "__main__":
    main()
