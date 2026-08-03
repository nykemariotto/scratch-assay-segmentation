# -*- coding: utf-8 -*-
"""
stage4/read_triage.py — reads the manual triage from whst_output/overlays_sorted/
and produces data/inspecao_visual.csv with {categoria, subtipo} deduced from the
FOLDER each overlay was placed in, mapping every file back to its original via
whst_output/overlays_sorted_map.csv.

Category from the folder (whatever is left in the ROOT = OK):
    raiz                 -> OK
    _SEG_RUIM/_super/    -> SEG_RUIM / super
    _SEG_RUIM/_sub/      -> SEG_RUIM / sub
    _IMG_INVALIDA/       -> IMG_INVALIDA
    _AMBIGUO/            -> AMBIGUO

Validacao (obrigatoria):
  - total de arquivos encontrados deve bater com o mapa;
  - each map entry must match EXACTLY 1 file (nothing went missing);
  - nenhum arquivo pode casar 2x (nada duplicou — ex.: copiado em vez de movido);
  - nenhum arquivo pode ficar de fora do mapa (orfao).
Casamento primario pelo nome; fallback pelo prefixo md5 embutido (__{md5}.jpg).
"""
import csv, os, re, sys
from collections import defaultdict

ROOT = os.environ.get("TRIAGE_ROOT", "whst_output/overlays_sorted")
MAP_CSV = os.environ.get("TRIAGE_MAP", "whst_output/overlays_sorted_map.csv")
OUT = os.environ.get("TRIAGE_OUT", "data/inspecao_visual.csv")
MD5_RE = re.compile(r"__([0-9a-f]{10})\.jpg$", re.I)


def categorize(rel_parts):
    """rel_parts = caminho relativo a ROOT, ja dividido. Retorna (cat, sub, warn)."""
    if len(rel_parts) == 1:
        return "OK", "", None
    top = rel_parts[0]
    if top == "_SEG_RUIM":
        if len(rel_parts) >= 3 and rel_parts[1] in ("_super", "_sub"):
            return "SEG_RUIM", rel_parts[1].lstrip("_"), None
        if len(rel_parts) == 2:
            return "SEG_RUIM", "", f"loose in _SEG_RUIM without _super/_sub: {rel_parts[-1]}"
        return "SEG_RUIM", rel_parts[1].lstrip("_"), f"subpasta inesperada em _SEG_RUIM: {rel_parts[1]}"
    if top == "_IMG_INVALIDA":
        return "IMG_INVALIDA", "", None
    if top == "_AMBIGUO":
        return "AMBIGUO", "", None
    return "?PASTA_DESCONHECIDA", "", f"folder not recognised: {top}"


def main():
    if not os.path.isfile(MAP_CSV):
        sys.exit(f"map not found: {MAP_CSV} (run stage4/build_overlays_sorted.py)")
    mp = list(csv.DictReader(open(MAP_CSV, encoding="utf-8-sig")))
    by_name = {r["sorted_basename"]: r for r in mp}
    by_md5 = {r["raw_md5"][:10]: r for r in mp}
    N = len(mp)                      # derived from the map (not hard-coded)
    assert len(by_name) == N, f"mapa tem {N} entradas"

    found, warns, orphans = [], [], []
    matched_key = defaultdict(list)   # sorted_basename from the map -> files that matched

    for dirpath, _, files in os.walk(ROOT):
        for fn in files:
            if not fn.lower().endswith(".jpg"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            parts = rel.split(os.sep)
            cat, sub, warn = categorize(parts)
            if warn:
                warns.append(warn)
            # casa pelo nome; fallback md5
            rec = by_name.get(fn)
            via = "nome"
            if rec is None:
                m = MD5_RE.search(fn)
                if m and m.group(1).lower() in by_md5:
                    rec = by_md5[m.group(1).lower()]; via = "md5"
            if rec is None:
                orphans.append(rel)
                continue
            matched_key[rec["sorted_basename"]].append(rel)
            found.append((rec, cat, sub, via))

    # ---- integridade ----
    n = len(found)
    missing = [k for k in by_name if k not in matched_key]
    dups = {k: v for k, v in matched_key.items() if len(v) > 1}

    print("=== INTEGRIDADE ===")
    print(f"  arquivos .jpg encontrados na arvore : {n + len(orphans)}")
    print(f"  casados com o mapa                  : {n}")
    print(f"  esperado (mapa)                     : {N}")
    print(f"  missing (in the map, no file)       : {len(missing)}")
    print(f"  duplicados (>1 arquivo p/ 1 entrada): {len(dups)}")
    print(f"  orfaos (arquivo fora do mapa)       : {len(orphans)}")
    for w in warns:
        print(f"  [aviso] {w}")
    for k in missing[:20]:
        print(f"  [faltando] {k}")
    for k, v in list(dups.items())[:20]:
        print(f"  [duplicado] {k} -> {v}")
    for o in orphans[:20]:
        print(f"  [orfao] {o}")

    ok_total = (n == N and not missing and not dups and not orphans)
    print(f"\n  SUM MATCHES {N} and NO LOSS/DUP/ORPHAN: {'YES' if ok_total else 'NO'}")

    # ---- escreve CSV (mesmo se houver problema: util p/ diagnostico) ----
    fields = ["sorted_basename", "categoria", "subtipo", "cell_line", "group_key",
              "timepoint_h", "campo", "raw_file_original", "raw_md5",
              "whst_input_file", "test_image", "match_via"]
    out_rows = []
    for rec, cat, sub, via in found:
        out_rows.append({"sorted_basename": rec["sorted_basename"], "categoria": cat,
                         "subtipo": sub, "cell_line": rec["cell_line"],
                         "group_key": rec["group_key"], "timepoint_h": rec["timepoint_h"],
                         "campo": rec["campo"], "raw_file_original": rec["raw_file_original"],
                         "raw_md5": rec["raw_md5"], "whst_input_file": rec["whst_input_file"],
                         "test_image": rec["test_image"], "match_via": via})
    out_rows.sort(key=lambda x: x["sorted_basename"])
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)

    # ---- resumo por categoria ----
    from collections import Counter
    cnt = Counter((r["categoria"], r["subtipo"]) for r in out_rows)
    print("\n=== CONTAGEM POR CATEGORIA ===")
    for (c, s), k in sorted(cnt.items()):
        print(f"  {c:<20} {s:<8} {k:>4}")
    print(f"\nSalvo: {OUT} ({len(out_rows)} linhas)")
    sys.exit(0 if ok_total else 1)


if __name__ == "__main__":
    main()
