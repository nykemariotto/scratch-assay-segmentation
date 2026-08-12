# -*- coding: utf-8 -*-
"""
stage4/recover_missing_baselines.py — looks for a recoverable t0 among the raw TIFFs,
for the series currently outside the paired analysis for lack of a baseline.

It copies and changes NOTHING: it only DIAGNOSES whether a plausible raw file
exists. The decision to incorporate comes later, with the list in hand.

Estrategia de busca, em ordem de confianca:
  1. indice cru cacheado (stage4/cache_raw_index.json): pastas <lote>\\<tp>\\<arquivo>,
     so a t0 is a file in the '0h'/'0hr' folder whose name contains the well.
  2. data/hash_cache_huvec.json: mesma ideia, com metadados ja parseados.
  3. a report of why the series lost its t0 (never existed / judged invalid /
     removed as a test image), because that changes what 'recover' means.
"""
import csv, json, os, re
from collections import defaultdict


def L(f):
    return list(csv.DictReader(open(f, encoding="utf-8-sig")))


C = L("stage4/closure_final_por_serie.csv")
A = L("data/whst_areas_final.csv")
H = {r["whst_input_file"]: r for r in L("data/visual_triage.csv")}
corr = L("whst_input/correspondencia.csv")
QUAR = "whst_output/_removed_out_of_scope/MANIFESTO.csv"
removidas = {r["whst_input_file"] for r in L(QUAR)} if os.path.isfile(QUAR) else set()

bys = defaultdict(list)
for r in A:
    bys[r["series_key"]].append(r)

sb = [r for r in C if r["motivo"] == "sem_baseline"]
print(f"series with no baseline: {len(sb)}\n")

# ---------- indice cru ----------
idx = []
if os.path.isfile("stage4/cache_raw_index.json"):
    d = json.load(open("stage4/cache_raw_index.json", encoding="utf-8"))
    idx += [(x.get("rel", ""), x.get("f", "")) for x in d.get("imgs", [])]
if os.path.isfile("data/hash_cache_huvec.json"):
    d = json.load(open("data/hash_cache_huvec.json", encoding="utf-8"))
    for grp in ("raw", "bd"):
        idx += [(x.get("rel", ""), x.get("name", "")) for x in d.get(grp, [])]
idx = [(a, b) for a, b in idx if b]
print(f"arquivos no indice cru: {len(idx)}\n")

T0DIR = re.compile(r"(^|[\\/])0\s*h(r|rs)?([\\/]|$)", re.I)
T0NAME = re.compile(r"(^|[ _])0\s*h(r|rs)?([ _.]|$)", re.I)


def parse_au(au):
    """analysis_unit -> (lote, tratamento, poco)."""
    p = au.split("||") if "||" in au else au.split("|")
    p = (p + ["", "", ""])[:3]
    return p[0], p[1], p[2]


def norm(s):
    """normalise for folder comparison: no accents or punctuation, lowercase."""
    s = (s.replace("ç", "c").replace("ã", "a").replace("á", "a").replace("â", "a")
          .replace("é", "e").replace("ê", "e").replace("í", "i").replace("ó", "o")
          .replace("ô", "o").replace("õ", "o").replace("ú", "u"))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def campo_do_nome(f):
    """token final 1/2 do nome do arquivo cru = campo."""
    s = re.sub(r"\.(tif|tiff)$", "", f, flags=re.I).strip()
    t = [x for x in re.split(r"[ _]+", s) if x in ("1", "2")]
    return t[-1] if t else ""


def procura(au, campo):
    """t0 cru do MESMO lote/tratamento/poco/campo.

    Casar so pelo poco e errado: o mesmo poco (ex.: 'A1') existe em varios
    batches, and a t0 from another batch is another experiment — worse than having
    no baseline. The folder is therefore required to contain the batch (and the
    treatment,
    quando ele aparece como subpasta) alem do poco e do campo.
    """
    lote, trat, well = parse_au(au)
    nl, nt, wl = norm(lote), norm(trat), well.lower()
    out = []
    for rel, f in idx:
        # componentes do caminho, comparados por IGUALDADE — 'originais' e
        # 'Originais (1)' sao lotes DIFERENTES e casariam por substring.
        comps = [norm(c) for c in re.split(r"[\\/]+", rel) if c]
        if nl and nl not in comps:                # lote tem de bater exatamente
            continue
        if nt and nt != "none" and nt not in comps:
            continue                              # tratamento, quando existir
        fl = f.lower()
        if not re.match(rf"^{re.escape(wl)}[ _.]", fl):
            continue
        if not (T0DIR.search(rel) or T0NAME.search(fl)):
            continue
        c = campo_do_nome(f)
        if campo and c and str(campo) != c:       # campo tem de bater
            continue
        out.append((rel, f))
    # remove duplicatas de caminho
    seen, uniq = set(), []
    for a, b in out:
        if (a, b) not in seen:
            seen.add((a, b)); uniq.append((a, b))
    return uniq


linhas = []
for r in sorted(sb, key=lambda r: (r["cell_line"], r["analysis_unit"])):
    sk = r["series_key"]
    rs = bys[sk]
    au = r["analysis_unit"]
    _, _, well = parse_au(au)
    t0 = [x for x in rs if int(x["timepoint_h"]) == 0]
    if t0:
        k = t0[0]["whst_input_file"]
        causa = ("removido: imagem-teste (fora do escopo)" if k in removidas
                 else f"t0 presente porem invalido ({H[k]['categoria']})")
    else:
        causa = "t0 ausente do pipeline"
    achados = procura(au, r["campo"])
    linhas.append({"series_key": sk, "cell_line": r["cell_line"], "analysis_unit": au,
                   "campo": r["campo"], "well": well,
                   "timepoints_atuais": r["timepoints"], "causa": causa,
                   "n_candidatos_crus": len(achados),
                   "candidatos": " | ".join(f"{a}\\{b}" for a, b in achados[:6])})
    print(f"  [{r['cell_line']}] {au[:34]:<36} c{r['campo']:<4} tps={r['timepoints']}")
    print(f"      causa: {causa}")
    if achados:
        print(f"      candidatos crus t0 ({len(achados)}):")
        for a, b in achados[:6]:
            print(f"        {a}\\{b}")
    else:
        print("      NENHUM candidato cru (mesmo lote/tratamento/poco/campo)")

with open("stage4/baselines_recuperaveis.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
    w.writeheader(); w.writerows(linhas)

n_ok = sum(1 for x in linhas if x["n_candidatos_crus"])
print(f"\n=== RESUMO ===")
print(f"  series with no baseline       : {len(linhas)}")
print(f"  com candidato cru encontrado  : {n_ok}")
print(f"  with no candidate             : {len(linhas)-n_ok}")
from collections import Counter
print(f"  por causa: {dict(Counter(x['causa'] for x in linhas))}")
print(f"\nSalvo: stage4/baselines_recuperaveis.csv")
