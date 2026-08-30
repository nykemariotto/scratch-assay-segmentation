# -*- coding: utf-8 -*-
"""
stage1/investigate_grouping.py — Escolha da chave de agrupamento do split, por METADATA.
It uses no pixels. Adapted to the real stage1/mapping_b_to_a.csv (columns: linha_celular,
well_campo, pasta_a, timepoint_h, arquivo_a, status).

It decomposes each image into cohort / treatment / field / well / snap / timepoint and
responde, por coorte, as perguntas decisivas de leakage.
"""
import csv
import re
from collections import Counter, defaultdict

CSV = "stage1/mapping_b_to_a.csv"


def decompose(r):
    """Retorna dict com coorte, treatment, rep, field, well, snap, tp."""
    cl = r["linha_celular"]
    wc = r["well_campo"].strip()
    tp = r["timepoint_h"].strip()
    a = r["arquivo_a"].strip()
    out = {"cell": cl, "tp": tp, "raw": wc}

    if cl == "HUVEC":
        out["cohort"] = "HUVEC"
        out["treatment"] = None
        out["well"] = wc  # ex: A1
        # replicata de campo vem do nome do arquivo cru: "A1 0H 1" -> 1
        m = re.search(r"\b([A-Fa-f]\d+)\D+\d+\D+(\d+)", a)
        out["field"] = m.group(2) if m else None
        out["group_id"] = wc  # candidato: well
        return out

    # SKOV
    if wc.lower().startswith("snap"):
        out["cohort"] = "SKOV_snap"
        out["treatment"] = None
        m = re.match(r"snap[-_ ]?(\d+)", wc.lower())
        out["snap"] = f"snap-{int(m.group(1)):02d}" if m else wc
        out["field"] = out["snap"]
        out["group_id"] = out["snap"]
        return out

    # SKOV tratamento: ex "ct1_2", "75ug+ptx3_5", "75geo1_2", "ptx2_4"
    m = re.match(r"^(75ug\+ptx|75geo|ptx|ct)(\d+)_(\d+)$", wc)
    out["cohort"] = "SKOV_treat"
    if m:
        out["treatment"] = m.group(1)
        out["rep"] = m.group(2)
        out["field"] = m.group(3)
    else:
        out["treatment"] = wc
        out["rep"] = None
        out["field"] = None
    out["group_id"] = wc  # treatment+rep+field
    return out


def viability(ngroups):
    n_test = max(1, round(ngroups * 0.15))
    n_val = max(1, round(ngroups * 0.15))
    n_train = ngroups - n_test - n_val
    ok = n_train >= 3 and n_test >= 2 and n_val >= 2
    return n_train, n_val, n_test, ("OK" if ok else "FRACO")


def main():
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
            if r["status"] != "descartado_scale"]
    dec = [decompose(r) for r in rows]

    print(f"Data images: {len(dec)}  (2 scale images excluded)\n")

    for cohort in ["HUVEC", "SKOV_snap", "SKOV_treat"]:
        sub = [d for d in dec if d["cohort"] == cohort]
        print("=" * 72)
        print(f"COHORT: {cohort}   (n={len(sub)} images)")
        print("=" * 72)

        # timepoints presentes
        tps = Counter(d["tp"] for d in sub)
        print(f"Timepoints: {dict(sorted(tps.items(), key=lambda x: int(x[0])))}")

        # ---------- DECISIVE TEST 1: does id (field/well/snap) cross treatment? --------
        has_treat = any(d.get("treatment") for d in sub)
        if cohort == "SKOV_treat":
            # o "id" fino aqui e o numero de campo _N; ele cruza tratamentos?
            field_to_treat = defaultdict(set)
            for d in sub:
                if d.get("field"):
                    field_to_treat[d["field"]].add(d["treatment"])
            cross = {f: t for f, t in field_to_treat.items() if len(t) > 1}
            print(f"\n>>> DECISIVE: field number '_N' appearing in >1 treatment: "
                  f"{len(cross)}/{len(field_to_treat)}")
            for f, t in list(cross.items())[:6]:
                print(f"      campo _{f}: {sorted(t)}")
            print("    -> the key MUST be treatment+rep+field (the full well_campo)")
        else:
            print("\n>>> DECISIVE: this cohort has NO recoverable treatment dimension "
                  "(nenhum rotulo de tratamento no nome/pasta/COCO).")

        # ---------- DECISIVE TEST 2: same id at multiple timepoints? ----------
        id_to_tps = defaultdict(set)
        for d in sub:
            id_to_tps[d["group_id"]].add(d["tp"])
        multi_tp = {k: v for k, v in id_to_tps.items() if len(v) > 1}
        single_tp = {k: v for k, v in id_to_tps.items() if len(v) == 1}
        print(f"\n>>> mesmo id em >1 timepoint: {len(multi_tp)}/{len(id_to_tps)} ids")
        print(f"    (ids em 1 so timepoint: {len(single_tp)})")
        print(f"    distribuicao de #timepoints por id: "
              f"{dict(Counter(len(v) for v in id_to_tps.values()))}")

        # ---------- contagem de grupos por chave candidata ----------
        print("\nGrupos independentes por chave candidata:")
        keys = {}
        if cohort == "HUVEC":
            keys["well"] = len(set(d["well"] for d in sub))
            keys["well+field"] = len(set((d["well"], d["field"]) for d in sub))
        elif cohort == "SKOV_snap":
            keys["snap"] = len(set(d["snap"] for d in sub))
            keys["snap+timepoint"] = len(set((d["snap"], d["tp"]) for d in sub))
            keys["coorte inteira (1 grupo)"] = 1
        else:  # SKOV_treat
            keys["treatment+rep+field (well_campo)"] = len(set(d["group_id"] for d in sub))
            keys["treatment+rep"] = len(set((d["treatment"], d.get("rep")) for d in sub))
            keys["treatment"] = len(set(d["treatment"] for d in sub))
        for kname, ng in keys.items():
            nt, nv, nte, verd = viability(ng)
            print(f"  [{kname:<34}] {ng:>4} grupos -> "
                  f"train={nt} val={nv} test={nte}  {verd}")

        # images per group (for the smallest/largest)
        gsz = Counter(d["group_id"] for d in sub)
        szv = sorted(gsz.values())
        print(f"\nImagens por grupo (chave group_id): "
              f"min={szv[0]} mediana={szv[len(szv)//2]} max={szv[-1]}")
        print()

    # ---------- HUVEC: well cruza timepoint? (leakage classico) ----------
    print("=" * 72)
    print("NOTA HUVEC: o well A1 aparece em todos os timepoints por design.")
    print("Agrupar por WELL mantem todos os timepoints+campos do well juntos = seguro.")
    print("=" * 72)
    huv = [d for d in dec if d["cohort"] == "HUVEC"]
    well_tp = defaultdict(set)
    for d in huv:
        well_tp[d["well"]].add(d["tp"])
    print(f"wells: {len(well_tp)}  |  timepoints/well (distrib): "
          f"{dict(Counter(len(v) for v in well_tp.values()))}")


if __name__ == "__main__":
    main()
