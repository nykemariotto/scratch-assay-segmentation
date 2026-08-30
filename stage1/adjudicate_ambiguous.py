# -*- coding: utf-8 -*-
"""
stage1/adjudicate_ambiguous.py — adjudication of the 12 AMBIGUO cases before the correction.

Fluxo:
  1) python stage1/adjudicate_ambiguous.py --template
       writes stage1/adjudication_ambiguous.csv with one row per AMBIGUO and the
       'decisao' column EMPTY, to be filled in while looking at inspect_ambiguous.png.
       Valores aceitos: OK | super | sub | invalida
  2) (voce preenche a coluna decisao)
  3) python stage1/adjudicate_ambiguous.py --apply
       valida o preenchimento, aplica em data/visual_triage.csv (categoria/subtipo),
       faz backup do arquivo anterior e manda re-rodar stage4/whst_series_analysis.py.

Without --apply nothing is modified. The script refuses invalid or incomplete decisions.
"""
import csv, os, shutil, sys

HUM = "data/visual_triage.csv"
AUTO = "data/whst_pass1_qc.csv"
ADJ = "stage1/adjudication_ambiguous.csv"
VALID = {"OK": ("OK", ""), "super": ("SEG_RUIM", "super"),
         "sub": ("SEG_RUIM", "sub"), "invalida": ("IMG_INVALIDA", "")}


def load():
    hum = list(csv.DictReader(open(HUM, encoding="utf-8-sig")))
    auto = {r["whst_input_file"]: r for r in csv.DictReader(open(AUTO, encoding="utf-8-sig"))}
    return hum, auto


def template():
    hum, auto = load()
    amb = [r for r in hum if r["categoria"] == "AMBIGUO"]
    if not amb:
        sys.exit("no AMBIGUOUS row in data/visual_triage.csv (already adjudicated?)")
    rows = []
    for r in sorted(amb, key=lambda r: (auto[r["whst_input_file"]]["analysis_unit"],
                                        int(r["timepoint_h"]))):
        a = auto[r["whst_input_file"]]
        rows.append({"analysis_unit": a["analysis_unit"], "timepoint_h": r["timepoint_h"],
                     "campo": r["campo"], "area_pct_auto": a["area_pct"],
                     "whst_input_file": r["whst_input_file"], "decisao": ""})
    with open(ADJ, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {ADJ} with {len(rows)} AMBIGUOUS rows.")
    print("Preencha a coluna 'decisao' com: OK | super | sub | invalida")
    print("Use o painel inspect_ambiguous.png (vermelho = o frame a adjudicar).")
    print("Then run: python stage1/adjudicate_ambiguous.py --apply")


def apply():
    if not os.path.isfile(ADJ):
        sys.exit(f"{ADJ} does not exist. Run first: python stage1/adjudicate_ambiguous.py --template")
    adj = list(csv.DictReader(open(ADJ, encoding="utf-8-sig")))
    faltando = [a for a in adj if not a["decisao"].strip()]
    invalidas = [a for a in adj if a["decisao"].strip() and a["decisao"].strip() not in VALID]
    if faltando:
        sys.exit(f"ABORTED: {len(faltando)} row(s) with no decision filled in.")
    if invalidas:
        for a in invalidas:
            print("  invalid:", repr(a["decisao"]), "em", a["whst_input_file"][:50])
        sys.exit(f"ABORTED: use only {sorted(VALID)}")

    hum, _ = load()
    dec = {a["whst_input_file"]: a["decisao"].strip() for a in adj}
    n = 0
    for r in hum:
        if r["whst_input_file"] in dec:
            cat, sub = VALID[dec[r["whst_input_file"]]]
            r["categoria"], r["subtipo"] = cat, sub
            n += 1
    assert n == len(adj), f"aplicadas {n} de {len(adj)}"
    shutil.copy2(HUM, HUM + ".pre_adjudicacao.bak")
    with open(HUM, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hum[0].keys())); w.writeheader(); w.writerows(hum)
    from collections import Counter
    print(f"aplicadas {n} adjudicacoes em {HUM} (backup: {HUM}.pre_adjudicacao.bak)")
    print("  distribution:", dict(Counter(a["decisao"].strip() for a in adj)))
    print("\nNOW RE-RUN (required, to close the series verdicts):")
    print("  python stage4/whst_series_analysis.py")
    print("  python stage4/build_correction_worklist.py")


if __name__ == "__main__":
    if "--apply" in sys.argv:
        apply()
    elif "--template" in sys.argv:
        template()
    else:
        print(__doc__)
