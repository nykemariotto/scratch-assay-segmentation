# -*- coding: utf-8 -*-
"""
stage4/whst_series_analysis.py — tasks 1-3 over the blinded visual triage.

Central insight: the closure fraction is a ratio, so a CONSTANT multiplicative
bias within a series cancels:
    (k*a0 - k*at)/(k*a0) = (a0-at)/a0
So an entirely over-segmented series can still give the correct closure.

T1 — classifies each series (series_key) as CONSISTENT (all over / all OK / all
     under) or MIXED; for the consistent ones it computes closure and tests
     biological plausibility (baseline present, monotonic, within [0,1], no absurd
     jump). If consistent-over gives a plausible closure, the bias cancels.
T2 — severity: for each 'over' image, estimates the RELATIVE EXCESS of the WHST
     area against whatever reference is available, in order of preference:
       ref1 = sibling field scored OK (same well/unit, same tp, other field)
       ref2 = same condition (batch|treatment) OK, same tp, other unit (median)
       ref3 = median of the batch at the same tp (OK)
     excess = area_over/ref - 1. Binned as mild (<20%) / moderate / severe (>50%).
T3 — correction list = MIXED series + consistent series with an IMPLAUSIBLE
     closure + broken baselines + severe cases outside a usable series.
     Consistently over-segmented series WITH a plausible closure stay out, and
     that is declarable.

Outputs:
  - data/inspecao_visual.csv  (rewritten, + classe_validade and analysis columns)
  - data/whst_series_analysis.csv  (one row per series)
"""
import csv, os
from collections import defaultdict
import statistics as st

HUM = "data/inspecao_visual.csv"
AUTO = "data/whst_pass1_qc.csv"

# ---------- join ----------
hum = {r["whst_input_file"]: r for r in csv.DictReader(open(HUM, encoding="utf-8-sig"))}
auto = {r["whst_input_file"]: r for r in csv.DictReader(open(AUTO, encoding="utf-8-sig"))}
assert set(hum) == set(auto)


def hcat(r):
    return r["categoria"] if r["categoria"] != "SEG_RUIM" else "SEG_" + r["subtipo"]


def parse_gk(gk):
    """(batch, treatment, well/field). HUVEC uses '||', SKOV uses '|'."""
    parts = gk.split("||") if "||" in gk else gk.split("|")
    parts = (parts + ["", "", ""])[:3]
    return parts[0], parts[1], parts[2]


# AREA: uses the corrected one when it exists (data/whst_areas_final.csv), otherwise
# the automatic one. Frames with status 'invalida' have no area and are treated as
# IMG_INVALIDA.
FINAL = {}
if os.path.isfile("data/whst_areas_final.csv"):
    for _r in csv.DictReader(open("data/whst_areas_final.csv", encoding="utf-8-sig")):
        FINAL[_r["whst_input_file"]] = _r
    print(f"[areas] using data/whst_areas_final.csv "
          f"({sum(1 for v in FINAL.values() if v['fonte_area'].startswith('corrigida'))} corrigidas)")

IM = []
for k in hum:
    a, h = auto[k], hum[k]
    lote, trat, well = parse_gk(a["analysis_unit"])
    fin = FINAL.get(k)
    if fin and fin["area_pct_final"] != "":
        area = float(fin["area_pct_final"])
    else:
        area = float(a["area_pct"])
    IM.append(dict(k=k, sk=a["series_key"], au=a["analysis_unit"], lote=lote, trat=trat,
                   well=well, campo=a["campo"], tp=int(a["timepoint_h"]),
                   area=area, cl=a["cell_line"], hc=hcat(h),
                   fonte=(fin["fonte_area"] if fin else "automatica")))

by_sk = defaultdict(list)
for r in IM:
    by_sk[r["sk"]].append(r)


# ---------- area per (series, tp): median if duplicated ----------
def area_by_tp(rs):
    d = defaultdict(list)
    for r in rs:
        d[r["tp"]].append(r["area"])
    return {tp: st.median(v) for tp, v in d.items()}


# ---------- pattern classification ----------
def pattern(rs):
    cats = set(r["hc"] for r in rs)
    if cats <= {"IMG_INVALIDA"}:
        return "TODA_INVALIDA"
    has_inval = "IMG_INVALIDA" in cats
    core = cats - {"AMBIGUO", "IMG_INVALIDA"}
    if core == {"OK"}:
        base = "CONSISTENTE_OK"
    elif core == {"SEG_super"}:
        base = "CONSISTENTE_SUPER"
    elif core == {"SEG_sub"}:
        base = "CONSISTENTE_SUB"
    elif len(core) == 0:
        base = "SO_AMBIGUO_INVALIDA"
    else:
        base = "MISTO"
    if has_inval:                      # marca tambem MISTO+INVALIDA
        base += "+INVALIDA"
    return base


# ---------- closure + plausibility ----------
def closure(abt):
    """abt = {tp: area}. Returns (has_baseline, ordered_seq[(tp,closure)], plausible, reason)."""
    if 0 not in abt:
        return False, [], False, "sem_baseline"
    a0 = abt[0]
    tps = sorted(abt)
    if len(tps) < 2:
        return True, [(0, 0.0)], False, "serie_1_ponto"
    if a0 <= 0:
        return True, [], False, "baseline_zero"
    seq = [(tp, (a0 - abt[tp]) / a0) for tp in tps]   # closure(0)=0
    cvals = [c for tp, c in seq if tp > 0]
    # range
    range_ok = all(-0.05 <= c <= 1.05 for c in cvals)
    # non-decreasing monotonicity, with a tolerance
    tol = 0.10
    seqvals = [c for _, c in seq]
    mono_ok = all(seqvals[i] >= seqvals[i - 1] - tol for i in range(1, len(seqvals)))
    # absurd jump
    no_absurd = all(abs(c) <= 1.5 for c in cvals)
    plaus = range_ok and mono_ok and no_absurd
    motivo = []
    if not range_ok: motivo.append("fora_[0,1]")
    if not mono_ok: motivo.append("nao_monotonica")
    if not no_absurd: motivo.append("salto_absurdo")
    return True, seq, plaus, ("+".join(motivo) if motivo else "plausivel")


# ---------- T2: severity reference ----------
def ref_area(img, target_ok_only=True):
    """estimates a reference area (approximate truth) for an over-segmented image."""
    same_tp = [r for r in IM if r["tp"] == img["tp"] and r["k"] != img["k"]]
    def ok(rs): return [r for r in rs if r["hc"] == "OK"]
    # ref1: same unit (well), other field, OK
    r1 = ok([r for r in same_tp if r["au"] == img["au"] and r["campo"] != img["campo"]])
    if r1: return st.median([r["area"] for r in r1]), "irmao_campo"
    # ref2: same condition (batch, treatment), other unit, OK
    r2 = ok([r for r in same_tp if r["lote"] == img["lote"] and r["trat"] == img["trat"]])
    if r2: return st.median([r["area"] for r in r2]), "cond_ok"
    # ref3: same batch, OK, same tp
    r3 = ok([r for r in same_tp if r["lote"] == img["lote"]])
    if r3: return st.median([r["area"] for r in r3]), "lote_ok"
    return None, "sem_referencia"


# ============================ REGRA_INVALIDOS ============================
# Declared a priori in PROTOCOLO_CORRECAO_MANUAL.md section 3.1:
#   R1  invalid at t>0             -> discard that timepoint only; the series goes on
#   R2  invalid at t0 with sibling -> use the sibling field as the baseline
#   R3  invalid at t0 no sibling   -> series without a baseline, out of the pairing
# (the general logic: a missing baseline kills the series; a missing timepoint does not)

# ---- frames to be DECIDED during the correction itself ----
# The operator wrote "noisy image but segmentable": the image is poor, but the
# note does not say whether the segmentation is good, so the category cannot be
# inferred. Rather than deciding for them on paper, they enter the worklist: if the
# contour can be traced they are valid; if not, the macro records 'pulada' and the
# decision is anchored in the act of trying.
RESOLVER = set()
if os.path.isfile("data/annotation_report.csv"):
    for _r in csv.DictReader(open("data/annotation_report.csv", encoding="utf-8-sig")):
        if _r.get("motivo_classificado") == "ruidosa_recuperavel":
            RESOLVER.add(_r["whst_input_file"])

# index for R2: (analysis_unit, campo) -> area at t0, non-invalid frames only
t0_por_campo = defaultdict(dict)
for r in IM:
    if r["tp"] == 0 and r["hc"] != "IMG_INVALIDA":
        t0_por_campo[r["au"]][r["campo"]] = r["area"]

# ============================ EXECUCAO ============================
series = {}
for sk, rs in by_sk.items():
    rs_val = [r for r in rs if r["hc"] != "IMG_INVALIDA"]     # R1 aplicado aqui
    n_inval = len(rs) - len(rs_val)
    abt = area_by_tp(rs_val)
    pat = pattern(rs)
    emprestado = ""
    if not rs_val:
        pat = "TODA_INVALIDA"
    elif 0 not in abt and any(r["tp"] == 0 for r in rs):
        # t0 existed but was invalid -> R2/R3
        au, campo = rs[0]["au"], rs[0]["campo"]
        irmaos = {c: a for c, a in t0_por_campo.get(au, {}).items() if c != campo}
        if irmaos:
            c_irm = sorted(irmaos)[0]
            abt[0] = irmaos[c_irm]                            # R2: empresta
            emprestado = f"sim(c{c_irm})"
    has_base, seq, plaus, motivo = closure(abt)
    series[sk] = dict(sk=sk, n=len(rs), pat=pat, has_base=has_base, plaus=plaus,
                      motivo=motivo, seq=seq, cl=rs[0]["cl"], lote=rs[0]["lote"],
                      cats=[r["hc"] for r in rs], n_inval=n_inval,
                      emprestado=emprestado)

# ---- T1 report ----
from collections import Counter
print("=== TASK 1: over-segmentation pattern per series (n=%d series) ===" % len(series))
pat_count = Counter(s["pat"] for s in series.values())
for p, c in sorted(pat_count.items(), key=lambda x: -x[1]):
    print(f"  {p:<24} {c:>3}")
print()
print("=== closure of the CONSISTENT series (does the bias cancel?) ===")
for pat in ("CONSISTENTE_SUPER", "CONSISTENTE_OK", "CONSISTENTE_SUB"):
    ss = [s for s in series.values() if s["pat"] == pat]
    if not ss: continue
    plaus = sum(1 for s in ss if s["plaus"])
    nobase = sum(1 for s in ss if not s["has_base"] or s["motivo"].startswith("sem_baseline") or s["motivo"] == "serie_1_ponto")
    print(f"  {pat:<20} n={len(ss):>3}  closure_plausivel={plaus:>3}  "
          f"implausible={len(ss)-plaus-0:>3}  (of which no-baseline/1pt={nobase})")
    # detail of the implausible ones
    for s in ss:
        if not s["plaus"] and s["has_base"] and s["motivo"] != "serie_1_ponto":
            cv = ", ".join(f"{tp}h:{c:+.2f}" for tp, c in s["seq"])
            print(f"      [implaus] {s['sk'][:44]:<44} {s['motivo']:<22} [{cv}]")
print()
# exemplos de consistente-super PLAUSIVEL (viés cancela -> usavel)
print("  examples of CONSISTENTE_SUPER + plausible closure (usable without correction):")
ex = [s for s in series.values() if s["pat"] == "CONSISTENTE_SUPER" and s["plaus"]]
for s in ex[:6]:
    cv = ", ".join(f"{tp}h:{c:+.2f}" for tp, c in s["seq"])
    print(f"      {s['sk'][:46]:<46} [{cv}]")
print(f"  total consistently-over and usable (the bias cancels): {len(ex)}")

# ---- T2: severity of the over-segmented ----
print("\n=== TASK 2: severity (relative excess) of the OVER-segmented images ===")
supers = [r for r in IM if r["hc"] == "SEG_super"]
sev = []
for r in supers:
    ref, meth = ref_area(r)
    exc = (r["area"] / ref - 1.0) if (ref and ref > 0) else None
    sev.append(dict(img=r, ref=ref, meth=meth, exc=exc))
    r["_exc"], r["_refm"] = exc, meth
estim = [s for s in sev if s["exc"] is not None]
print(f"  over total={len(supers)}  with reference={len(estim)}  without={len(supers)-len(estim)}")
by_meth = Counter(s["meth"] for s in sev)
print("  reference used:", dict(by_meth))


def dist(excs, rot):
    excs = sorted(excs)
    leve = sum(1 for e in excs if 0 <= e < 0.20)
    mod = sum(1 for e in excs if 0.20 <= e <= 0.50)
    sev_n = sum(1 for e in excs if e > 0.50)
    neg = sum(1 for e in excs if e < 0)
    md = excs[len(excs) // 2]
    print(f"  [{rot}] n={len(excs)}  min={excs[0]:+.0%} median={md:+.0%} max={excs[-1]:+.0%}  "
          f"| leve<20%={leve} mod20-50%={mod} severo>50%={sev_n} negativos={neg}")


if estim:
    dist([s["exc"] for s in estim], "todas c/ ref (inclui refs de outro poco = ruidoso)")
    clean = [s for s in estim if s["meth"] == "irmao_campo"]
    if clean:
        dist([s["exc"] for s in clean], "SUBCONJUNTO LIMPO: campo-irmao OK (mesmo poco)")
    print("  8 most severe:")
    for s in sorted(estim, key=lambda s: -s["exc"])[:8]:
        i = s["img"]
        print(f"    {i['area']:>7.2f}% vs ref {s['ref']:>6.2f}%  exc={s['exc']:+.0%} ({s['meth']})  "
              f"{i['au'][:34]} tp{i['tp']} c{i['campo']}")
    print("  NOTE: the cond_ok/lote_ok references confound excess with well-to-well")
    print("        biological variation; only 'irmao_campo' isolates over-segmentation.")

# ---- T3: the correction list ----
# Regra (estrita ao argumento do cancelamento):
#   USAVEL sem correcao  = a consistent pattern (over/OK/under) AND a plausible closure.
#                          (viés multiplicativo cancela -> a razao esta certa,
#                           mesmo com inflacao absoluta severa)
#   SEM_BASELINE         = no t0 or <2 timepoints -> outside the paired analysis for
#                          LACK OF A BASELINE, not because of segmentation. It does not
#                          enter the manual correction list (re-segmenting does not
#                          create the t0).
#   CORRIGIR             = MIXED (the bias does not cancel) OR consistent with a closure
#                          implausivel (k varia com o tempo). Baseline problematico
#                          entra sempre (t0 errado invalida a serie inteira).
print("\n=== TASK 3: the correction list (what actually matters) ===")
correcao = set()      # correcao manual firme (re-segmentar no Fiji)
revisar = set()       # decisao humana (frame invalido na serie)
serie_decisao = {}
# priority: (1) contains an invalid frame -> REVIEW ; (2) no baseline -> outside the
# pareada ; (3) consistente + closure plausivel -> USAVEL ; (4) resto -> CORRIGIR.
for sk, s in series.items():
    rs = by_sk[sk]
    indeterminado = (not s["has_base"]) or s["motivo"] in ("sem_baseline", "serie_1_ponto", "baseline_zero")
    consistente = s["pat"].startswith(("CONSISTENTE_SUPER", "CONSISTENTE_OK", "CONSISTENTE_SUB"))
    # invalid frames NEVER enter the correction (there is no wound to re-segment);
    # they are recorded in the review list for traceability only. The decision about
    # them was already taken by REGRA_INVALIDOS (R1/R2/R3), not case by case.
    for r in rs:
        if r["hc"] == "IMG_INVALIDA":
            # "noisy but segmentable" frames go to the CORRECTION (the decision is
            # ato: contorno tracado = valido; 'pulada' = invalido de fato)
            (correcao if r["k"] in RESOLVER else revisar).add(r["k"])
    if s["pat"] == "TODA_INVALIDA":
        serie_decisao[sk] = "TODA_INVALIDA_fora"
        continue
    if indeterminado:                       # includes R3 (invalid t0 with no sibling)
        serie_decisao[sk] = "SEM_BASELINE_fora_pareada"
        continue
    if consistente and s["plaus"]:
        serie_decisao[sk] = "USAVEL_sem_correcao"
        continue
    # -> needs manual correction (only bad segmentation is correctable)
    serie_decisao[sk] = "CORRIGIR_misto" if "MISTO" in s["pat"] else "CORRIGIR_consistente_implausivel"
    for r in rs:
        if r["hc"] in ("SEG_super", "SEG_sub"):
            correcao.add(r["k"])

dcount = Counter(serie_decisao.values())
print("  decision per series:")
for d, c in sorted(dcount.items(), key=lambda x: -x[1]):
    print(f"    {d:<40} {c:>3}")

n_usavel = dcount.get("USAVEL_sem_correcao", 0)
n_sembase = dcount.get("SEM_BASELINE_fora_pareada", 0)
n_toda = dcount.get("TODA_INVALIDA_fora", 0)
n_corr = len(series) - n_usavel - n_sembase - n_toda
n_emp = sum(1 for s in series.values() if s["emprestado"])
print(f"\n  >>> IMAGES IN THE MANUAL CORRECTION LIST: {len(correcao)}")
print(f"      invalid frames (recorded, outside the correction) : {len(revisar)}")
print(f"      series usable without correction (bias cancels)   : {n_usavel}")
print(f"      series out of the pairing for lack of a baseline  : {n_sembase}  (includes R3)")
print(f"      series 100% invalid                               : {n_toda}")
print(f"      series needing manual correction                  : {n_corr}")
print(f"      series with a baseline BORROWED from a sibling (R2): {n_emp}")
for sk, s in series.items():
    if s["emprestado"]:
        print(f"        {sk[:52]:<54} baseline {s['emprestado']}")

# ---- classe_validade + enriquecimento do data/inspecao_visual.csv ----
CLASSE = {"OK": "valida_ok", "SEG_super": "falha_metodo_WHST", "SEG_sub": "falha_metodo_WHST",
          "IMG_INVALIDA": "invalidez_imagem", "AMBIGUO": "a_adjudicar",
          # baselines recovered afterwards: outside the triage statistics until the
          # manual correction establishes their validity (they become OK or IMG_INVALIDA).
          "NAO_TRIADA": "pendente_baseline_recuperado"}
rows_out = list(csv.DictReader(open(HUM, encoding="utf-8-sig")))
exc_by_k = {r["k"]: r.get("_exc") for r in IM}
refm_by_k = {r["k"]: r.get("_refm") for r in IM}
for r in rows_out:
    k = r["whst_input_file"]; im = next(x for x in IM if x["k"] == k)
    hc = hcat(r)
    r["classe_validade"] = CLASSE.get(hc, "desconhecida")
    r["serie_padrao"] = series[im["sk"]]["pat"]
    r["serie_closure_plausivel"] = "sim" if series[im["sk"]]["plaus"] else "nao"
    r["serie_decisao"] = serie_decisao.get(im["sk"], "USAVEL_sem_correcao")
    e = exc_by_k.get(k)
    r["excesso_rel"] = f"{e:.3f}" if e is not None else ""
    r["excesso_ref"] = refm_by_k.get(k) or ""
    r["na_lista_correcao"] = "sim" if k in correcao else "nao"
    r["na_lista_revisao"] = "sim" if k in revisar else "nao"
fields = list(rows_out[0].keys())
with open(HUM, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows_out)
print(f"\n  rewrote {HUM} (+ classe_validade and analysis columns)")

# ---- tabela por serie ----
with open("data/whst_series_analysis.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series_key", "cell_line", "lote", "n_img", "padrao", "tem_baseline",
                "closure_plausivel", "motivo", "closure_seq", "decisao",
                "n_frames_invalidos", "baseline_emprestado"])
    for sk, s in sorted(series.items()):
        cv = ";".join(f"{tp}h:{c:.3f}" for tp, c in s["seq"])
        w.writerow([sk, s["cl"], s["lote"], s["n"], s["pat"], s["has_base"],
                    s["plaus"], s["motivo"], cv, serie_decisao.get(sk, "USAVEL_sem_correcao"),
                    s["n_inval"], s["emprestado"] or "nao"])
print("  saved data/whst_series_analysis.csv")

# ---- classe_validade resumo ----
cc = Counter(CLASSE[hcat(r)] for r in rows_out)
print("\n=== classe_validade (exclusion rate vs method failure) ===")
for c, n in sorted(cc.items(), key=lambda x: -x[1]):
    print(f"  {c:<20} {n:>3}  ({n/len(rows_out):.1%})")
