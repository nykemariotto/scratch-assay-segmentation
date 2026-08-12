# -*- coding: utf-8 -*-
"""
stage3/aggregate.py — PHASE 2 (CPU): from the records to Table 2, the CIs and the
tests.

Consumes `stage3/records/*.json`, produced by stage3/eval_test.py. Needs no GPU
and can be re-run freely.

WHAT IT PRODUCES
  1. Table 2 — one row per CONFIGURATION (not per run): mean ± SD over the
     5 seeds. What varies between seeds is training noise, and reporting a single
     run as though it were "the performance of the model" is exactly what a
     single-run design gets criticised for.
  2. Bootstrap CI GROUPED by acquisition field, alongside the naive CI, so the
     difference between the two is visible.
  3. Padding ablation — comparison paired by seed between yolo11m black and
     white. Paired because the same seed shares every source of stochasticity
     except the padding colour; comparing unpaired means would throw that
     information away.
  4. Distinguishability — 95% CI of the DIFFERENCE between each pair, by PAIRED
     grouped bootstrap. Not CI overlap: overlap is far too conservative a
     criterion and calls "indistinguishable" what is distinguishable.

SECTIONS 3 AND 4 ANSWER DIFFERENT QUESTIONS, and the manuscript needs both:
  section 3 — is the difference between configurations larger than TRAINING NOISE?
              (paired by seed, same test set)
  section 4 — is the difference larger than the SAMPLING VARIATION OF THE TEST SET?
              (paired by acquisition group)
A difference is only solid if it survives both.

  python stage3/aggregate.py
  python stage3/aggregate.py --B 5000 --conf 0.8
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
os.chdir(os.path.dirname(AQUI))

import numpy as np

from ap_core import (IDX_50, IDX_75, average_precision, bootstrap_ingenuo_config,
                     cluster_bootstrap_config, cluster_bootstrap_pareado_config,
                     mapa_5095, prf)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REGS_PADRAO = os.path.join("stage3", "registros")

# run name -> (configuration, seed).  The configuration is what becomes a Table 2 row.
#
# ONE NAME FOR EACH CONFIGURATION, and it is the manuscript's. These labels used to be
# descriptive strings — "YOLO11m-seg · black · COCO" — which the manuscript then had to
# translate into S / M / X / M-white / M-scratch at writing time. That left the same
# five configurations carrying three different names: one in the paper, one in the run
# directories, one in this CSV, with the interface about to add a fourth. The paper's
# is the one that survives; the others were intermediates.
#
# The descriptor is not lost, it moves to its own column. A deposited CSV has to say
# what M is without the reader holding Table 1 open, and the run directory it came from
# is what makes the row traceable back to the weights.
ROTULOS = {
    "yolo11s-seg_black_coco": "S",
    "yolo11m-seg_black_coco": "M",
    "yolo11x-seg_black_coco": "X",
    "yolo11m-seg_white_coco": "M-white",
    "yolo11m-seg_black_scratch": "M-scratch",
    "unet_black": "U-Net",
}

DESCRICAO = {
    "yolo11s-seg_black_coco": "YOLO11s-seg · black padding · COCO init",
    "yolo11m-seg_black_coco": "YOLO11m-seg · black padding · COCO init",
    "yolo11x-seg_black_coco": "YOLO11x-seg · black padding · COCO init",
    "yolo11m-seg_white_coco": "YOLO11m-seg · white padding · COCO init",
    "yolo11m-seg_black_scratch": "YOLO11m-seg · black padding · from scratch",
    "unet_black": "canonical U-Net · black padding · from scratch (comparator)",
}


def parse_run(nome):
    if "_seed" not in nome:
        return nome, None
    base, s = nome.rsplit("_seed", 1)
    try:
        return base, int(s)
    except ValueError:
        return nome, None


def carrega(pasta, permitir_fixture=False):
    fs = sorted(glob.glob(os.path.join(pasta, "*.json")))
    if not fs:
        sys.exit(f"no records in {pasta}/ — run stage3/eval_test.py first")
    out, fx = {}, []
    for f in fs:
        d = json.load(open(f, encoding="utf-8"))
        # GUARD: the fixtures carry the SAME NAMES as the real runs. Without this
        # check, pointing --dir at the wrong folder would produce a Table 2 of
        # invented numbers that looks perfectly normal.
        if d.get("FIXTURE") and not permitir_fixture:
            fx.append(os.path.basename(f))
            continue
        out[d["run"]] = d
    if fx and not out:
        print(f"[fixtures] {len(fx)} synthetic file(s) in this folder.")
        print("To exercise the mechanics with them, pass --permitir-fixture.")
        sys.exit("ABORTED: there is only synthetic data here.")
    if fx:
        sys.exit(f"ABORTED: the folder mixes {len(fx)} fixture(s) with "
                 f"{len(out)} real run(s). Separate them before aggregating.")
    return out


def metricas_de(d, conf):
    regs = list(d["registros"].values())
    m = prf(regs, conf)
    m.update({"mAP50": average_precision(regs, IDX_50),
              "mAP75": average_precision(regs, IDX_75),
              "mAP50_95": mapa_5095(regs)})
    return m


def ms(v):
    """mean ± sample SD; SD undefined for n<2."""
    if len(v) < 2:
        return (v[0] if v else float("nan")), float("nan")
    return st.mean(v), st.stdev(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.8,
                    help="threshold for precision/recall/F1 (does not affect mAP)")
    # 5000, nao 2000. Os intervalos publicados sairam de 5000 reamostragens; com o
    # default antigo quem clonasse o repositorio e rodasse sem argumento obtinha IC
    # diferentes dos do artigo, sem nenhum aviso. As estimativas pontuais nao mudam
    # com B — so os limites de percentil —, o que torna a divergencia discreta o
    # bastante para passar por ruido de arredondamento. O valor usado fica gravado em
    # table2_detalhe.json, mas isso audita depois do fato; o default e o que protege
    # antes.
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dir", default=REGS_PADRAO,
                    help="records folder (use stage3/records_fixture for the dry run)")
    ap.add_argument("--out", default=os.path.join("stage3", "table2.csv"))
    ap.add_argument("--permitir-fixture", action="store_true",
                    help="accept synthetic records (only to validate the mechanics)")
    args = ap.parse_args()

    dados = carrega(args.dir, args.permitir_fixture)
    porcfg = {}
    for nome, d in dados.items():
        cfg, seed = parse_run(nome)
        porcfg.setdefault(cfg, []).append((seed, nome, d))
    for v in porcfg.values():
        v.sort()

    print("=" * 96)
    print(f"STAGE 3 · {len(dados)} runs in {len(porcfg)} configurations · "
          f"confidence threshold {args.conf} for P/R/F1")
    print("=" * 96)

    # ---------------------------------------------------------- 1. Table 2
    tabela = []
    for cfg, runs in porcfg.items():
        met = {k: [] for k in ("mAP50", "mAP75", "mAP50_95", "precision", "recall", "f1")}
        for _, _, d in runs:
            m = metricas_de(d, args.conf)
            for k in met:
                met[k].append(m[k])
        linha = {"config": ROTULOS.get(cfg, cfg), "cfg": cfg,
                 "description": DESCRICAO.get(cfg, ""), "run_base": cfg,
                 "n_seeds": len(runs)}
        for k, v in met.items():
            mu, sd = ms(v)
            linha[k] = mu
            linha[k + "_sd"] = sd
            linha[k + "_vals"] = v
        tabela.append(linha)
    tabela.sort(key=lambda r: -r["mAP50"])

    print("\n1. TABLE 2 — mean ± SD over the seeds  (values in %)\n")
    cab = f"{'Configuration':<38} {'n':>2}  {'mAP@50':>13} {'mAP@75':>13} {'mAP@50-95':>13} {'F1':>13}"
    print(cab)
    print("-" * len(cab))
    for r in tabela:
        def c(k):
            sd = r[k + "_sd"]
            return f"{100*r[k]:5.1f}±{100*sd:4.1f}" if sd == sd else f"{100*r[k]:9.1f}  "
        print(f"{r['config']:<38} {r['n_seeds']:>2}  {c('mAP50'):>13} {c('mAP75'):>13} "
              f"{c('mAP50_95'):>13} {c('f1'):>13}")

    # ------------------------------------------- 2. IC agrupado x ingenuo
    print(f"\n2. 95% BOOTSTRAP CI · resampling acquisition GROUPS  (B={args.B})")
    print("   CI of the configuration MEAN: the groups are resampled once per")
    print("   iteration and applied to all 5 seeds; the statistic is the mean across")
    print("   them, so the centre of the CI is the same number as in Table 2.\n")
    print(f"{'Configuration':<38} {'mAP@50':>7}  {'grouped CI':>18}  {'naive CI':>18}  {'ratio':>6}")
    print("-" * 96)
    ics = {}
    # Accumulator for what used to exist only on the console. A number with no
    # file that generates it is not merely irreproducible: it is UNAUDITABLE.
    PERSIST = {"conf": args.conf, "B": args.B, "seed": args.seed,
               "ic_por_configuracao": [], "padding_ablation": {},
               "pares": [], "indistinguiveis": []}
    fn = lambda rs: average_precision(rs, IDX_50)
    for r in tabela:
        runs = porcfg[r["cfg"]]
        por_seed = [d["registros"] for _, _, d in runs]
        grupo_de = {k: v["grupo"] for k, v in runs[0][2]["registros"].items()}
        cb = cluster_bootstrap_config(por_seed, grupo_de, fn, B=args.B, seed=args.seed)
        # the naive interval is ALSO over the configuration mean — it used to be
        # over a single run, and the width ratio then compared different things
        ib = bootstrap_ingenuo_config(por_seed, fn, B=args.B, seed=args.seed)
        lc, li = cb["hi"] - cb["lo"], ib["hi"] - ib["lo"]
        ics[r["cfg"]] = cb
        PERSIST["ic_por_configuracao"].append({
            "config": r["config"], "cfg": r["cfg"], "mAP50": cb["obs"],
            "ic_agrupado": [cb["lo"], cb["hi"]], "ic_ingenuo": [ib["lo"], ib["hi"]],
            "razao_largura": (lc / li) if li else None})
        print(f"{r['config']:<38} {100*cb['obs']:7.1f}  "
              f"[{100*cb['lo']:6.1f},{100*cb['hi']:6.1f}]  "
              f"[{100*ib['lo']:6.1f},{100*ib['hi']:6.1f}]  "
              f"{(lc/li if li else float('nan')):6.2f}x")
    print("\n   The last column is the factor by which the naive CI understates uncertainty.")
    print("   Resampling images treats frames of the same field as independent — they are not.")
    print("   (both over the configuration mean, so that they are comparable)")

    # consistency: the centre of the CI MUST match the mean in Table 2
    for r in tabela:
        if abs(ics[r["cfg"]]["obs"] - r["mAP50"]) > 1e-9:
            print(f"   [WARNING] {r['config']}: centre of the CI "
                  f"{100*ics[r['cfg']]['obs']:.2f} ≠ Table 2 mean {100*r['mAP50']:.2f}")

    # --------------------------------------------- 3. ablation de padding
    print("\n3. PADDING ABLATION — paired by seed")
    CFG_B, CFG_W = "yolo11m-seg_black_coco", "yolo11m-seg_white_coco"
    if CFG_B not in porcfg or CFG_W not in porcfg:
        # fail loudly: if the config names change, the ablation would vanish
        # silently, reporting "seeds in common: 0" as though it were missing data
        print(f"   [WARNING] ablation configuration missing — present: {sorted(porcfg)}")
    pb = {s: d for s, n, d in porcfg.get(CFG_B, [])}
    pw = {s: d for s, n, d in porcfg.get(CFG_W, [])}
    comuns = sorted(set(pb) & set(pw))
    if len(comuns) < 2:
        print(f"   seeds in common: {len(comuns)} — not enough")
    else:
        difs = []
        print(f"\n   {'seed':>5} {'black':>8} {'white':>8} {'Δ (b−w)':>10}")
        for s in comuns:
            b = metricas_de(pb[s], args.conf)["mAP50"]
            w = metricas_de(pw[s], args.conf)["mAP50"]
            difs.append(b - w)
            print(f"   {s:>5} {100*b:8.1f} {100*w:8.1f} {100*(b-w):+10.2f}")
        mu, sd = ms(difs)
        print(f"\n   mean Δ = {100*mu:+.2f} pp · SD {100*sd:.2f} pp · n={len(difs)}")
        if sd > 0:
            t = mu / (sd / np.sqrt(len(difs)))
            print(f"   paired t = {t:+.2f} with {len(difs)-1} df")
        pos = sum(1 for d in difs if d > 0)
        print(f"   black wins in {pos}/{len(difs)} seeds")
        PERSIST["padding_ablation"] = {
            "config_black": CFG_B, "config_white": CFG_W, "seeds": comuns,
            "diferencas_pp": [100 * d for d in difs],
            "media_pp": 100 * mu, "dp_pp": 100 * sd, "n": len(difs),
            "t_pareado": (mu / (sd / np.sqrt(len(difs)))) if sd > 0 else None,
            "gl": len(difs) - 1, "black_vence": pos}
        print("\n   PAIRED deliberately: the same seed shares every source of")
        print("   stochasticity except the padding colour. Comparing unpaired means")
        print("   would throw away exactly what the single-variable design bought.")

    # ------------------------------------------- 4. distinguibilidade
    print("\n4. ARE THE CONFIGURATIONS DISTINGUISHABLE?  (95% CI of the DIFFERENCE, paired)")
    print("   CI overlap is NOT a test of difference — it is too conservative and")
    print("   would call 'indistinguishable' what is distinguishable. The models are")
    print("   evaluated on the SAME images: resampling the same groups for both and")
    print("   taking the difference inside each resample cancels the variance of the")
    print("   test set. A CI of the difference that excludes zero = a difference.\n")
    nomes = [r["cfg"] for r in tabela]
    fn = lambda rs: average_precision(rs, IDX_50)
    print(f"   {'A':<30} {'B':<30} {'Δ (pp)':>8}  {'IC 95%':>18}  ")
    print("   " + "-" * 92)
    indist = []
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            sa = [d2["registros"] for _, _, d2 in porcfg[nomes[i]]]
            sb = [d2["registros"] for _, _, d2 in porcfg[nomes[j]]]
            g = {k: v["grupo"] for k, v in sa[0].items()}
            d = cluster_bootstrap_pareado_config(sa, sb, g, fn,
                                                 B=args.B, seed=args.seed)
            marca = "*" if d["exclui_zero"] else " "
            if not d["exclui_zero"]:
                indist.append((nomes[i], nomes[j]))
            PERSIST["pares"].append({
                "a": ROTULOS.get(nomes[i], nomes[i]),
                "b": ROTULOS.get(nomes[j], nomes[j]),
                "cfg_a": nomes[i], "cfg_b": nomes[j], "delta_pp": 100 * d["obs"],
                "ic_lo_pp": 100 * d["lo"], "ic_hi_pp": 100 * d["hi"],
                "exclui_zero": bool(d["exclui_zero"])})
            print(f"   {ROTULOS.get(nomes[i],nomes[i])[:29]:<30} "
                  f"{ROTULOS.get(nomes[j],nomes[j])[:29]:<30} "
                  f"{100*d['obs']:+8.2f}  [{100*d['lo']:+7.2f},{100*d['hi']:+7.2f}] {marca}")
    print("\n   (*) the CI excludes zero — a difference supportable at the 5% level")
    print("   Difference between the MEANS: the same groups are resampled for both")
    print("   configurations and the statistic is the difference of the means over the")
    print("   5 seeds. That cancels the variance of the test set and of the seed at")
    print("   once. Section 3 tests the same comparison against training noise; a")
    print("   difference is only solid if it survives BOTH.")
    if indist:
        print(f"\n   {len(indist)} pair(s) with NO supportable difference. For these the")
        print("   manuscript cannot claim superiority — which is what multiple seeds are for.")
    else:
        print("\n   Every pair is distinguishable.")

    # ---------------------------------------------------------- CSV
    os.makedirs("stage3", exist_ok=True)
    dest = args.out
    with open(dest, "w", newline="", encoding="utf-8") as f:
        cols = ["config", "description", "run_base", "n_seeds"] + [f"{k}{sf}" for k in
                ("mAP50", "mAP75", "mAP50_95", "precision", "recall", "f1")
                for sf in ("", "_sd")] + ["ic_lo_mAP50", "ic_hi_mAP50"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in tabela:
            r = dict(r)
            r["ic_lo_mAP50"] = ics[r["cfg"]]["lo"]
            r["ic_hi_mAP50"] = ics[r["cfg"]]["hi"]
            w.writerow(r)
    print(f"\nwritten: {dest}")

    # ── PERSIST WHAT USED TO EXIST ONLY ON THE CONSOLE ──────────────────────
    # `stage3/table2.csv` held the mean, the SD and the grouped CI per
    # configuration. The naive CI, the ratio between the widths, the seed-paired
    # ablation and the 10-pair distinguishability matrix were printed and nothing
    # more. A number with no file that generates it is not merely irreproducible:
    # it is unauditable, because nobody re-checks what they cannot see the
    # provenance of. That is how an inflated intra-observer IoU reached eight
    # documents before anyone looked at the pairs one by one.
    PERSIST["indistinguiveis"] = [[ROTULOS.get(x, x) for x in par] for par in indist]
    PERSIST["n_pares"] = len(PERSIST["pares"])
    PERSIST["n_indistinguiveis"] = len(indist)
    dj = os.path.splitext(dest)[0] + "_detalhe.json"
    json.dump(PERSIST, open(dj, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"written: {dj}")


if __name__ == "__main__":
    main()
