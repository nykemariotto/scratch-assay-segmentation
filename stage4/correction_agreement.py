# -*- coding: utf-8 -*-
"""
stage4/correction_agreement.py — validation metrics for the manual correction.

(A) MAGNITUDE OF THE MANUAL INTERVENTION  (pass 1 vs automatic)
    IoU(auto_mask, corrected_mask) per image.
    -> quantifies how much the manual correction changed the WHST result.
    -> Methods: "manual correction altered the automated segmentation with a
       median IoU of X (IQR ...)".

(B) INTRA-OBSERVER REPRODUCIBILITY  (pass 1 vs pass 2, blinded)
    IoU(pass1_mask, pass2_mask) + Lin's CCC between the areas.
    -> Methods: "manual correction showed intra-observer reproducibility of
       IoU X and CCC Y (n=15, blinded re-correction after an interval)".

Lin's CCC:  2*rho*sx*sy / (sx^2 + sy^2 + (mx-my)^2)
IoU:        |A n B| / |A u B|  over binary masks.

Self-test:  python stage4/correction_agreement.py --selftest
"""
import csv, os, sys
import numpy as np
from PIL import Image

AUTO_MASKS = "whst_output/masks"
P1_MASKS = "whst_output/rois_corrected/masks"
P2_MASKS = "whst_output/rois_blind_repeat/masks"
HIDDEN = "stage4/.recorrecao_oculta.csv"
OUT = "data/correction_agreement.csv"


# ------------------------- metrics -------------------------
def iou(a, b):
    """IoU between two binary masks (bool arrays of the same shape).

    CONVENTION for an empty union: when both masks are empty, the two readings
    agree that the wound closed (area = 0). Mathematically 0/0 is undefined, but
    discarding those cases as NaN would remove precisely the
    concordancias perfeitas e SUBESTIMARIA a reprodutibilidade. Retorna 1.0.
    If only one is empty, the intersection is 0 and the IoU is 0 (total disagreement).
    """
    if a.shape != b.shape:
        raise ValueError(f"shapes diferentes: {a.shape} vs {b.shape}")
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    if union == 0:
        return 1.0                      # ambas vazias = concordam (fechada)
    return float(inter / union)


def ccc(x, y):
    """Lin's concordance correlation coefficient (population, ddof=0)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=0), y.var(ddof=0)
    cov = ((x - mx) * (y - my)).mean()
    den = vx + vy + (mx - my) ** 2
    return float(2 * cov / den) if den > 0 else float("nan")


def load_mask(path):
    a = np.asarray(Image.open(path).convert("L"))
    return a > 127


def base_of(whst_input_file):
    b = whst_input_file
    for ext in (".tiff", ".tif"):
        if b.lower().endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


def describe(v, rot, pct=True):
    v = np.asarray([x for x in v if not np.isnan(x)], dtype=float)
    if not len(v):
        print(f"  {rot}: no data"); return
    f = (lambda z: f"{z:.4f}")
    print(f"  {rot}: n={len(v)}  median={f(np.median(v))}  "
          f"IQR=[{f(np.percentile(v,25))}, {f(np.percentile(v,75))}]  "
          f"min={f(v.min())}  max={f(v.max())}")


# ------------------------- self-test -------------------------
def selftest():
    print("=== AUTOTESTE das metricas ===")
    a = np.zeros((100, 100), bool); a[10:60, 10:60] = True      # 2500 px
    b = np.zeros((100, 100), bool); b[10:60, 10:60] = True
    assert abs(iou(a, b) - 1.0) < 1e-12, "IoU identico deve ser 1"
    c = np.zeros((100, 100), bool); c[60:90, 60:90] = True      # disjunto
    assert iou(a, c) == 0.0, "IoU disjunto deve ser 0"
    d = np.zeros((100, 100), bool); d[10:60, 35:85] = True      # metade sobrepoe
    # inter = 50x25=1250 ; union = 2500+2500-1250 = 3750 -> 1/3
    assert abs(iou(a, d) - 1 / 3) < 1e-12, f"IoU esperado 1/3, veio {iou(a,d)}"
    print("  IoU: identical=1, disjoint=0, half-overlap=1/3  OK")

    x = [1, 2, 3, 4, 5]
    assert abs(ccc(x, x) - 1.0) < 1e-12, "CCC identico deve ser 1"
    y = [v + 10 for v in x]                                     # viés constante
    assert ccc(x, y) < 0.2, f"CCC com viés grande deveria cair, veio {ccc(x,y)}"
    z = [2 * v for v in x]                                      # viés multiplicativo
    assert ccc(x, z) < 1.0
    # known value: perfect correlation, same mean/variance -> 1
    print(f"  CCC: identical=1  shift(+10)={ccc(x,y):.4f}  scale(x2)={ccc(x,z):.4f}  OK")
    print("self-test: EVERY ASSERTION PASSED")


# ------------------------- execution -------------------------
def main():
    if not os.path.isdir(P1_MASKS):
        sys.exit(f"{P1_MASKS}/ does not exist yet — run pass 1 in Fiji "
                 f"(stage4/whst_manual_correction.ijm) antes desta analise.")

    p1 = {f[:-9]: os.path.join(P1_MASKS, f) for f in os.listdir(P1_MASKS)
          if f.endswith("_mask.png")}
    auto = {f[:-9]: os.path.join(AUTO_MASKS, f) for f in os.listdir(AUTO_MASKS)
            if f.endswith("_mask.png")}
    p2 = {}
    if os.path.isdir(P2_MASKS):
        p2 = {f[:-9]: os.path.join(P2_MASKS, f) for f in os.listdir(P2_MASKS)
              if f.endswith("_mask.png")}

    print(f"masks: auto={len(auto)}  pass1={len(p1)}  pass2={len(p2)}")

    rows = []
    # ---- (A) auto vs corrected ----
    ious_ac, dpct = [], []
    for b, path in sorted(p1.items()):
        if b not in auto:
            print(f"  [warning] no automatic mask for {b}"); continue
        ma, mc = load_mask(auto[b]), load_mask(path)
        v = iou(ma, mc)
        aa, ac = ma.sum(dtype=np.int64), mc.sum(dtype=np.int64)
        tot = ma.size
        d = 100.0 * (ac - aa) / tot
        ious_ac.append(v); dpct.append(d)
        rows.append({"base": b, "iou_auto_vs_pass1": round(v, 5),
                     "area_pct_auto": round(100.0 * aa / tot, 3),
                     "area_pct_pass1": round(100.0 * ac / tot, 3),
                     "delta_pp": round(d, 3), "iou_pass1_vs_pass2": "",
                     "area_pct_pass2": ""})
    print("\n=== (A) MAGNITUDE OF THE MANUAL INTERVENTION (auto vs corrected) ===")
    describe(ious_ac, "IoU(auto, corrigido)")
    if dpct:
        dd = np.array(dpct)
        print(f"  area delta (pp): median={np.median(dd):+.2f}  "
              f"min={dd.min():+.2f}  max={dd.max():+.2f}")
        print(f"  corrections that REDUCED the area: {(dd<0).sum()}/{len(dd)} "
              f"({100*(dd<0).mean():.0f}%)  [esperado: maioria, pois o WHST super-segmenta]")

    # ---- (B) intra-observer reproducibility ----
    idx = {r["base"]: r for r in rows}
    if p2:
        ious_12, a1, a2 = [], [], []
        for b, path in sorted(p2.items()):
            if b not in p1:
                print(f"  [warning] pass 2 without pass 1: {b}"); continue
            m1, m2 = load_mask(p1[b]), load_mask(path)
            v = iou(m1, m2)
            ious_12.append(v)
            tot = m1.size
            x, y = 100.0 * m1.sum(dtype=np.int64) / tot, 100.0 * m2.sum(dtype=np.int64) / tot
            a1.append(x); a2.append(y)
            if b in idx:
                idx[b]["iou_pass1_vs_pass2"] = round(v, 5)
                idx[b]["area_pct_pass2"] = round(y, 3)
        print("\n=== (B) REPRODUTIBILIDADE INTRA-OBSERVADOR (passada1 vs passada2, cega) ===")
        describe(ious_12, "IoU(pass1, pass2)")
        if len(a1) >= 2:
            c = ccc(a1, a2)
            r = float(np.corrcoef(a1, a2)[0, 1])
            print(f"  CCC de Lin (areas %): {c:.4f}   (Pearson r={r:.4f}, n={len(a1)})")
            print(f"  viés medio pass2-pass1: {np.mean(np.array(a2)-np.array(a1)):+.3f} pp")
            med = float(np.nanmedian(ious_12))
            print("\n  >>> SENTENCE FOR THE METHODS:")
            print(f'      "Manual correction showed intra-observer reproducibility')
            print(f'       de IoU {med:.3f} (mediana) e CCC de Lin {c:.3f}')
            print(f'       (n={len(a1)} images re-corrected blind)."')
        # check it matched the draw (images without a mask = 'invalida'/'pulada')
        if os.path.isfile(HIDDEN):
            want = {base_of(r["whst_input_file"]) for r in
                    csv.DictReader(open(HIDDEN, encoding="utf-8-sig"))}
            got = set(p2)
            falta = want - got
            if falta:
                st = {}
                if os.path.isfile("stage4/manual_correction_pass2.csv"):
                    st = {base_of(r["whst_input_file"]): r["status"] for r in
                          csv.DictReader(open("stage4/manual_correction_pass2.csv", encoding="utf-8-sig"))}
                for b in sorted(falta):
                    s = st.get(b, "not processed")
                    print(f"\n  [note] no mask on pass 2: status='{s}'")
                    print(f"         {b[:66]}")
                    if s in ("invalida", "pulada"):
                        print(f"         (expected: '{s}' produces no mask)")
            if got - want:
                print(f"  [warning] {len(got-want)} re-corrections OUTSIDE the draw")

        # ---- STATUS agreement (IoU does not capture this) ----
        f1, f2 = "stage4/manual_correction_pass1.csv", "stage4/manual_correction_pass2.csv"
        if os.path.isfile(f1) and os.path.isfile(f2):
            s1 = {r["whst_input_file"]: r["status"] for r in
                  csv.DictReader(open(f1, encoding="utf-8-sig"))}
            s2 = {r["whst_input_file"]: r["status"] for r in
                  csv.DictReader(open(f2, encoding="utf-8-sig"))}
            comuns = [k for k in s2 if k in s1]
            iguais = [k for k in comuns if s1[k] == s2[k]]
            print(f"\n=== (C) OUTCOME AGREEMENT (ok/fechada/invalida) ===")
            print(f"  images on both passes: {len(comuns)}")
            print(f"  same outcome: {len(iguais)}/{len(comuns)} ({len(iguais)/max(len(comuns),1):.0%})")
            for k in comuns:
                if s1[k] != s2[k]:
                    print(f"  [disagrees] pass1='{s1[k]}' pass2='{s2[k]}'  {k[:58]}")
    else:
        print("\n=== (B) pass 2 has not been run yet ===")
        print("  Run the macro again choosing '2 - RE-correcao cega'.")

    if rows:
        with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved: {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
