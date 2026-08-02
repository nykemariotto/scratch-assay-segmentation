# -*- coding: utf-8 -*-
"""
QC (b) — razao area_anotacao / area_WHST por imagem do test. RODAR APOS a medicao
in Fiji. Lists the discrepant ones (ratio outside ~0.5-2.0).

INPUT (--whst): CSV of the WHST measurements of the 223 raw images in whst_input/. Columns
esperadas (nomes configuraveis abaixo):
    whst_input_file  -> nome do arquivo em whst_input/ (chave de juncao)
    whst_area_px     -> measured wound area (in PIXELS of the native raw image)
  (optional) whst_area_pct -> wound area as a % of the image
Se so tiver % , passe --usar-pct.

HANDLING THE 640x640 (center-crop, 16% smaller FOV): the annotation of those 118 lives in the
espaco 640 (FOV recortado) e o WHST mede a crua nativa (FOV completo) -> comparacao
de area ABSOLUTA invalida. Por padrao sao EXCLUIDAS da razao e listadas a parte.

Uso:
    python stage4/qc_ratio_whst.py --whst medicoes_whst.csv
    python stage4/qc_ratio_whst.py --selftest      # generates a synthetic CSV and validates the logic
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

# ---- nomes de coluna esperados no CSV do WHST (ajuste se necessario) ----
COL_FILE = "whst_input_file"
COL_AREA_PX = "whst_area_px"
COL_AREA_PCT = "whst_area_pct"

CORR = os.path.join("whst_input", "correspondencia.csv")
CD = "coco_partitions"
RATIO_LO, RATIO_HI = 0.5, 2.0


def shoelace(seg):
    x = np.array(seg[0::2]); y = np.array(seg[1::2])
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def load_annotation_areas():
    """test_image -> (area_poligono_px_no_espaco_da_imagem, W, H, n_ann)."""
    out = {}
    d = json.load(open(os.path.join(CD, "instances_test.json"), encoding="utf-8"))
    ann = defaultdict(list)
    for a in d["annotations"]:
        ann[a["image_id"]].append(a)
    for im in d["images"]:
        aa = ann.get(im["id"], [])
        area = sum(shoelace(s) for a in aa for s in a.get("segmentation", []) if len(s) >= 6)
        out[im["file_name"]] = (area, im["width"], im["height"], len(aa))
    return out


def detect_cols(header):
    """aceita tanto whst_input_file/whst_area_* quanto filename/area_* (Fiji)."""
    f = "whst_input_file" if "whst_input_file" in header else ("filename" if "filename" in header else COL_FILE)
    px = "whst_area_px" if "whst_area_px" in header else ("area_px" if "area_px" in header else COL_AREA_PX)
    pct = "whst_area_pct" if "whst_area_pct" in header else ("area_pct" if "area_pct" in header else COL_AREA_PCT)
    return f, px, pct


def run(whst_csv, usar_pct):
    corr = list(csv.DictReader(open(CORR, encoding="utf-8-sig")))
    # whst_input_file -> correspondence row (test only, not baseline)
    file_to_test = {r["whst_input_file"]: r for r in corr
                    if r.get("is_baseline", "no") == "no" and r["test_image"] not in
                    ("(baseline_recuperado)", "(baseline_alternativo)")}
    ann = load_annotation_areas()

    reader = csv.DictReader(open(whst_csv, encoding="utf-8-sig"))
    cfile, cpx, cpct = detect_cols(reader.fieldnames)
    print(f"colunas detectadas: file={cfile} area_px={cpx} area_pct={cpct}")
    whst = {}
    for r in reader:
        f = r[cfile].strip()
        if usar_pct:
            whst[f] = ("pct", float(r[cpct]))
        else:
            whst[f] = ("px", float(r[cpx]))

    rows, excl_640, sem_ann, sem_medida = [], [], [], []
    for wf, cr in file_to_test.items():
        tb = cr["test_image"]
        a = ann.get(tb)
        if a is None:
            continue
        area_px, W, H, n = a
        is640 = (W, H) == (640, 640)
        if wf not in whst:
            sem_medida.append(tb); continue
        kind, val = whst[wf]
        if n == 0:
            sem_ann.append((tb, val)); continue
        if is640:
            excl_640.append(tb); continue
        # native: the annotation is already in raw pixels (2452x2056)
        if kind == "px":
            ratio = area_px / val if val > 0 else float("inf")
        else:  # pct: convert the annotation to % of the image
            ann_pct = 100.0 * area_px / (W * H)
            ratio = ann_pct / val if val > 0 else float("inf")
        rows.append({"test_image": tb, "group_key": cr["group_key"], "cell_line": cr["cell_line"],
                     "timepoint_h": cr["timepoint_h"], "whst_input_file": wf,
                     "ann_area_px": round(area_px, 1), "ann_pct": round(100.0*area_px/(W*H), 3),
                     "whst": val, "ratio": round(ratio, 3),
                     "flag": "OK" if RATIO_LO <= ratio <= RATIO_HI else "DISCREPANTE"})

    rows.sort(key=lambda r: r["ratio"])
    with open("stage4/qc_ratio_whst.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test_image", "group_key", "cell_line", "timepoint_h",
                                          "whst_input_file", "ann_area_px", "ann_pct", "whst", "ratio", "flag"])
        w.writeheader(); w.writerows(rows)

    disc = [r for r in rows if r["flag"] == "DISCREPANTE"]
    print(f"=== QC(b) annotation/WHST ratio ===")
    print(f"  native test images compared: {len(rows)}")
    print(f"  DISCREPANTES (razao <{RATIO_LO} ou >{RATIO_HI}): {len(disc)}")
    for r in disc[:20]:
        print(f"    ratio={r['ratio']:<6} {r['cell_line']} tp{r['timepoint_h']} "
              f"ann={r['ann_pct']}% whst={r['whst']} | {r['test_image'][:34]}")
    print(f"\n  EXCLUIDAS (640x640, FOV recortado — declarar): {len(excl_640)}")
    print(f"  without annotation (negatives; undefined ratio): {len(sem_ann)}")
    print(f"  without a WHST measurement in the CSV: {len(sem_medida)}")
    print(f"\nSalvo: stage4/qc_ratio_whst.csv")


def selftest():
    # generates a synthetic CSV consistent with the correspondence and runs
    corr = list(csv.DictReader(open(CORR, encoding="utf-8-sig")))
    ann = load_annotation_areas()
    fn = "_whst_selftest.csv"
    with open(fn, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow([COL_FILE, COL_AREA_PX])
        for r in corr:
            if r.get("is_baseline", "no") != "no":
                continue
            a = ann.get(r["test_image"])
            if not a:
                continue
            area_px, W, H, n = a
            # simulates WHST ~= annotation (ratio ~1) apart from noise; native only
            whst_px = max(1.0, area_px * (2056/640)**2) if (W, H) == (640, 640) else max(1.0, area_px * 1.05)
            w.writerow([r["whst_input_file"], round(whst_px, 1)])
    print("self-test: synthetic CSV generated, running QC(b)...\n")
    run(fn, usar_pct=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--whst", help="CSV of the WHST measurements")
    ap.add_argument("--usar-pct", action="store_true", help="usar coluna whst_area_pct em vez de px")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.whst:
        run(args.whst, args.usar_pct)
    else:
        sys.exit("informe --whst <csv> ou --selftest")
