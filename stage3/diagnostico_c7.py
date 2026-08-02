# -*- coding: utf-8 -*-
"""
stage3/diagnostico_c7.py — systematic failure × noise in the discordant observations.

WHAT SURVIVES RETRAINING IS NOT THE NUMBERS, IT IS THE TEST.

The manuscript used to describe every negative closure fraction as measurement
noise, when at least one series shows the model negative at all three timepoints
while the manual measurement is positive and increasing. The criterion that
separates the two cases:

  NOISE      -> the discrepancy between the methods varies in size and in sign
                along the series; independent per-image errors.
  SYSTEMATIC -> the discrepancy is an approximately CONSTANT offset over a
                trajectory of the right shape. That is what a wrong baseline (t=0)
                produces, because it enters every closure fraction through the
                denominator of (area₀ − areaₜ)/area₀ — a single error contaminates
                the whole series.

The discriminant is the range (max − min) of the difference within the series,
compared with the range across the other discordant observations.

Uso:
    python stage3/diagnostico_c7.py [caminho.csv]

Expects columns identifying the series and the timepoint, plus the two measurements. It accepts both
o formato antigo (annotator/replicate/timepoint_h/imagej/ai) quanto o novo
(analysis_unit/campo/timepoint_h/... ) — ver COLUNAS abaixo.
"""
import csv
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# (columns identifying the series, the timepoint column, reference, automatic)
COLUNAS = [
    (("annotator", "replicate"), "timepoint_h", "imagej", "ai"),
    (("analysis_unit",), "timepoint_h", "ref_closure", "ai_closure"),
    (("analysis_unit", "campo"), "timepoint_h", "imagej", "ai"),
]

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "paired_data", "paired_imagej_vs_ai_clean.csv")
if not os.path.isfile(CSV):
    sys.exit(f"could not find {CSV}")

linhas = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
if not linhas:
    sys.exit("empty csv")
cols = set(linhas[0])

esquema = next((e for e in COLUNAS
                if set(e[0]) <= cols and e[1] in cols and e[2] in cols and e[3] in cols), None)
if esquema is None:
    sys.exit(f"schema not recognised. Columns present: {sorted(cols)}")
CHAVE, TP, REF, AUTO = esquema
print(f"file    : {CSV}")
print(f"schema  : series={'+'.join(CHAVE)} · time={TP} · ref={REF} · auto={AUTO}")
print(f"rows    : {len(linhas)}\n")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


obs = []
for r in linhas:
    a, b = num(r[REF]), num(r[AUTO])
    if a is None or b is None:
        continue
    obs.append({"serie": tuple(r[k] for k in CHAVE), "tp": num(r[TP]),
                "ref": a, "auto": b, "dif": a - b})

# discordant = at least one of the two is negative
neg = [o for o in obs if o["ref"] < 0 or o["auto"] < 0]
conc = [o for o in neg if o["ref"] * o["auto"] > 0]
disc = [o for o in neg if o["ref"] * o["auto"] <= 0]
print(f"observations with negative closure : {len(neg)} of {len(obs)} "
      f"({100*len(neg)/len(obs):.1f}%)")
print(f"  agreeing in sign (noise)         : {len(conc)}")
print(f"  disagreeing in sign              : {len(disc)}\n")

if not disc:
    print("No discordant observation — the systematic-failure argument does not apply")
    print("to this pairing. The corresponding limitation has to be rewritten.")
    sys.exit(0)

# discordant series with >= 3 timepoints: the internal range can be measured
por_serie = {}
for o in disc:
    por_serie.setdefault(o["serie"], []).append(o)

print("=== discordant series with ≥3 timepoints ===")
sistematicas = []
for s, itens in sorted(por_serie.items()):
    if len(itens) < 3:
        continue
    itens.sort(key=lambda o: o["tp"])
    difs = [o["dif"] for o in itens]
    amp = max(difs) - min(difs)
    media = sum(difs) / len(difs)
    fora = [o["dif"] for o in disc if o["serie"] != s]
    amp_fora = (max(fora) - min(fora)) if len(fora) > 1 else float("nan")
    mono_ref = all(itens[i]["ref"] < itens[i+1]["ref"] for i in range(len(itens)-1))
    mono_auto = all(itens[i]["auto"] < itens[i+1]["auto"] for i in range(len(itens)-1))
    print(f"\n  series {' / '.join(s)}")
    print(f"    {'tp':>6} {'ref':>10} {'auto':>10} {'ref-auto':>10}")
    for o in itens:
        print(f"    {o['tp']:>6.0f} {o['ref']:>+10.4f} {o['auto']:>+10.4f} {o['dif']:>+10.4f}")
    print(f"    mean offset ......... {media:+.3f}")
    print(f"    internal range ...... {amp:.4f}")
    print(f"    range in the others   {amp_fora:.4f}"
          + (f"  ({amp_fora/amp:.0f}× maior)" if amp > 1e-9 and amp_fora == amp_fora else ""))
    print(f"    monotônicas (ref/auto): {mono_ref} / {mono_auto}")
    if amp > 1e-9 and amp_fora == amp_fora and amp_fora / amp >= 5:
        sistematicas.append((s, media, amp, amp_fora))
        print("    -> SISTEMÁTICA: deslocamento quase constante sobre trajetória de forma correta")
    else:
        print("    -> not separable from noise by this criterion")

print("\n" + "=" * 66)
if sistematicas:
    print(f"{len(sistematicas)} systematic series. Sentence for the limitations:\n")
    for s, media, amp, amp_fora in sistematicas:
        print(f'  "… one series ({" / ".join(s)}) … the difference is nearly constant '
              f'(mean {media:+.3f}, spread {amp:.3f}) against a spread of {amp_fora:.3f} '
              f'across the other discordant observations."')
else:
    print("No systematic series by this criterion.")
    print("If the new pairing lands here, the corresponding limitation has to drop the")
    print("replicate-9 case and say instead that, under the leakage-free partition, the")
    print("remaining discordances are compatible with noise.")
