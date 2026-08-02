# -*- coding: utf-8 -*-
"""
stage1/add_snap21_and_cleanup.py — Tarefa (2): adiciona Snap-21 (baseline ALTERNATIVO do
grupo P2|CARBO|F5, cujo primario Snap-20 tem confianca baixa) ao whst_input/,
marcado como baseline_alternativo. Tambem remove da correspondencia a imagem de
test image excluded from the dataset (B1_8hr 2, the duplicate removed in task 1).
It brings whst_input/ to 223 images.
"""
import csv, os, re, hashlib, shutil

P2_CARBO_0H = r"G:\.shortcut-targets-by-id\1CMwTWGfjZgxB1XuvetRHUONTxDknvhDU\WH_Carbo_25-10_27-10-22\0h\carbo\Snap-21.tiff"
OUT = "whst_input"
CORR = os.path.join(OUT, "correspondencia.csv")
REMOVED_TEST = "B1_8hr 2_png.rf.XKx5gS3ghJ7c4u8nIxAL.png"  # excluida do dataset na tarefa 1


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe(s):
    return re.sub(r"[^A-Za-z0-9+.-]+", "_", s).strip("_")


assert os.path.exists(P2_CARBO_0H), f"Snap-21 not found: {P2_CARBO_0H}"
h = md5(P2_CARBO_0H)
dest_name = f"{h[:10]}__BASELINE_ALT__SKOV-3__P2_CARBO_F5__tp0h__Snap-21.tiff"
dest = os.path.join(OUT, dest_name)
if not os.path.exists(dest):
    shutil.copy2(P2_CARBO_0H, dest)
print(f"copiado Snap-21 -> {dest_name}")

rows = list(csv.DictReader(open(CORR, encoding="utf-8-sig")))
before = len(rows)
# remove a linha do test excluido
rows = [r for r in rows if r["test_image"] != REMOVED_TEST]
removed = before - len(rows)
print(f"removida da correspondencia a imagem de test excluida ({removed} linha): {REMOVED_TEST[:34]}")

# adiciona Snap-21 como baseline_alternativo
rows.append({
    "test_image": "(baseline_alternativo)", "cell_line": "SKOV-3",
    "group_key": "P2|CARBO|F5", "timepoint_h": 0,
    "raw_file_original": "Snap-21.tiff", "raw_md5": h,
    "whst_input_file": dest_name, "is_duplicate_of": "",
    "is_baseline": "alternativo",
    "baseline_note": "baseline_alternativo do P2|CARBO|F5 (primario Snap-20, conf. baixa; analise de sensibilidade na Etapa 4)",
})

cols = ["test_image", "cell_line", "group_key", "timepoint_h", "raw_file_original",
        "raw_md5", "whst_input_file", "is_duplicate_of", "is_baseline", "baseline_note"]
with open(CORR, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["cell_line"], r["group_key"], int(r["timepoint_h"]),
                                            r["is_baseline"])))

# ---- verificacao final ----
tiffs = [f for f in os.listdir(OUT) if f.lower().endswith(".tiff")]
n_ci = len({f.lower() for f in tiffs})
from collections import Counter
bl = Counter(r["is_baseline"] for r in rows)
print(f"\n=== whst_input/ final ===")
print(f"  TIFFs: {len(tiffs)}  (case-insensitive unicos: {n_ci})")
print(f"  correspondencia: {len(rows)} linhas | is_baseline: {dict(bl)}")
assert len(tiffs) == n_ci, "COLISAO de nome!"
assert len(tiffs) == 223, f"esperado 223 TIFFs, obtido {len(tiffs)}"
print(f"  ASSERT ok: 223 TIFFs, no collision")
