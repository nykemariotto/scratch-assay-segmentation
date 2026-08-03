# -*- coding: utf-8 -*-
"""Indexa por MD5 HUVEC-RAW e Banco de dados/HUVEC, salvando cache JSON completo."""
import hashlib, os, json, re

BD = os.environ.get("BANCO_A", "<banco_a>") + r"\HUVEC"
RAW = os.environ.get("BANCO_A", "<banco_a>") + r"\HUVEC-RAW"
IMG = (".tif", ".tiff", ".bmp", ".png")
OUT = os.environ.get("SCRATCH_ASSAY_ROOT", ".") + r"\hash_cache_huvec.json"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def walk(root, is_raw):
    recs = []
    for dirpath, _, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        batch = parts[0] if parts else None
        tp = None
        treatment = None
        if parts:
            if re.match(r"^\d+\s*hr?$", parts[-1], re.I):
                tp = parts[-1]
                if len(parts) >= 3:
                    treatment = parts[-2]
        for f in files:
            if f.lower().endswith(IMG) and "scale" not in f.lower():
                p = os.path.join(dirpath, f)
                try:
                    h = md5(p)
                except Exception as e:
                    print("ERRO", p, e)
                    continue
                recs.append(dict(rel=rel, batch=batch, treatment=treatment,
                                 tp=tp, name=f, md5=h,
                                 size=os.path.getsize(p)))
    return recs


if __name__ == "__main__":
    raw = walk(RAW, True)
    print("RAW", len(raw), flush=True)
    bd = walk(BD, False)
    print("BD", len(bd), flush=True)
    json.dump({"raw": raw, "bd": bd}, open(OUT, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("ok")
