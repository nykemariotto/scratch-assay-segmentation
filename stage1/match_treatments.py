# -*- coding: utf-8 -*-
"""Casa por hash MD5 os snaps anotados (Banco de dados/SKOV) com as pastas de
tratamento P1 (geo/ptx) e P2 (carbo), recuperando tratamento+experimento+timepoint.
NAO usa feature matching — identidade exata de arquivo."""
import hashlib, os, json
from collections import defaultdict, Counter

BD = os.environ.get("BANCO_A", "<banco_a>") + r"\SKOV"
P1 = r"G:\.shortcut-targets-by-id\16gG82kKalY_NsrHcFf6mK3smW3pw5zmr\Wound Healing"
P2 = r"G:\.shortcut-targets-by-id\1CMwTWGfjZgxB1XuvetRHUONTxDknvhDU\WH_Carbo_25-10_27-10-22"
TPS = ["0h", "24h", "48h", "72h"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_flat(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        if f.lower().endswith((".tif", ".tiff")) and "scale" not in f.lower():
            try:
                out[md5(os.path.join(d, f))] = f
            except Exception:
                pass
    return out


def hash_treatments(root, tp):
    """Retorna hash -> (treatment, filename) para uma pasta timepoint com subpastas."""
    out = {}
    base = os.path.join(root, tp)
    if not os.path.isdir(base):
        return out
    for tr in os.listdir(base):
        trd = os.path.join(base, tr)
        if os.path.isdir(trd):
            for hh, fn in hash_flat(trd).items():
                out[hh] = (tr, fn)
    return out


def main():
    report = {}
    for tp in TPS:
        bd = hash_flat(os.path.join(BD, tp))
        p1 = hash_treatments(P1, tp)
        p2 = hash_treatments(P2, tp)
        m1 = {h: p1[h] for h in bd if h in p1}
        m2 = {h: p2[h] for h in bd if h in p2}
        only_both = [h for h in bd if h in p1 and h in p2]
        unmatched = [bd[h] for h in bd if h not in p1 and h not in p2]
        report[tp] = {
            "bd_snaps": len(bd),
            "match_P1": len(m1),
            "match_P2": len(m2),
            "match_ambos": len(only_both),
            "sem_match": len(unmatched),
            "unmatched_examples": unmatched[:10],
            "P1_treatments_hit": dict(Counter(t for t, _ in m1.values())),
            "P2_treatments_hit": dict(Counter(t for t, _ in m2.values())),
        }
        print(f"=== {tp} === BD={len(bd)}  P1_match={len(m1)}  P2_match={len(m2)}  "
              f"ambos={len(only_both)}  sem_match={len(unmatched)}")
        if m1:
            print(f"    P1 tratamentos: {dict(Counter(t for t,_ in m1.values()))}")
        if m2:
            print(f"    P2 tratamentos: {dict(Counter(t for t,_ in m2.values()))}")
        if unmatched:
            print(f"    sem match (amostra): {unmatched[:8]}")

    with open("stage1/match_treatments.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nSalvo: stage1/match_treatments.json")


if __name__ == "__main__":
    main()
