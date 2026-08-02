# -*- coding: utf-8 -*-
"""
stage2/preflight.py — verificação obrigatória ANTES de gastar 34 h de GPU.

Escrito depois de perder 34 h porque o patch de padding valia no processo pai e
not in the workers. `stage2/verify_padding.py` missed it: it built its OWN dataset in the
processo principal — media o lugar certo no processo errado.

PRINCIPLE. Verification in a parallel pipeline does not count. What counts is:
  (a) proving the invariant INSIDE the process that does the work, and
  (b) comparar as SAÍDAS de treinos reais curtos.

O canário (fase E) treina de fato, por 2 épocas, chamando o mesmo
the grid's `stage2/train_config.py`, and asserts three things the broken grid did NOT
satisfazia:

    black != white   -> the padding reaches training   (this is what was failing)
    seed42 != seed43 -> o seed muda alguma coisa
    black == black   -> mesma config, mesmo resultado (determinismo)

Custa ~6 min. A grade custa 34 h.

    python stage2/preflight.py            # tudo
    python stage2/preflight.py --rapido   # skips the canary (does NOT clear the grid)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = sys.executable
RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(RAIZ)
CANARIO = os.path.join("runs", "segment", "_preflight")

falhas, avisos = [], []


def ok(cond, rot, extra="", fatal=True):
    marca = "ok " if cond else ("FALHA" if fatal else "aviso")
    print(f"  [{marca}] {rot}{('  ' + extra) if extra else ''}")
    if not cond:
        (falhas if fatal else avisos).append(rot)
    return cond


def info(rot, extra=""):
    """Informational only. NOT a check.

    Existe porque quatro chamadas a ok() eram decorativas — `ok(True, ...)` e
    `ok(cond or True, ...)` printed "[ok ]" without being able to fail. A check that
    cannot fail is worse than no check: it creates false confidence, which is
    exatamente como o stage2/verify_padding.py deixou o D12 passar. Marcar como info
    torna a distincao visivel na saida.
    """
    print(f"  [info ] {rot}{('  ' + extra) if extra else ''}")


def sec(t):
    print(f"\n{t}\n" + "-" * len(t))


# ══════════════════════════════════════════════════════════ A. ambiente
def fase_ambiente():
    sec("A. AMBIENTE")
    import torch
    ok(torch.cuda.is_available(), "GPU disponível",
       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    import ultralytics
    info(f"ultralytics {ultralytics.__version__}")
    info(f"torch {torch.__version__} · cuda {torch.version.cuda}")

    livre = shutil.disk_usage(RAIZ).free / 2**30
    # 25 runs x (best+last); yolo11x-seg pesa ~120 MB por run no pior caso
    ok(livre > 20, f"espaço em disco: {livre:.0f} GB livres", "(mínimo exigido: 20 GB)")

    for m in ("yolo11s-seg", "yolo11m-seg", "yolo11x-seg"):
        p = f"{m}.pt"
        ok(os.path.isfile(p), f"checkpoint COCO presente: {p}",
           f"({os.path.getsize(p)/2**20:.0f} MB)" if os.path.isfile(p) else "")

    # The `scratch` arm instantiates YOLO("<model>.yaml"). If Ultralytics does not
    # resolver esse nome, 5 runs falham depois de horas de fila. A checagem
    # anterior era `os.path.isfile(...) or True` — sempre passava.
    from ultralytics import YOLO
    for m in ("yolo11s-seg", "yolo11m-seg", "yolo11x-seg"):
        try:
            n = sum(x.numel() for x in YOLO(f"{m}.yaml").model.parameters())
            ok(n > 0, f"arquitetura {m}.yaml constrói (braço scratch)",
               f"({n/1e6:.1f} M parâmetros)")
        except Exception as e:
            ok(False, f"arquitetura {m}.yaml constrói (braço scratch)",
               f"-> {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════ B. dados
def fase_dados():
    sec("B. DADOS E PARTIÇÃO")
    import yaml
    d = yaml.safe_load(open("data.yaml", encoding="utf-8"))
    # mesmo criterio do unet_data.py: `path` relativo e relativo ao YAML
    aqui = os.path.dirname(os.path.abspath("data.yaml"))
    base = d.get("path") or aqui
    if not os.path.isabs(base):
        base = os.path.join(aqui, base)
    cont = {}
    for k, esperado in (("train", 932), ("val", 197), ("test", 234)):
        p = d[k] if os.path.isabs(d[k]) else os.path.join(base, d[k])
        imgs = [f for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                for f in glob.glob(os.path.join(p, e))]
        lab = glob.glob(os.path.join(p.replace("images", "labels"), "*.txt"))
        cont[k] = len(imgs)
        ok(len(imgs) == esperado, f"{k}: {len(imgs)} images", f"(expected {esperado})")
        ok(len(lab) == esperado, f"{k}: {len(lab)} rótulos", f"(esperado {esperado})")
    ok(sum(cont.values()) == 1363, f"total ativo = {sum(cont.values())}", "(esperado 1363)")

    M = list(csv.DictReader(open("data/mapping_dataset_final_strat.csv", encoding="utf-8-sig")))
    A = [r for r in M if r["partition"] in ("train", "val", "test")]
    from collections import defaultdict
    for col, n_esp in (("group_key", 265), ("split_key", 246)):
        g = defaultdict(set)
        for r in A:
            g[r[col]].add(r["partition"])
        cruzam = sum(1 for v in g.values() if len(v) > 1)
        ok(cruzam == 0, f"{col}: {len(g)} chaves, {cruzam} cruzando partição",
           "(tem de ser 0 — é a exigência central da partição)")

    # Negatives: an EMPTY label (image with no annotated wound) != a MISSING label.
    # A checagem anterior era ok(True, ...) — reportava o número e nunca falhava.
    vaz = sum(1 for f in glob.glob(os.path.join(base, "labels", "train", "*.txt"))
              if os.path.getsize(f) == 0)
    ok(vaz == 103, f"negativos em train (rótulo vazio): {vaz}", "(esperado 103)")
    ausentes = sum(1 for f in glob.glob(os.path.join(base, "images", "train", "*.*"))
                   if not os.path.isfile(os.path.join(
                       base, "labels", "train",
                       os.path.splitext(os.path.basename(f))[0] + ".txt")))
    ok(ausentes == 0, f"train images with no label file: {ausentes}",
       "(tem de ser 0 — ausente ≠ negativo)")


# ══════════════════════════════════════════════════════════ C. patch
def fase_patch():
    sec("C. PATCH DE PADDING (processo pai + workers)")
    r = subprocess.run([PY, "stage2/test_padding_patch.py"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    linhas = [l for l in (r.stdout or "").splitlines() if "[ok ]" in l or "[FALHA]" in l]
    for l in linhas:
        print("   " + l.strip())
    ok(r.returncode == 0, f"stage2/test_padding_patch.py ({len(linhas)} verificações)",
       "" if r.returncode == 0 else "-> ver saída acima")


# ══════════════════════════════════════════════════════════ D. avaliação
def fase_avaliacao():
    sec("D. EVALUATION PIPELINE (no use training if you cannot measure)")
    for s, rot in ((["stage3/test_ap_core.py"], "núcleo do AP (9 casos)"),):
        r = subprocess.run([PY] + s, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        ok(r.returncode == 0, rot)
    ok(os.path.isfile("stage3/eval_test.py") and os.path.isfile("stage3/aggregate.py"),
       "stage3/eval_test.py e stage3/aggregate.py presentes")

    # guard: the pre-correction runs MUST be refused by the evaluation
    arq = os.path.join("runs", "segment", "runs_revision_D12_invalido")
    if os.path.isdir(arq):
        import json
        velhos = glob.glob(os.path.join(arq, "*", "provenance.json"))
        com_ev = sum(1 for p in velhos
                     if "evidencia_padding_no_batch" in json.load(open(p, encoding="utf-8")))
        ok(len(velhos) > 0 and com_ev == 0,
           f"archived pre-correction runs still lack padding evidence: {com_ev}/{len(velhos)}",
           "(this is what makes stage3/eval_test.py refuse them)")

    # U-Net comparator: the two tests that closed the augmentation defect
    for s, rot in ((["unet_comparator/smoke_test.py"], "U-Net · smoke test"),
                   (["unet_comparator/test_epoca_workers.py"],
                    "U-Net · augmentation varies by epoch inside the workers")):
        if not os.path.isfile(s[0]):
            ok(False, rot, "-> arquivo ausente", fatal=False)
            continue
        r = subprocess.run([PY] + s, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        ok(r.returncode == 0, rot, "" if r.returncode == 0 else "-> falhou", fatal=False)


# ══════════════════════════════════════════════════════════ E. canário
def treina_canario(padding, seed, epochs, nome):
    d = os.path.join(CANARIO, nome)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run(
        [PY, "stage2/train_config.py", "--model", "yolo11m-seg", "--padding", padding,
         "--init", "coco", "--seed", str(seed), "--epochs", str(epochs),
         "--batch", "4", "--workers", "2", "--patience", "1000",
         "--project", "_preflight", "--name", nome],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r, d


def pesos(d):
    import torch
    p = os.path.join(d, "weights", "last.pt")
    if not os.path.isfile(p):
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return {k: v.float() for k, v in ck["model"].state_dict().items()
            if v.dtype.is_floating_point}


def iguais(a, b):
    import torch
    if a is None or b is None:
        return None
    ks = [k for k in a if k in b and a[k].shape == b[k].shape]
    return sum(1 for k in ks if torch.equal(a[k], b[k])), len(ks)


def fase_canario(epochs):
    sec(f"E. CANARY — a real {epochs}-epoch training run (the test that was missing)")
    print("   Treina de verdade, pelo mesmo stage2/train_config.py da grade, e compara")
    print("   the OUTPUTS. It is the only way to prove the padding reaches training.\n")

    plano = [("black", 42, "cn_black_42"), ("white", 42, "cn_white_42"),
             ("black", 43, "cn_black_43"), ("black", 42, "cn_black_42_bis")]
    saidas = {}
    for pad, seed, nome in plano:
        t = time.time()
        r, d = treina_canario(pad, seed, epochs, nome)
        bom = r.returncode == 0 and os.path.isfile(os.path.join(d, "weights", "last.pt"))
        print(f"   {nome:<18} {pad:<6} seed{seed}  {'ok' if bom else 'FALHOU'}  "
              f"({time.time()-t:.0f}s)")
        if not bom:
            print((r.stderr or "")[-1200:])
            falhas.append(f"canary {nome} did not train")
            return
        saidas[nome] = d

    print()
    # ---- evidência gravada de dentro do treino
    for nome in ("cn_black_42", "cn_white_42"):
        p = os.path.join(saidas[nome], "provenance.json")
        ev = json.load(open(p, encoding="utf-8")).get("evidencia_padding_no_batch", {})
        alvo = "frac_0" if "black" in nome else "frac_255"
        ok(ev and "erro" not in ev, f"{nome}: evidência gravada do batch real",
           f"0={100*ev.get('frac_0',0):.2f}% 114={100*ev.get('frac_114',0):.2f}% "
           f"255={100*ev.get('frac_255',0):.2f}%")
        ok(ev.get(alvo, 0) > 0.005,
           f"{nome}: padding pedido presente no batch REAL de treino",
           f"({alvo}={100*ev.get(alvo,0):.2f}%)")
        ok(ev.get("frac_114", 0) < 0.5 * ev.get(alvo, 1e-9),
           f"{nome}: grey 114 does NOT dominate",
           f"(114={100*ev.get('frac_114',0):.2f}%)")
        ok(not os.path.isfile(os.path.join(saidas[nome], "AVISO_PADDING.txt")),
           f"{nome}: no AVISO_PADDING")

    print()
    # ---- os três invariantes
    wb42, ww42 = pesos(saidas["cn_black_42"]), pesos(saidas["cn_white_42"])
    wb43 = pesos(saidas["cn_black_43"])
    wb42b = pesos(saidas["cn_black_42_bis"])

    ig, tot = iguais(wb42, ww42)
    ok(ig < tot, "INVARIANTE 1 — black ≠ white (o padding chega ao treino)",
       f"({ig}/{tot} tensores iguais; a grade quebrada dava {tot}/{tot})")

    ig, tot = iguais(wb42, wb43)
    ok(ig < tot, "INVARIANTE 2 — seed42 ≠ seed43 (o seed muda algo)",
       f"({ig}/{tot} iguais)")

    ig, tot = iguais(wb42, wb42b)
    ok(ig == tot, "INVARIANTE 3 — mesma config duas vezes = idêntico (determinismo)",
       f"({ig}/{tot} iguais)")

    # ---- the training losses have to diverge, not only the weights
    def loss(d):
        rows = list(csv.DictReader(open(os.path.join(d, "results.csv"), encoding="utf-8")))
        c = next(x for x in rows[0] if "train/box_loss" in x)
        return [float(r[c]) for r in rows]
    lb, lw = loss(saidas["cn_black_42"]), loss(saidas["cn_white_42"])
    idem = sum(1 for x, y in zip(lb, lw) if x == y)
    ok(idem == 0, "INVARIANTE 4 — a perda de treino diverge desde a 1ª época",
       f"({idem}/{len(lb)} épocas idênticas; a grade quebrada dava 100/100)")


# ══════════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="pula o canário")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--manter", action="store_true", help="do not delete the canary")
    args = ap.parse_args()

    print("=" * 74)
    print("PREFLIGHT — verificação antes de 34 h de GPU")
    print("=" * 74)

    fase_ambiente()
    fase_dados()
    fase_patch()
    fase_avaliacao()
    if args.rapido:
        print("\n[--rapido] canary SKIPPED — this does NOT clear the grid.")
        avisos.append("canary not executed")
    else:
        fase_canario(args.epochs)
        if not args.manter and os.path.isdir(CANARIO):
            shutil.rmtree(CANARIO, ignore_errors=True)
            print(f"\n   (canário removido: {CANARIO})")

    print("\n" + "=" * 74)
    if falhas:
        print(f"DO NOT TRAIN — {len(falhas)} failure(s):")
        for f in falhas:
            print("   ✗", f)
        sys.exit(1)
    if avisos:
        print(f"{len(avisos)} aviso(s):")
        for a in avisos:
            print("   !", a)
    print("PREFLIGHT OK" + (" (com avisos)" if avisos else "") + " — pode rodar a grade.")
    if args.rapido:
        sys.exit(2)


if __name__ == "__main__":
    main()
