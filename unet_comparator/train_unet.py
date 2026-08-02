# -*- coding: utf-8 -*-
"""
train_unet.py — treina UMA U-Net com seed explícito, espelhando stage2/train_config.py.

Mesmo contrato da grade YOLO: mesma partição, 640x640, padding preto, 100 épocas,
batch 4, sem early stopping, determinismo forçado, e os mesmos artefatos de
proveniência (`provenance.json`, `pip_freeze.txt`, `COMPLETED.json`) para que o
downstream trate os dois braços igual.

Uso:
    python unet_comparator/train_unet.py --seed 42
    python unet_comparator/run_unet_grid.py          # os 5 seeds

NÃO RODAR ENQUANTO A GRADE YOLO ESTIVER OCUPANDO A GPU.
"""
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from unet_data import WoundDataset, carrega_splits
from unet_model import UNet, BCEDiceLoss


def set_deterministic(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def avalia(modelo, loader, dev, limiar=0.5):
    """IoU e Dice médios por imagem. Imagem sem ferida (negativo) conta 1.0 se a
    predição também for vazia — a convenção que o stage4/correction_agreement.py usa."""
    modelo.eval()
    ious, dices = [], []
    for x, y, _, _ in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        p = (torch.sigmoid(modelo(x)) > limiar).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        uni = ((p + y) > 0).float().sum(dim=(1, 2, 3))
        soma = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        for i in range(x.shape[0]):
            ious.append(1.0 if uni[i] == 0 else (inter[i] / uni[i]).item())
            dices.append(1.0 if soma[i] == 0 else (2 * inter[i] / soma[i]).item())
    return float(np.mean(ious)), float(np.mean(dices))


def diagnostica_csv(rcsv, epochs_esperadas, min_seq=3):
    """Le o results.csv e decide se o run e valido.

    A versao antiga da sentinela so CONTAVA linhas. Foi por isso que o seed42 saiu
    com `"status": "ok"` tendo 60 epocas de train_loss = NaN — 100 linhas
    presentes, 100 epocas declaradas. Contar linha nao e verificar resultado.

    Esta funcao existe SEPARADA do treino de proposito: um verificador que so pode
    ser exercitado rodando 3 h de GPU nao e testavel, e um verificador nao testado
    e exatamente o que deixou o D12 e o estouro de fp16 passarem. Ver
    test_sentinela.py, que a roda contra o CSV real do run destruido.

    Devolve um dicionario de diagnostico; as chaves `_ruins` e `_degeneradas`
    trazem as listas de epocas e sao consumidas por quem chama.
    """
    with open(rcsv, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    ruins = [int(r["epoch"]) for r in linhas
             if not all(math.isfinite(_f(r[c]))
                        for c in ("train_loss", "val_iou", "val_dice"))]

    # IoU == Dice e assinatura de predicao degenerada (toda-zeros ou toda-uns):
    # Dice = 2*IoU/(1+IoU), logo so coincidem em 0 ou 1. Mas as colunas sao MEDIAS
    # sobre imagens, e duas medias podem coincidir por acaso — com 6 casas no CSV
    # isso tem ~1e-6 de chance por epoca. Reprovar um run inteiro por UMA
    # coincidencia seria falso positivo; exigimos `min_seq` epocas CONSECUTIVAS,
    # que e o que a falha real produz (no seed42 foram 60 seguidas).
    marca = [math.isfinite(_f(r["val_iou"]))
             and abs(_f(r["val_iou"]) - _f(r["val_dice"])) < 1e-9
             and 0.0 < _f(r["val_iou"]) < 1.0 for r in linhas]
    degeneradas, seq = set(), []
    for r, m in zip(linhas, marca):
        seq = seq + [int(r["epoch"])] if m else []
        if len(seq) >= min_seq:
            degeneradas |= set(seq)
    degeneradas = sorted(degeneradas)

    return {"requested_epochs": epochs_esperadas,
            "completed_epochs": len(linhas),
            "stopped_early": len(linhas) < epochs_esperadas,
            "nan_epochs": len(ruins),
            "primeira_epoca_nao_finita": ruins[0] if ruins else None,
            "epocas_iou_igual_dice": len(degeneradas),
            "final_val_iou": round(_f(linhas[-1]["val_iou"]), 6) if linhas else None,
            "_ruins": ruins, "_degeneradas": degeneradas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--padding", default="black", choices=["black", "white"])
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--project", default=os.path.join("runs", "segment", "unet_comparator"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    fill = 0 if args.padding == "black" else 255
    run = args.name or f"unet_{args.padding}_seed{args.seed}"
    outdir = os.path.join(args.project, run)
    os.makedirs(outdir, exist_ok=True)

    set_deterministic(args.seed)
    splits = carrega_splits(args.data)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds_tr = WoundDataset(splits["train"], args.imgsz, treino=True, fill=fill, seed=args.seed)
    ds_va = WoundDataset(splits["val"], args.imgsz, treino=False, fill=fill)
    # Gerador do shuffle com semente FIXA, nao com o seed do treino.
    # Espelha o Ultralytics (D13): la a ordem dos dados nao varia com o seed.
    # Se aqui variasse, o braco U-Net teria uma fonte de variancia a mais que o
    # braco YOLO e os desvios-padrao da Table 2 nao seriam comparaveis.
    from unet_data import SEED_AUG
    g = torch.Generator()
    g.manual_seed(SEED_AUG)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=(dev == "cuda"), drop_last=False, generator=g)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=(dev == "cuda"))
    print(f"{run} · treino {len(ds_tr)} · val {len(ds_va)} · device {dev}")

    modelo = UNet().to(dev)
    crit = BCEDiceLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=5e-4)
    # WARMUP — o braco YOLO tem `warmup_epochs: 3.0` (esta no args.yaml de todo run
    # da grade); o comparador nao tinha nenhum. Alem da assimetria de projeto, era a
    # fonte provavel da instabilidade: o CosineAnnealingLR partia de lr=1e-3 ja na
    # primeira epoca, sobre pesos aleatorios, com batch 4. Os tres runs colapsaram
    # nas ~20 primeiras epocas (seed43 seis vezes e sobreviveu; seed44 tres vezes e
    # morreu com NaN na epoca 13) — quais seeds completam virava sorteio, e reportar
    # so os sobreviventes seria vies de sobrevivencia.
    #
    # O warmup e por ITERACAO, nao por epoca: com 233 lotes por epoca, 3 epocas dao
    # 699 passos de rampa, o que e suave. Por epoca seriam 3 degraus.
    WARMUP_EPOCAS = 3
    # a cosseno so comeca a contar DEPOIS do warmup, e parte de args.lr
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - WARMUP_EPOCAS))
    escala = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    rcsv = os.path.join(outdir, "results.csv")
    with open(rcsv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_iou", "val_dice", "lr", "seconds"])

    passos_warmup = WARMUP_EPOCAS * len(dl_tr)
    passo = 0
    print(f"warmup: {WARMUP_EPOCAS} épocas = {passos_warmup} iterações, "
          f"de {args.lr/100:.2e} a {args.lr:.2e}", flush=True)

    melhor, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        ds_tr.set_epoca(ep)          # OBRIGATORIO: sem isto a augmentation congela
        modelo.train()
        te, perdas = time.time(), []
        for i_lote, (x, y, _, _) in enumerate(dl_tr):
            # rampa linear de lr durante o warmup, aplicada POR ITERACAO
            if passo < passos_warmup:
                fator = 0.01 + 0.99 * (passo + 1) / passos_warmup
                for gp in opt.param_groups:
                    gp["lr"] = args.lr * fator
            passo += 1
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                perda = crit(modelo(x), y)

            # ABORTA no primeiro NaN/Inf — nao adianta pular o lote e seguir.
            # O GradScaler protege o opt.step(), mas NAO os buffers do BatchNorm:
            # running_mean/running_var sao atualizados no FORWARD. Assim que uma
            # ativacao vira inf/NaN, os buffers absorvem NaN e o modelo esta morto,
            # ainda que os pesos sigam finitos. Foi o que aconteceu no seed42: as
            # 100 epocas "terminaram", 60 delas com o modelo ja destruido.
            v = float(perda.detach())
            if not math.isfinite(v):
                falha = {"status": "diverged", "run": run, "epoch": ep,
                         "batch": i_lote, "loss": str(v),
                         "epochs_completed_before": ep - 1,
                         "motivo": "loss nao-finita; treino abortado. Buffers de "
                                   "BatchNorm sao poluidos no forward e nao se "
                                   "recuperam — continuar produziria um run morto "
                                   "que mesmo assim reportaria 100 epocas."}
                json.dump(falha, open(os.path.join(outdir, "FAILED.json"), "w",
                                      encoding="utf-8"), indent=2, ensure_ascii=False)
                sys.exit(f"\nABORTADO: loss {v} na epoca {ep}, lote {i_lote}. "
                         f"FAILED.json gravado em {outdir}")

            escala.scale(perda).backward()
            # unscale_ antes do clipping: o clipping precisa do gradiente na
            # escala real, senao o max_norm nao significa nada.
            escala.unscale_(opt)
            # max_norm=10.0 e o valor que o Ultralytics usa em
            # BaseTrainer.optimizer_step (v8.4.102, engine/trainer.py). Copiar o
            # numero, em vez de escolher um, mantem o comparador alinhado ao braco
            # YOLO — um max_norm diferente seria uma assimetria a defender diante
            # da checagem sem nenhum ganho.
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=10.0)
            escala.step(opt)
            escala.update()
            perdas.append(v)
        # a cosseno so age DEPOIS do warmup; durante ele, o lr e a rampa acima
        if ep > WARMUP_EPOCAS:
            sched.step()

        # LIBERA O CACHE DO ALOCADOR ANTES DE VALIDAR.
        # Sem isto a validacao levava 181-267 s contra 14,5 s — 12,5x. O treino roda
        # em fp16 sob autocast e deixa ~7,0 GB dos 8,6 GB reservados no cache; a
        # validacao roda em fp32, FORA do autocast, e pede o dobro de memoria por
        # amostra. O alocador tinha de devolver blocos ao driver e pedir outros, e
        # cada cudaMalloc/cudaFree serializa — dai a GPU marcar 100% de utilizacao
        # puxando so 53 W de 160 W: nao era computo, era o alocador.
        # O resultado e BIT A BIT IDENTICO (IoU 0,500330 com e sem). Medido em
        # diag_validacao2.py, que compara as quatro condicoes.
        if dev == "cuda":
            torch.cuda.empty_cache()
        iou, dice = avalia(modelo, dl_va, dev)
        dt = time.time() - te
        with open(rcsv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, round(float(np.mean(perdas)), 6),
                                    round(iou, 6), round(dice, 6),
                                    opt.param_groups[0]["lr"], round(dt, 1)])
        if iou > melhor:
            melhor = iou
            torch.save({"model": modelo.state_dict(), "epoch": ep, "val_iou": iou,
                        "seed": args.seed, "padding": args.padding}, os.path.join(outdir, "best.pt"))
        print(f"  época {ep:>3}/{args.epochs}  loss {np.mean(perdas):.4f}  "
              f"IoU {iou:.4f}  Dice {dice:.4f}  ({dt:.0f}s)", flush=True)

    torch.save({"model": modelo.state_dict(), "epoch": args.epochs, "seed": args.seed,
                "padding": args.padding}, os.path.join(outdir, "last.pt"))
    total = time.time() - t0

    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=120).stdout
    except Exception:
        freeze = ""
    open(os.path.join(outdir, "pip_freeze.txt"), "w", encoding="utf-8").write(freeze)
    json.dump({"run": run, "model": "unet-canonical-base64", "padding": args.padding,
               "padding_fill_value": fill, "init": "scratch", "seed": args.seed,
               "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
               "lr": args.lr, "optimizer": "AdamW", "loss": "0.5*BCE + 0.5*Dice",
               "data": args.data, "wall_seconds": round(total, 1),
               "best_val_iou": round(melhor, 6),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "nota": "reimplementacao arquitetural (Ronneberger 2015) como comparador "
                       "arquitetural; NAO e a execucao da ferramenta publicada por Dogru et al."},
              open(os.path.join(outdir, "provenance.json"), "w", encoding="utf-8"), indent=2)

    # ── sentinela ────────────────────────────────────────────────────────────
    diag = diagnostica_csv(rcsv, args.epochs)
    diag.update({"run": run, "best_val_iou": round(melhor, 6),
                 "wall_seconds": round(total, 1)})
    ruins, degeneradas = diag.pop("_ruins"), diag.pop("_degeneradas")
    feitas = diag["completed_epochs"]

    if ruins or degeneradas or feitas < args.epochs:
        diag["status"] = "diverged"
        diag["motivo"] = ("epocas nao-finitas e/ou val_iou == val_dice (predicao "
                          "degenerada). COMPLETED.json NAO gravado — o downstream "
                          "exige essa sentinela e por isso ignora este run.")
        json.dump(diag, open(os.path.join(outdir, "FAILED.json"), "w",
                             encoding="utf-8"), indent=2, ensure_ascii=False)
        sys.exit(f"\nRUN INVALIDO: {run} — {len(ruins)} epocas nao-finitas, "
                 f"{len(degeneradas)} degeneradas. FAILED.json gravado.")

    diag["status"] = "ok"
    json.dump(diag, open(os.path.join(outdir, "COMPLETED.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nConcluido: {run}  ({total/60:.1f} min)  melhor IoU val {melhor:.4f}")


if __name__ == "__main__":
    main()
