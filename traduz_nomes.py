# -*- coding: utf-8 -*-
r"""
traduz_nomes.py — poe em ingles os nomes de arquivo e diretorio que vao ao deposito.

POR QUE. O deposito e citavel e internacional; um leitor que nao le portugues encontrava
`inspecao_visual.csv`, `stage3/registros/` e `concordancia_final.py` sem saber o que sao.
O conteudo dos scripts ja estava em ingles — so os nomes ficaram para tras.

O QUE NAO ENTRA. Ferramenta interna que nunca vai ao pacote (`varredura_lingua.py`,
`verifica_*.py`, `fix_*.py`) fica como esta: renomear o que ninguem publica e churn.

O RISCO, E COMO ELE E CONTIDO. Renomear arquivo que o codigo LE quebra a reproducao —
e a reproducao deste deposito foi provada por execucao, entao ha o que quebrar. Por isso
o script (1) renomeia, (2) reescreve TODA referencia textual em .py/.md/.txt/.yml/.ijm,
e (3) o operador roda em seguida o mesmo teste de sempre: extrair o code.zip num
diretorio vazio e rodar `stage3/agreement_final.py`. Se os numeros do artigo saem
iguais, o renome esta certo. Nao ha atalho para essa etapa.

A ordem do MAPA importa: o mais longo primeiro, senao `registros` casa dentro de
`registros_fixture` e o segundo renome nunca acontece.

    python traduz_nomes.py --ver      # mostra o que faria
    python traduz_nomes.py --aplicar
"""
import argparse
import io
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# (de, para) — do mais longo para o mais curto
MAPA = [
    # diretorios
    ("stage3/registros_fixture", "stage3/records_fixture"),
    ("stage3/registros", "stage3/records"),
    # dados
    ("closure_final_longo", "closure_final_long"),
    ("inspecao_visual", "visual_triage"),
    ("qc_suspeitas", "qc_flagged"),
    # stage1
    ("adjudicacao_ambiguo", "adjudication_ambiguous"),
    ("adjudicate_ambiguo", "adjudicate_ambiguous"),
    ("inspect_ambiguo", "inspect_ambiguous"),
    # stage3
    ("_concordancia_estat", "_agreement_stats"),
    ("concordancia_final_longo", "agreement_final_long"),
    ("concordancia_final", "agreement_final"),
    ("figuras_concordancia", "figures_agreement"),
    ("figuraS1_celulas_isoladas", "figureS1_isolated_cells"),
    ("figura1_fluxo", "figure1_study_flow"),
    ("figura6_exemplo", "figure6_example"),
    ("ablacao_padding", "padding_ablation"),
    ("benchmark_classico_longo", "benchmark_classical_long"),
    ("benchmark_classico", "benchmark_classical"),
    ("estatisticas_para_224", "stats_for_224"),
    ("iou_por_imagem", "iou_per_image"),
    ("tres_vias", "three_way"),
    ("unet_varredura_resumo", "unet_sweep_summary"),
    ("unet_varredura", "unet_sweep"),
    ("yolo_varredura_conf_resumo", "yolo_conf_sweep_summary"),
    ("yolo_varredura_conf", "yolo_conf_sweep"),
    ("varredura_conf", "conf_sweep"),
    ("paired_new_longo", "paired_new_long"),
    # stage4
    ("correcao_manual_validacao", "manual_correction_validation"),
    ("correcao_manual_completacao", "manual_correction_completion"),
    ("correcao_manual_baselines", "manual_correction_baselines"),
    ("correcao_manual_pass", "manual_correction_pass"),
    # unet_comparator
    ("diag_validacao", "diag_validation"),
    ("varredura_limiar", "threshold_sweep"),
]

# Nunca renomear nem reescrever nestes: sao ferramenta interna, fora do deposito.
FORA = ("varredura_lingua.py", "varredura_pre_deploy.py", "verifica_", "audita_",
        "gera_", "fix_", "sync_", "commita_revisao.py", "docx_versiona.py",
        "traduz_nomes.py", "monta_publicacao.py", "checklist_en.py")

DIRS = ("stage1", "stage2", "stage3", "stage4", "unet_comparator", "webapp",
        "data", "protocols", "coco_partitions", ".github")
TEXTO = (".py", ".md", ".txt", ".yml", ".yaml", ".ijm", ".csv", ".json", ".cff")


# NUNCA TOCAR. O gabarito da triagem cega e bloqueado por nome, na lista PROIBIDO de
# `gera_manifesto_zenodo.py` — que este script nao reescreve, por estar em FORA. Renomea-lo
# aqui deixaria o bloqueio apontando para um arquivo que nao existe mais, e o gabarito
# passaria a entrar no deposito: publicar a chave anularia a cegueira que o metodo declara.
# Os .bak sao backup de trabalho: renomea-los e risco sem beneficio.
INTOCAVEL = ("TRIAGEM_CEGA", "recorrecao_oculta", "validacao_gabarito", ".bak")


def interno(p):
    b = os.path.basename(p)
    if any(x in p for x in INTOCAVEL):
        return True
    return any(b.startswith(f) or b == f for f in FORA)


def alvos():
    """Arquivos cujo NOME muda, e arquivos cujo CONTEUDO pode citar os nomes."""
    nomes, textos = [], []
    for r in list(DIRS) + ["."]:
        for dp, dns, fs in os.walk(r):
            dp = dp.replace("\\", "/")
            if any(x in dp for x in (".git", "_scratch", "__pycache__", "dataset/",
                                     "runs/", "models/", "node_modules", "_publicar",
                                     "revisao/05_manuscrito")):
                continue
            for f in fs:
                p = (dp + "/" + f).lstrip("./")
                if interno(p):
                    continue
                if any(de in f for de, _ in MAPA):
                    nomes.append(p)
                if f.endswith(TEXTO):
                    textos.append(p)
            if r == ".":
                break
    return sorted(set(nomes)), sorted(set(textos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()

    nomes, textos = alvos()
    # diretorios a renomear, do mais fundo para o mais raso
    dirs = [(de, para) for de, para in MAPA if os.path.isdir(de)]

    print(f"{len(nomes)} arquivo(s) mudam de nome · {len(dirs)} diretorio(s) · "
          f"{len(textos)} arquivo(s) de texto varridos por referencia\n")

    refs = 0
    for p in textos:
        try:
            s = io.open(p, encoding="utf-8").read()
        except Exception:
            continue
        n = sum(s.count(de) for de, _ in MAPA)
        refs += n
    print(f"{refs} referencia(s) textuais a reescrever\n")

    if not a.aplicar:
        for p in nomes[:12]:
            novo = p
            for de, para in MAPA:
                novo = novo.replace(de, para)
            print(f"  {p}\n      -> {novo}")
        if len(nomes) > 12:
            print(f"  … +{len(nomes)-12}")
        print("\n(use --aplicar para efetivar)")
        return

    # 1. conteudo primeiro: enquanto os caminhos antigos ainda existem
    for p in textos:
        try:
            s = io.open(p, encoding="utf-8").read()
        except Exception:
            continue
        o = s
        for de, para in MAPA:
            s = s.replace(de, para)
        if s != o:
            io.open(p, "w", encoding="utf-8", newline="\n").write(s)

    # 2. arquivos
    for p in nomes:
        novo = p
        for de, para in MAPA:
            novo = novo.replace(de, para)
        if novo != p and os.path.isfile(p):
            os.makedirs(os.path.dirname(novo) or ".", exist_ok=True)
            os.replace(p, novo)

    # 3. diretorios
    for de, para in dirs:
        if os.path.isdir(de):
            os.replace(de, para)

    print("aplicado. AGORA VERIFIQUE POR EXECUCAO:")
    print("  python gera_manifesto_zenodo.py --montar --so code.zip")
    print("  extrair o zip num diretorio vazio e rodar stage3/agreement_final.py")


if __name__ == "__main__":
    main()
