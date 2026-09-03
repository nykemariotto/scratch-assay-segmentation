# -*- coding: utf-8 -*-
"""
tira_enfase_caps.py — desfaz a caixa alta de enfase nos comentarios e docstrings.

O QUE E, E POR QUE SAI. Escrever "does NOT change", "correcting ALL frames", "NEVER
opens" e abrir paragrafo com "WHY." ou "THE PROBLEM." e um tique — 1.144 ocorrencias
em 134 arquivos do deposito. Ninguem escreve assim por habito proprio; e marca de texto
gerado por modelo, e o deposito e publico e citavel.

O que isto NAO e: camuflagem. O §2.10 do manuscrito declara o uso de IA em detalhe, e e
la que a declaracao mora — a solucao e transparencia, nao esconder. Limpar prosa com cara
de maquina e edicao de qualidade; apagar a declaracao seria outra coisa, e nao se faz.

DUAS FORMAS, DUAS REGRAS:
  · no meio da frase, precedida de minuscula  -> minuscula   ("does NOT" -> "does not")
  · abrindo linha ou frase, como pseudo-titulo -> capitalizada ("WHY." -> "Why.")

O QUE FICA INTOCADO: sigla (YOLO, CCC, MD5), identificador de codigo, string literal,
nome de arquivo, e qualquer coisa fora de comentario ou docstring. Nada aqui muda uma
linha executavel — mas a verificacao por execucao roda depois assim mesmo, porque
"e so comentario" e exatamente o que se diz antes de quebrar alguma coisa.

    python tira_enfase_caps.py            # mostra o que faria
    python tira_enfase_caps.py --aplicar
"""
import argparse
import collections
import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Palavras inglesas comuns que aparecem em caixa alta so por enfase. Sigla nao entra:
# a lista e fechada de proposito, porque adivinhar o que e sigla e como se erra aqui.
PALAVRAS = {
    "NOT", "THE", "THIS", "ONE", "WHY", "BEFORE", "AFTER", "TWO", "THREE", "READ",
    "WHAT", "NEVER", "ALL", "EVERY", "ONLY", "ANY", "NONE", "BOTH", "SAME",
    "DIFFERENT", "WRONG", "RIGHT", "TRUE", "FALSE", "YES", "NO", "AND", "OR",
    "BUT", "WITH", "WITHOUT", "FROM", "INTO", "OVER", "UNDER", "TRAINING", "TEST",
    "VALIDATION", "PROBLEM", "REASON", "CONSEQUENCE", "HOW", "WHEN", "WHERE",
    "WHICH", "WHO", "MUST", "CANNOT", "SHOULD", "ALWAYS", "FIRST", "SECOND",
    "LAST", "NEW", "OLD", "REAL", "EACH", "ITSELF", "AGAIN", "STILL", "ALREADY",
}
DUPLETES = [("auditably and verifiably", "auditably"),
            ("clear and explicit", "explicit"),
            ("complete and correct", "correct")]

DIRS = ("stage1", "stage2", "stage3", "stage4", "unet_comparator", "webapp",
        "data", "protocols", "coco_partitions")
INTERNO = ("verifica_", "audita_", "varredura_", "gera_", "fix_", "sync_",
           "commita_", "docx_versiona", "traduz_nomes", "tira_enfase_caps",
           "monta_publicacao", "checklist_en")


def linhas_de_prosa(texto, ext):
    """Devolve (indice, linha) das linhas que sao comentario ou estao em docstring."""
    saida, dentro = [], False
    for i, l in enumerate(texto.split("\n")):
        aspas = l.count('"""') + l.count("'''")
        if ext in (".py",):
            if dentro:
                saida.append(i)
                if aspas % 2:
                    dentro = False
                continue
            if aspas % 2:
                dentro = True
                saida.append(i)
                continue
            if l.lstrip().startswith("#"):
                saida.append(i)
        else:                                   # .md, .ijm: tudo e prosa
            saida.append(i)
    return saida


# Uma sigla legitima no meio de uma sequencia em caixa alta nao pode ser rebaixada.
SIGLAS = re.compile(r"^(YOLO\d*|COCO|IOU|CSV|JSON|GPU|CPU|MD5|DOI|API|PNG|TIFF|AGPL|MIT|"
                    r"SAM|WHST|CLAIM|GRRAS|RAW|TOST|SD|CI|OK|ID|XML|ROI|HUVEC|SKOV|UNESP|"
                    r"NOTICE|README|LICENSE|PDF|DPI|EMU|OOXML|LFS|HF|URL|ZIP|SHA|UTC|YAML|"
                    r"CFF|CCC|AP|MAP|NMS|FBS|HSV|RGB|PT|EN|BR|NA|D\d+)$")


def _frase(seq):
    """Sentence case numa sequencia inteira em caixa alta, poupando siglas."""
    ps = seq.split(" ")
    out = []
    for i, p in enumerate(ps):
        nu = p.strip(".,:;—-()")
        if SIGLAS.match(nu):
            out.append(p)
        elif i == 0:
            out.append(p.capitalize())
        else:
            out.append(p.lower())
    return " ".join(out)


# Duas ou mais palavras em caixa alta seguidas: e pseudo-titulo, trata como bloco.
# Palavra a palavra, "THE MEAN IS NO USE HERE" virava "The MEAN IS no USE HERE" —
# so as que estao na lista mudavam, e o resultado ficava pior do que o original.
SEQ = re.compile(r"(?<![A-Za-z0-9_])([A-ZÀ-Ü][A-ZÀ-Ü0-9]{1,}(?:[ ,:.-]+[A-ZÀ-Ü][A-ZÀ-Ü0-9]{1,})+)"
                 r"(?![A-Za-z0-9_])")


def conserta(l):
    def bloco(m):
        s = m.group(1)
        # se e so sigla, deixa
        if all(SIGLAS.match(p.strip(".,:;—-()")) for p in s.split() if p.strip(".,:;—-()")):
            return s
        return _frase(s)
    novo = l   # sequencias em caixa alta ficam intactas — ver nota abaixo

    def troca(m):
        p = m.group(0)
        antes, depois = novo[:m.start()], novo[m.end():]
        # So enfase INEQUIVOCA: palavra cercada de minusculas dos dois lados. Abrindo
        # linha ou frase pode ser titulo legitimo; ao lado de outra caixa alta pode ser
        # sigla composta. Estreitar ate o falso positivo ser zero e melhor do que pegar
        # mais e corromper um termo tecnico — foi o que aconteceu com RT-DETR e GB VRAM.
        if not re.search(r"[a-z]\s*$", antes):
            return p
        if not re.match(r"\s*[a-z]", depois):
            return p
        return p.lower()
    novo = re.sub(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(PALAVRAS, key=len, reverse=True))
                  + r")(?![A-Za-z0-9_])", troca, novo)
    for de, para in DUPLETES:
        novo = novo.replace(de, para)
    return novo


def arquivos():
    """So o que esta NO DEPOSITO. Varrer o repositorio inteiro alcancava documentos de
    trabalho em portugues que ninguem publica — churn com risco e sem beneficio."""
    import zipfile
    try:
        pub = {n for n in zipfile.ZipFile("_publicar/zenodo/code.zip").namelist()}
    except Exception:
        pub = None
    out = []
    for r in list(DIRS) + ["."]:
        for dp, _, fs in os.walk(r):
            dp = dp.replace("\\", "/")
            if any(x in dp for x in (".git", "_scratch", "__pycache__", "_publicar",
                                     "runs/", "dataset/", "models/", "revisao/")):
                continue
            for f in fs:
                if not f.endswith((".py", ".md", ".ijm")):
                    continue
                if any(f.startswith(x) for x in INTERNO):
                    continue
                p = (dp + "/" + f).lstrip("./")
                if pub is not None and p not in pub:
                    continue
                out.append(p)
            if r == ".":
                break
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()

    tot, arqs = 0, 0
    amostra, cont = [], collections.Counter()
    for p in arquivos():
        try:
            s = io.open(p, encoding="utf-8").read()
        except Exception:
            continue
        L = s.split("\n")
        idx = linhas_de_prosa(s, os.path.splitext(p)[1])
        mudou = False
        for i in idx:
            novo = conserta(L[i])
            if novo != L[i]:
                n = sum(1 for _ in re.finditer(
                    r"(?<![A-Za-z0-9_])(" + "|".join(PALAVRAS) + r")(?![A-Za-z0-9_])", L[i]))
                tot += n
                cont[p] += n
                if len(amostra) < 8:
                    amostra.append((p, L[i].strip()[:96], novo.strip()[:96]))
                L[i] = novo
                mudou = True
        if mudou:
            arqs += 1
            if a.aplicar:
                io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(L))

    print(f"{tot} ocorrencia(s) em {arqs} arquivo(s)\n")
    for p, v, n in amostra:
        print(f"  [{p}]\n    - {v}\n    + {n}\n")
    if not a.aplicar:
        print("  arquivos mais afetados:")
        for p, n in cont.most_common(6):
            print(f"    {n:>4}  {p}")
        print("\n(use --aplicar para efetivar; verifique por execucao depois)")
    else:
        print("aplicado. VERIFIQUE POR EXECUCAO:")
        print("  python gera_manifesto_zenodo.py --montar --so code.zip")
        print("  extrair num diretorio vazio e rodar stage3/agreement_final.py")


if __name__ == "__main__":
    main()
