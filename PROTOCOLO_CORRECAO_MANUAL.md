# Protocolo de correção manual das medições WHST

**Manuscrito 4336348 — Cytometry Part A**
**Status: definido ANTES do início da correção.** Este documento é versionado no
repositório junto aos scripts; o commit que o introduz precede qualquer arquivo
em `whst_output/rois_corrigidos/`, o que torna verificável que o critério não foi
ajustado depois de ver os resultados.

---

## 1. Critério de borda (declarado a priori, aplicado uniformemente)

> **A borda da ferida é onde termina o monolayer confluente e começa a região
> aberta/esparsa.**

Regras operacionais:

| Situação | Decisão |
|---|---|
| Área sem cobertura celular contínua | **Dentro** da ferida |
| Células isoladas/migratórias esparsas que ainda não formam monolayer | **Dentro** da ferida |
| Cobertura celular contínua e confluente, mesmo com densidade variável | **Fora** da ferida (é monolayer) |
| Células mais avançadas isoladas, à frente do monolayer | A borda acompanha o **monolayer**, não essas células |
| Debris, bolhas, artefatos ópticos | Não contam como ferida; interpolar a borda do monolayer por baixo do artefato |

O mesmo critério vale para **todos os timepoints** e para **ambas as passadas**
de correção. O critério está replicado no cabeçalho de
`stage4/whst_manual_correction.ijm`.

---

## 2. Pré-requisito: adjudicação dos casos ambíguos

A triagem visual cega produziu 12 imagens `AMBIGUO`. Elas são adjudicadas
**antes** de a lista de correção ser fechada, porque entram no cálculo de
closure fraction e podem alterar o veredito da série.

```bash
python stage1/adjudicate_ambiguo.py --template
```
Preencher a coluna `decisao` de `stage1/adjudicacao_ambiguo.csv` com
`OK | super | sub | invalida`, usando o painel `inspect_ambiguo.png`
(vermelho = frame a adjudicar; verde = OK; amarelo = super; azul = sub).

```bash
python stage1/adjudicate_ambiguo.py --apply
python stage4/whst_series_analysis.py
python stage4/build_correction_worklist.py
```

`stage4/build_correction_worklist.py` **aborta** enquanto houver `AMBIGUO` pendente —
a ordem é imposta pelo código, não pela disciplina do operador.

---

## 3. Definição da lista de correção

Derivada de `stage4/whst_series_analysis.py`, com motivo registrado por série em
`data/whst_series_analysis.csv`:

| Decisão da série | Entra na correção? | Justificativa |
|---|---|---|
| `CORRIGIR_misto` | **Sim** | Viés não cancela: alguns frames super, outros OK |
| `CORRIGIR_consistente_implausivel` | **Sim** | Padrão consistente, mas closure implausível ⇒ *k* varia no tempo |
| `USAVEL_sem_correcao` | Não | Viés multiplicativo cancela na razão; closure plausível |
| `SEM_BASELINE_fora_pareada` | Não | Sem t₀ não há closure; re-segmentar não cria o baseline |
| `REVISAR_contem_invalida` | Revisão à parte | Decisão humana sobre o frame inválido |

Baselines (t₀) com segmentação ruim entram **sempre** e vêm primeiro em cada
série (a worklist os marca com `eh_baseline = SIM`): t₀ errado invalida a série
inteira.

**Frames `IMG_INVALIDA` nunca entram na lista de correção** — não há ferida a
re-segmentar.

### 3.1 Regra para frames inválidos (declarada a priori, aplicada uniformemente)

Mesma lógica já adotada em todo o trabalho: **falta de baseline mata a série;
falta de um timepoint intermediário, não.**

| Situação | Decisão | Consequência |
|---|---|---|
| **R1** — frame inválido em **t > 0** | Descarta **apenas aquele timepoint** | A série continua com os demais; seus frames `SEG_RUIM` **voltam para a lista de correção** |
| **R2** — frame inválido em **t₀**, com campo-irmão válido no t₀ | Usa o **irmão como baseline** | A série continua normalmente |
| **R3** — frame inválido em **t₀**, sem campo-irmão válido | Série **sem baseline** | Sai da análise pareada (mesma categoria de `SEM_BASELINE_fora_pareada`); seus frames **não são corrigidos** |

Esta regra substitui a decisão caso a caso: é uniforme, registrada antes da
aplicação e reproduzível a partir do código
(`stage4/whst_series_analysis.py`, bloco `REGRA_INVALIDOS`).

**Limitação declarada da R2.** Os dois campos de um poço são posições distintas
ao longo do mesmo risco, portanto têm larguras iniciais de ferida próximas mas
não idênticas. Medido nos poços em que ambos os campos foram classificados `OK`
no t₀ (n = 6 poços HUVEC), a diferença relativa entre os campos é de **8,6%
(mediana; mín 1,3%, máx 26,9%)**. O baseline emprestado carrega, portanto, um
erro dessa ordem, que se propaga para a closure fraction daquela série. A regra
é aplicada mesmo assim por ser preferível a descartar a série inteira, e as
séries afetadas são identificadas em `data/whst_series_analysis.csv`
(`baseline_emprestado = sim`) para que a análise possa ser refeita sem elas
como verificação de robustez.

Frames inválidos permanecem listados em `stage4/revision_worklist.csv` para registro,
mas a decisão sobre eles passa a ser automática pela regra acima — não há mais
julgamento caso a caso.

**Fundamento do critério `USAVEL`:** closure fraction é uma razão, então um viés
multiplicativo constante dentro da série cancela —
`(k·a₀ − k·aₜ)/(k·a₀) = (a₀ − aₜ)/a₀`. O teste empírico mostrou que isso vale em
apenas 4 das 21 séries consistentemente super-segmentadas: nas outras, a closure
é implausível, o que demonstra que *k* **não** é constante no tempo (a
super-segmentação piora quando a ferida encolhe, porque o excesso passa a
dominar a área medida). Essa é uma limitação mecanicamente explicada do WHST, e
está reportada como tal.

---

## 4. Execução da correção (Fiji)

`Plugins > Macros > Run...` → `stage4/whst_manual_correction.ijm`

- Selecionar a raiz do projeto e a passada.
- As imagens abrem **ordenadas por série** (linha celular → unidade →
  timepoint → campo), preservando o contexto temporal.
- O **ROI automático é pré-carregado** no ROI Manager como ponto de partida.
- Ajustar e clicar OK. `Select None` + OK registra `pulada`.
- Nada do original é sobrescrito: as saídas vão para pastas novas.
- Retomável: imagens já salvas são puladas automaticamente.

| Passada | Lista | Saída | Registro |
|---|---|---|---|
| 1 — correção | `stage4/correction_worklist.csv` | `whst_output/rois_corrigidos/` | `stage4/correcao_manual_pass1.csv` |
| 2 — re-correção cega | `stage4/.recorrecao_oculta.csv` | `whst_output/rois_recorrecao/` | `stage4/correcao_manual_pass2.csv` |

---

### 4.1 Desfechos possíveis por imagem

| Desfecho | Como registrar | Significado | Registro |
|---|---|---|---|
| **Contorno corrigido** | traçar/ajustar o ROI, OK | medida válida | `area`, status `ok` |
| **Ferida fechada** | `Ctrl+Shift+A` → OK → "FERIDA FECHADA" | **área = 0**, closure = 1,0 | `area=0`, status `fechada` |
| **Pular** | `Ctrl+Shift+A` → OK → "PULAR" | sem medida; **reaparece** na próxima execução | `NA`, status `pulada` |
| **Inválida** | `Ctrl+Shift+A` → OK → "IMAGEM INVÁLIDA" | frame descartado | `NA`, status `invalida` |

**Ferida fechada não é dado ausente.** É a medida zero — o endpoint mais
informativo da série (fechamento completo). Registrá-la como "pulada" criaria um
buraco onde há informação.

### 4.2 Ferida em regiões separadas

Quando a ferida está partida em dois ou mais trechos (monolayer fez ponte no
meio), traçar o primeiro trecho e, **segurando `Shift`**, traçar os demais: o
ImageJ compõe um ROI único e a área medida é a **soma**.

Isto é uma diferença deliberada em relação ao WHST automático, que aplica
`Analyze Particles` e retém apenas o **maior componente conectado** — portanto
subestima sistematicamente feridas fragmentadas. A correção manual mede a
extensão real. A diferença resultante entra no IoU(automático, corrigido) e é
parte do que o protocolo quantifica.

## 5. Reprodutibilidade intra-observador

15 imagens da lista são sorteadas com **seed fixo (`SEED = 4336348`)** para
correção em duplicata, com intervalo entre as passadas.

**Cegamento:** `stage4/correction_worklist.csv` — o arquivo que o observador usa — não
identifica quais imagens foram sorteadas. A lista fica em
`stage4/.recorrecao_oculta.csv`, que só é lido pelo macro na passada 2 e **não deve ser
aberto** antes da conclusão de ambas as passadas. O observador, portanto, não
sabe durante a passada 1 quais imagens repetirá.

Métricas (`stage4/correction_agreement.py`):

- **IoU(pass1, pass2)** — concordância espacial das máscaras.
- **CCC de Lin** entre as áreas — concordância dos valores, sensível a viés
  sistemático (ao contrário de Pearson).

```bash
python stage4/correction_agreement.py
```

Frase resultante para o Methods (valores preenchidos após a execução):

> A correção manual apresentou reprodutibilidade intra-observador de IoU ___
> (mediana) e CCC de Lin ___ (n = 15 imagens re-corrigidas às cegas, com
> intervalo entre as passadas).

---

## 6. Magnitude da intervenção manual

O mesmo script calcula **IoU(automático, corrigido)** para todas as imagens
corrigidas, quantificando o quanto a correção alterou o resultado do WHST — e o
delta de área em pontos percentuais. Isso transforma "houve correção manual" em
uma medida reportável, e evidencia a direção do erro (espera-se redução de área
na maioria, dado que o WHST super-segmenta).

---

## 7. Ordem de execução (resumo)

```bash
python stage1/adjudicate_ambiguo.py --template     # 1. gera template
#    (preencher a coluna 'decisao')
python stage1/adjudicate_ambiguo.py --apply        # 2. aplica
python stage4/whst_series_analysis.py              # 3. fecha vereditos de série
python stage4/build_correction_worklist.py         # 4. lista final + sorteio cego
#    Fiji: stage4/whst_manual_correction.ijm  -> passada 1
#    (intervalo)
#    Fiji: stage4/whst_manual_correction.ijm  -> passada 2
python stage4/correction_agreement.py              # 5. IoU + CCC
```

---

## 8. Papéis das avaliações: o que é padrão-ouro para quê

Há **duas** avaliações humanas neste trabalho, com validade distinta. Elas não
se substituem e ambas são reportadas.

| Avaliação | Arquivo | Válida para | Inválida para |
|---|---|---|---|
| **Triagem visual cega** (rastreio) | `data/inspecao_visual_TRIAGEM_CEGA.csv` (congelado, somente leitura) | Sensibilidade/especificidade do QC automático | Ser substituída por reavaliação informada nessa comparação |
| **Correção manual** (referência) | `whst_output/rois_corrigidos/` | **Reference standard** da análise pareada; área da ferida | — |

**Justificativa.** "Padrão-ouro" designa a referência **independente do teste
avaliado**, não a avaliação mais acurada em termos absolutos. Nas duas
condições em que essas propriedades coincidem, a avaliação mais informada é
preferível — e é por isso que a **correção manual em resolução plena, e não a
triagem, é o reference standard da análise pareada**. A exceção é estreita: a
comparação com o QC automático exige uma referência que não tenha sido
informada pela saída desse mesmo QC, sob pena de viés de incorporação
(*incorporation bias*, STARD). Como o operador já conhecia os resultados do QC
ao adjudicar os casos ambíguos e ao rever as imagens, apenas o estado
**anterior** a esse conhecimento é válido para aquela comparação — daí o
congelamento.

O arquivo congelado corresponde ao estado pré-adjudicação
(OK 65 / super 118 / sub 14 / inválida 14 / ambíguo 12; MD5 registrado pelo
script `stage4/freeze_blind_triage.py`). A adjudicação dos 12 ambíguos e a
classificação de modo de falha são posteriores e **não** entram nele.

## 9. Anotação por imagem durante a correção

O macro coleta, a cada imagem, uma **tag estruturada** e um **comentário
livre** (ambos opcionais):

`ferida_ja_fechada` · `mascara_deslocada` · `debris_ou_artefato` ·
`fora_de_foco` · `borda_ambigua` · `sem_ferida_no_campo` ·
`monolayer_nao_confluente` · `outro`

Registrados em `stage4/correcao_manual_pass1.csv` / `pass2.csv`. Como ficam ancorados
no ROI corrigido, servem de **validação independente** do campo `modo_falha`
calculado geometricamente por contenção (`stage4/classify_failure_mode.py`):
concordância entre os dois é evidência convergente; discordância localiza erro
da métrica.

A passada 2 não exibe as anotações da passada 1 — a re-correção precisa medir
julgamento independente, não recordação.

## 10. Resultados finais (para o Methods)

Números apurados após a correção manual (5 passadas), a validação de
procedência e as verificações de consistência. Denominador do pipeline WHST:
**223 imagens** — 223 originais − 5 imagens-teste removidas (§10.5)
+ 5 baselines recuperados do banco cru (§10.8).

### 10.1 Classificação de validade

| classe | n | % de 218 |
|---|---:|---:|
| falha do método WHST (imagem válida, segmentação ruim) | 123 | 55,2% |
| válida com segmentação correta | 77 | 34,5% |
| **invalidez de imagem — taxa de exclusão do ensaio** | **23** | **10,3%** |

A taxa de exclusão (10,3%) e a taxa de falha do método (55,2%) são grandezas
distintas e **não devem ser somadas**: a primeira descreve as imagens; a segunda,
o desempenho do WHST sobre imagens válidas.

### 10.2 Magnitude da intervenção manual

| métrica | valor |
|---|---|
| IoU(automático, corrigido) | **0,267** (mediana; IQR 0,101–0,434) |
| variação de área | **−22,5 pp** (mediana) |
| correções que reduziram a área | **86%** |

### 10.3 Reprodutibilidade intra-observador

> A correção manual apresentou reprodutibilidade intra-observador de
> **IoU 0,861** (mediana) e **CCC de Lin 0,996**, com viés médio de +0,10 pp
> (corrigido em 2026-07-31 — ver D26; os valores anteriores incluíam três pares em que o observador
> não registrou ferida em nenhuma das passadas, e um par de IoU exato)
> (n = 14 imagens re-corrigidas às cegas, com intervalo entre as passadas);
> a concordância de desfecho (corrigida / fechada / inválida) foi de **14/15 (93%)**.

O contraste com o IoU automático×manual de 0,262 é o argumento central: o mesmo
observador concorda consigo mesmo **3,4×** mais do que o método automático
concorda com ele. A variabilidade está no método, não no observador.

### 10.4 Séries analisáveis

| | n |
|---|---:|
| séries totais | 68 |
| **analisáveis (baseline + closure plausível)** | **52** |
| sem baseline | 9 |
| closure implausível após correção | 7 |

Closure no último timepoint: **mediana 0,691** (IQR 0,450–0,930; faixa
0,008–1,000); **13 séries** com fechamento completo (≥ 0,99).
Total: **187 medições** em formato longo (`data/closure_final_longo.csv`).

Procedência da área nas 52 analisáveis: **35 integralmente corrigidas**,
**13 integralmente automáticas** (viés cancela) e **4 mistas** — estas últimas
são séries que ganharam baseline recuperado (§10.8) e cujos demais frames já
haviam sido julgados corretos.

### 10.7 Validação e eliminação da procedência mista

A análise inicial produziu 24 séries que misturavam timepoints corrigidos e
automáticos. Como a closure é uma razão, uma diferença **sistemática** entre as
duas fontes enviesaria essas séries. A premissa implícita ("frame julgado OK na
triagem ≈ tão acurado quanto corrigido") foi testada, não assumida.

**Teste (n = 10, sorteio com seed fixo, corrigidos às cegas):**

| métrica | valor |
|---|---|
| IoU(automático, corrigido) em frames julgados **OK** | **0,840** |
| IoU(automático, corrigido) em frames julgados **ruins** (passada 1) | 0,262 |
| delta relativo com sinal | −1,1% (mediana) |
| teste dos sinais / Wilcoxon | p = 0,754 / p = 0,922 |
| deslocamento resultante na closure | 0,079 (mediana) |

O contraste 0,840 × 0,262 mostra que a triagem visual **separou corretamente**
frames bem e mal segmentados. Não se detectou viés direcional: o desvio de 10%
em módulo é **dispersão**, não deslocamento sistemático.

**Decisão.** Como a dispersão residual equivalia a ~28% do desvio entre séries —
custo de precisão, ainda que não de acurácia — optou-se por **corrigir todos os
frames automáticos das séries mistas** (24 frames adicionais) em vez de
restringir a análise às homogêneas (que custaria metade das séries). Após essa
passada, **nenhuma série analisável é mista**: a questão passou a ser resolvida
por construção, e não por inferência a partir de uma amostra de 10.

**Esforço total de correção manual: 140 frames** (101 na passada 1 + 10 na
validação de procedência + 24 na completação + 5 baselines recuperados),
correspondendo a **132/223 (59%)** das imagens do pipeline com medida corrigida
manualmente. Nenhuma imagem ficou pendente.

## 10.8 Baselines recuperados do banco cru

Nove das séries sem baseline tiveram o t0 procurado no banco cru
(`Banco de dados/HUVEC-RAW`), exigindo correspondência exata de **lote,
tratamento, poço e campo** — casar apenas pelo poço é incorreto, pois o mesmo
poço existe em vários lotes e um t0 de outro lote é outro experimento.

Cinco candidatos foram localizados, com MD5 inédito, e passaram pelo **mesmo
pipeline** (medição automática por `stage4/whst_batch.ijm` + correção manual com o
mesmo critério de borda). Entraram marcados como `BASELINE_REC` e
`origem = baseline_recuperado_posteriori`, e ficaram como `NAO_TRIADA` até a
correção — não passaram pela triagem visual cega, por terem sido localizados
depois, e portanto não entram em nenhuma estatística de triagem.

| série | área automática | veredito manual |
|---|---:|---|
| LUIS RAW ‖ PET ‖ C5 | 31,4% | inválida |
| LUIS RAW ‖ PET ‖ F2 | 13,1% | inválida |
| Migração n2 ‖ PEP ‖ A2 | 68,4% | inválida |
| originais ‖ A5 | 37,9% | **18,4%** (−51%) |
| originais ‖ B5 | 75,2% | **23,1%** (−69%) |

Dois se confirmaram, elevando as séries analisáveis de 49 para 52. Nos dois
casos o WHST super-segmentava fortemente; os valores corrigidos (18,4% e 23,1%)
são compatíveis com a mediana de t0 do lote (17,3%), enquanto os automáticos
(37,9% e 75,2%) não eram. As três restantes permanecem sem baseline.

### 10.5 Remoção de imagens fora do escopo

Cinco imagens ("em cruz") foram identificadas pelo operador como **teste do
algoritmo, não dado experimental**, e removidas do pipeline. Verificou-se que
**não constavam de `data/mapping_dataset_final_strat.csv`** — nunca integraram as
partições train/val/test — portanto a remoção não afeta o split leakage-free nem
exige re-treino. Ficam em quarentena com manifesto
(`whst_output/_removidas_fora_do_escopo/`).

### 10.6 Regra de consistência entre rótulo e medida

Um frame classificado como inválido **não gera área final**, e um frame que o
operador conseguiu segmentar é **promovido a válido** (traçar o contorno é a
prova de que a imagem é utilizável). Sem essa regra, a análise por série e a
tabela de closure divergiriam sobre quais frames compõem cada série.

## 11. Notas de integridade

- Nenhum script sobrescreve medições ou ROIs originais.
- `data/inspecao_visual.csv` recebe backup (`.pre_adjudicacao.bak`) antes da adjudicação.
- O sorteio é determinístico: mesmo seed ⇒ mesmo subconjunto, reproduzível por
  terceiros a partir do código.
- A taxa de exclusão do ensaio (**invalidez de imagem**) é reportada
  separadamente da falha de método do WHST sobre imagens válidas — são
  quantidades distintas e não devem ser somadas.
