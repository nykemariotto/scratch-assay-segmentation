# Protocolo Experimental Pré-Registrado
## Reconstrução completa — Manuscript 4336348, Cytometry Part A (Major Revision)

> **Natureza deste documento.** Especifica, ANTES de ver qualquer resultado, exatamente como cada etapa da reconstrução será executada. Serve três funções: (1) instruções executáveis para o Claude Code; (2) defesa metodológica citável no response letter ("we followed a pre-specified protocol"); (3) garantia de rigor contra p-hacking e cherry-picking. Congelar antes de rodar. Desvios devem ser documentados e justificados.
>
> **Ambiente de execução.** Trilha Compute roda localmente (Windows PowerShell, conda, GPU) via Claude Code. Trilha Chat projeta e analisa. Convenção: caminhos, seeds e comandos são explícitos e determinísticos.
>
> **Princípio-mestre.** Cada número no manuscrito revisado deve ser rastreável a um script versionado com seed fixo. Nada de black-box.

---

## Convenções globais

| Item | Valor |
| --- | --- |
| Semente global (Python/NumPy/PyTorch/CUDA) | `SEED = 42` (e réplicas 43, 44, 45, 46 para multi-seed) |
| Determinismo | `torch.backends.cudnn.deterministic = True`; `torch.use_deterministic_algorithms(True)` quando viável |
| Framework de treino | Ultralytics YOLO (versão fixada em `requirements.txt`) |
| Formato de imagem | conforme dataset (JPG 640×640 depositado; TIFF nativo se disponível para reprocessamento) |
| Versionamento | cada etapa gera artefatos com hash; splits e métricas commitados no GitHub |
| Registro de ambiente | `pip freeze` / `conda env export` salvos junto de cada run |

---

# ETAPA 1 — Reconstrução do dataset (split leakage-free)

**Resolve:** A1 (leakage), C2 (augmentation counts), C9 parcial (determinismo).
**Trilha:** Code projeta+executa; Chat valida o design.

## 1.1 Extração de metadata

Cada imagem tem well/timepoint/tratamento/linha-celular recuperável do nome do arquivo (ex.: `A2_24hr.tiff`). O Code deve:

1. Listar todos os arquivos de imagem + anotação.
2. Parsear cada nome com um regex documentado que extraia, no mínimo:
   - `cell_line` (HUVEC / SKOV-3)
   - `well_id` (identificador do poço — ex.: A2, B3)
   - `timepoint_h` (0, 8, 12, 24 para HUVEC; 0, 24, 48 para SKOV-3)
   - `treatment` / `clinical_group` (EOPE / LOPE + tratamento farmacológico) — extrair se recuperável, mas NÃO usado como variável de análise nem de estratificação (decisão D1). Registrar apenas para proveniência.
   - `replicate` (se houver réplicas independentes)
3. Produzir `dataset_metadata.csv` com uma linha por imagem e todas as colunas acima + caminho do arquivo + presença/ausência de anotação.
4. **Verificação de sanidade:** contar imagens por (cell_line × timepoint), por well, por treatment. Reportar totais e cruzar com o esperado. Sinalizar qualquer arquivo cujo nome não parseie.

> **[Chat valida]** o regex e o dicionário de mapeamento antes do split. Nomes ambíguos são resolvidos manualmente, não por heurística silenciosa.

## 1.2 Definição da unidade de agrupamento

**Unidade de agrupamento = well.** Justificativa: todas as imagens de um mesmo well ao longo dos timepoints são do mesmo campo físico → correlacionadas → não podem cruzar partições. Onde um well recebe um único tratamento, agrupar por well já agrupa por tratamento. Se um well tiver múltiplos tratamentos (improvável em placa de scratch), agrupar por (well × treatment).

**Regra dura:** nenhum well aparece em mais de uma partição (train/val/test).

## 1.3 Estratégia de split

- **Proporção:** 70% train / 15% val / 15% test (ajustável; documentar a escolha final).
- **Método:** split estratificado por `cell_line × tratamento`, agrupado por `well_id` (via super-chave com fallback). Usar `StratifiedGroupKFold` (ou `GroupShuffleSplit` com estratificação manual) com `random_state = SEED`.
- **Estratificação (cobertura, não análise):** estratificar por linha celular E tratamento garante que o test set cobre toda a diversidade visual (evita estratos ausentes — 4 braços NEB ficariam de fora se estratificasse só por linha). **Importante (coerência com D1):** estratificar ≠ analisar. A estratificação é higiene do split (preservar distribuição marginal para avaliação representativa); NÃO se reporta métricas por tratamento (D1 removeu a análise clínica). Estratifica-se pela variável sem reportar por ela.
- **Determinismo:** o split é 100% reproduzível a partir de `SEED`. Salvar `split_assignment.csv` (well_id → partição) e `split_images.csv` (image → partição).

## 1.4 Verificação de zero-overlap (crítico para R1.1)

O Code deve produzir, e o manuscrito deve incluir no Supplementary:

1. **Tabela de contagem por partição:** nº de wells, nº de imagens, distribuição por cell_line/timepoint/treatment em cada partição.
2. **Prova de disjunção:** interseção dos conjuntos de `well_id` entre train/val/test deve ser vazia. Asserção no código: `assert train_wells ∩ val_wells == ∅` etc. Falha o build se violado.
3. **Statement reproduzível:** "No well contributed images to more than one partition (verified programmatically; see `verify_split.py`)."

## 1.5 Augmentation (resolve C2)

- **Aplicar augmentation SOMENTE ao conjunto de treino**, DEPOIS do split (nunca antes — evita leakage de augmentation).
- Documentar exatamente: tipos de transformação, nº de outputs por imagem, e o count final resultante.
- Reportar a fórmula: `n_augmented = n_train_images × outputs_per_image` (menos quaisquer descartes por falta de anotação), com o número exato impresso pelo pipeline.
- Reconciliar com o texto do manuscrito: o número reportado deve ser o número que o script imprime, verbatim.

## 1.6 Entregáveis da Etapa 1

- `dataset_metadata.csv`, `split_assignment.csv`, `split_images.csv`
- `verify_split.py` (asserções de zero-overlap) + output
- Tabela de contagens por partição (para Supplementary Table)
- Relatório de augmentation counts

**✋ Gate 1:** não prosseguir para treino até o zero-overlap estar verificado e o Chat validar as contagens.

---

# ETAPA 2 — Re-treino local controlado

**Resolve:** A1 (métricas leakage-free), A2 (ablation limpa), A3 (multi-seed), C9 (seed explícito).
**Trilha:** Chat projeta o grid; Code executa.

## 2.1 Modelos

**Núcleo (obrigatório):** família YOLO11-seg — variantes a definir por orçamento de GPU. Mínimo defensável: `yolo11s-seg`, `yolo11m-seg`, `yolo11x-seg` (pequeno/médio/grande, cobre o trade-off tamanho×acurácia).

**Contraste de paradigma (recomendado):** uma arquitetura transformer de segmentação (RT-DETR-seg ou equivalente disponível em Ultralytics/local) — honra o contraste CNN vs transformer que o R1.12 apontou como ausente no deployment.

## 2.2 Ablation single-variable (resolve R1.2)

A regra que o revisor exige: **cada comparação de ablation varia UM fator, com todo o resto fixo.**

Fatores a isolar (cada um num par controlado):

| Fator | Nível A | Nível B | Fixo em ambos |
| --- | --- | --- | --- |
| **Edge padding** | black-edge | white-edge | mesma arquitetura, mesmo init, mesmo dado, mesmos hiperparâmetros, mesmo seed |
| **Init checkpoint** | COCO-pretrained | from-scratch (ou outro) | mesma arquitetura, mesmo padding, mesmo dado, mesmo seed |
| **Model size** | s / m / x | (comparação de escala) | mesmo padding, mesmo init, mesmo dado, mesmo seed |

> **Ponto crítico:** o design antigo entrelaçava padding com init e arquitetura. O novo design cria pares onde **apenas** o fator de interesse muda. Se, sob controle, o efeito do padding desaparecer, reportamos isso honestamente (R1.2 aceita "remover/qualificar a conclusão").

## 2.3 Protocolo de múltiplos seeds (resolve R1.3)

- Cada configuração de interesse é treinada com **N seeds independentes** (mínimo N=3; ideal N=5): 42, 43, 44 [, 45, 46].
- Reportar **mean ± SD** de cada métrica entre seeds.
- Isso quantifica a variabilidade de treino que o revisor apontou como ausente.

## 2.4 Hiperparâmetros

- Fixar e documentar: epochs, batch size, image size (640), learning rate schedule, optimizer, early-stopping criterion.
- Idênticos entre configurações comparadas (senão a comparação não é single-variable).
- Salvar o `args.yaml` de cada run.

## 2.5 Entregáveis da Etapa 2

- Pesos treinados de cada configuração × seed
- `args.yaml` + ambiente por run
- Logs de treino (curvas de loss/mAP)
- Tabela bruta de métricas por (config × seed)

**✋ Gate 2:** validar que os pares de ablation diferem em exatamente um fator antes de interpretar resultados.

---

# ETAPA 3 — Avaliação com incerteza

**Resolve:** A3 (CI), A5 (por linha celular), C4 (claims precisos), C6 (mAP vs threshold).
**Trilha:** Chat especifica estatística; Code calcula.

## 3.1 Métricas no test set leakage-free

- mAP@50, mAP@75 (computadas sobre a curva P–R completa — **independentes do threshold**, corrige R1.13/C6).
- Precision, Recall, F1 (reportadas no operating threshold, documentado).
- **Esclarecimento explícito** na legenda: mAP é threshold-independent; P/R/F1 dependem do threshold de operação.

## 3.2 Quantificação de incerteza (resolve R1.3)

- **Entre seeds:** mean ± SD de cada métrica.
- **Bootstrap AGRUPADO (cluster bootstrap):** IC 95% reamostrando por **grupo** (well/campo), NÃO por imagem (N resamples ≥ 1000, seed fixo). **Crítico (achado D4):** o test set tem 166 imagens mas só ~28 unidades independentes (poços); bootstrap ingênuo por imagem subestima a largura do IC em ~2,4×. Reamostrar por grupo corrige esse viés e reflete a incerteza real. Documentar explicitamente — é rigor que responde R1.3 corretamente.
- **Teste de distinguibilidade:** para claims de "config A > config B", reportar se os ICs se sobrepõem. Onde se sobrepõem, declarar estatisticamente indistinguível (corrige R1.3).

## 3.3 Desagregação por linha celular (resolve A5)

- Reportar todas as métricas de segmentação **separadamente para HUVEC e SKOV-3**.
- Substitui a Table 2 pooled por uma tabela desagregada.
- Isso permite ver se SKOV-3 (monolayer mais denso) tem performance distinta.

## 3.4 Entregáveis da Etapa 3

- Table 2 revisada: por config × linha celular, com mean ± SD e IC bootstrap
- Statement de distinguibilidade estatística
- `evaluate.py` reproduzível

**✋ Gate 3:** decisão condicional A5 (SKOV-3) informada pelas métricas desta etapa → alimenta Etapa 4.

---

# ETAPA 4 — Reference standard (WHST) refeito

**Resolve:** R2.2 (WHST citação + angle correction), A5 (reference para SKOV-3?).
**Trilha:** VOCÊ opera o ImageJ; Chat especifica o protocolo; Code processa os resultados.

## 4.1 Ferramenta e citação

- **Wound Healing Size Tool** (Suarez-Arnedo A, et al. PLoS ONE. 2020;15(7):e0232565) — o método manual/semi-automático dominante, mantém relevância para a comunidade.
- Citação já verificada e pronta.

## 4.2 Protocolo de medição (parâmetros a CONGELAR)

Você vai remedir você mesmo, então documentamos TUDO (resolve a lacuna que o R2.2 expôs):

1. **Parâmetro extraído:** wound area → closure fraction (não average width). Definir closure fraction = (área_t0 − área_tX) / área_t0.
2. **Angle correction:** decidir e DOCUMENTAR se a correção de inclinação do WHST fica ativa (default) ou desativada. Recomendação: manter o default do plugin (correção ativa) e reportar isso explicitamente, já que é como o tool é usado na prática. **Reportar a escolha, seja qual for.**
3. **Parâmetros do plugin:** variance filter radius, binarization threshold, contrast enhancement — registrar os valores usados.
4. **Supervisão humana:** cada segmentação conferida/corrigida pelo operador via Manual Tool quando o automático falhar. Documentar critério de correção.
5. **Cegamento:** idealmente, o operador do WHST não vê a saída do AI ao medir (evita viés). Documentar se foi cego.

## 4.3 Amostra de medição

- Definir sobre quais imagens o reference standard é aplicado. Mínimo: as imagens pareadas necessárias para o method-agreement (t0 + tX por well, HUVEC).
- **Decisão SKOV-3 (condicional, pós-Etapa 3):** se as métricas de segmentação de SKOV-3 justificarem, estender o WHST a SKOV-3; senão, restringir agreement a HUVEC e declarar escopo.

## 4.4 Entregáveis da Etapa 4

- `whst_measurements.csv` (well, timepoint, área, closure fraction, params usados)
- Documento de protocolo WHST (params congelados) para o Methods
- Statement sobre angle correction para R2.2

**✋ Gate 4:** protocolo WHST congelado e documentado ANTES de medir.

---

# ETAPA 5 — Benchmark vs ferramentas automáticas

**Resolve:** A4 (avanço sobre estado da arte).
**Trilha:** Chat seleciona; Code roda.

## 5.1 Comparadores (mínimo: 1 clássico + 1 DL)

**Clássico automático (escolher 1):**
- Wound Healing Size Tool **automatic mode** (Suarez-Arnedo 2020) — mesma família do reference, mas modo automático.
- TScratch (Gebäck et al., 2009).
- MRI Wound Healing Tool (ImageJ macro).

**Deep-learning (escolher 1):**
- DeepScratch (Javer A, Rittscher J, Sailem HZ. CSBJ 2020;18:2501–9) — referência verificada.

**Transformer (paradigma, decisão D8):**
- RF-DETR-seg local (pacote `rfdetr`, Apache 2.0, backbone DINOv2 + cabeça de segmentação MaskDINO). Reposicionado da ablation para o benchmark (D8): comparar arquiteturas diferentes é apropriado aqui, não na Etapa 2. Treinar local com seed explícito no mesmo dataset leakage-free. **Risco:** pacote em preview instável (v1.3.0 yanked) + backbone pesado para 8 GB — se falhar tecnicamente, declarar a tentativa e sustentar a Etapa 5 com os demais comparadores.
- U-Net workflow de Dogru et al. (2024) — já citado no manuscrito, não comparado.

> **Recomendação:** TScratch (clássico, amplamente usado, fácil de rodar) + DeepScratch (DL, código disponível). Confirmar disponibilidade de código/execução antes de fixar.

## 5.2 Critério de fairness

- **Mesmo test set leakage-free** para todos os comparadores.
- **Mesma definição de closure fraction / wound area.**
- Documentar qualquer ajuste necessário para rodar cada tool (parâmetros default vs tunados; se tunar, documentar).
- Comparadores que tratam células migrando para dentro do gap são especialmente informativos (conecta com R1.7/B2).

## 5.3 Análise head-to-head

- Métrica de comparação: concordância com o reference standard (WHST supervisionado) OU métricas de segmentação diretas (IoU/Dice) contra ground-truth, conforme aplicável a cada tool.
- Reportar onde o método proposto ganha/perde vs cada comparador.

## 5.4 Entregáveis da Etapa 5

- `benchmark_results.csv`
- Tabela/figura head-to-head
- Notas de execução de cada comparador (params, versão)

---

# ETAPA 6 — Análise, manuscrito, response letter

**Resolve:** B1, B2, B3, B4, C5, C7, R2.1 + reconciliação de todos os placeholders.
**Trilha:** Chat (aqui).

## 6.1 Method-agreement refeito (resolve B1)

- Recalcular Pearson, Lin's CCC, Bland–Altman, TOST com os dados novos (AI re-treinado vs WHST refeito).
- **Foreground os LoA** na narrativa; equivalência TOST enquadrada como equivalência da média populacional, não interchangeability individual.
- Reclassificar CCC conforme o valor obtido (se < 0.90, "moderate-to-substantial", não "high").
- **D1:** remover o bloco PER-CLINICAL-GROUP BREAKDOWN do `paired_analysis.py` e dos Results. Recalcular agreement pooled (e por linha celular, resolve A5) sem estratificação clínica.

## 6.2 Edições de texto (Bloco B/C restante)

- **D1:** substituir toda a análise por grupo clínico (EOPE/LOPE) pela frase de proveniência única (Variante B, ver master map) no Methods. Limpar menções clínicas em Abstract/Discussion/Conclusions. Estratificação e análise passam a ser por linha celular apenas.
- B2: parágrafo sobre células isoladas migrando no gap (+ análise opcional). Nota: o WHST também faz hole-filling (inclui células isoladas) → AI e reference têm o mesmo comportamento → comparação justa (ver protocolo WHST §3).
- B3: limitação de generalizabilidade (condições de imagem).
- B4: tabela de outputs do software (já auditada — pronta).
- C5: corrigir "paradigmas" dos modelos deployed + justificar escolha de deployment.
- C7: qualificar casos discordantes (sistemático vs ruído). Nota: casos como A2 24h e SKOV-3 tardios documentados no protocolo WHST.
- C1: corrigir exclusion rate invertido (verificar contra dados novos).
- C4, C11: reconciliar todos os claims de mAP com os valores novos.
- R2.1 + C10: passe editorial completo (consistência + concisão) via skill `humanizer`.

## 6.3 Atualização de repositórios

- Zenodo: novo depósito com dataset re-splittado, pesos novos, RF-DETR/transformer weights (resolve C9), scripts.
- GitHub: README overhaul (B5), CI atualizado, exemplo, quickstart.
- HF Space: modelos atualizados se aplicável.

## 6.4 Response letter

- Preencher todos os `[A PREENCHER]` do esqueleto com os resultados reais.
- Reconciliar valores em Abstract/Results/Discussion/Tables.

## 6.5 Auto-peer-review (blindagem final)

- Rodar a skill `peer-review` no manuscrito revisado, simulando R1 e R2.
- Resolver o que aparecer ANTES de reenviar.

---

# Mapa de rastreabilidade: etapa → pontos dos revisores

| Ponto | Etapa(s) | Status |
| --- | --- | --- |
| A1 leakage | 1, 2, 3 | ☐ |
| A2 ablation | 2 | ☐ |
| A3 estatística | 2, 3 | ☐ |
| A4 benchmark | 5 | ☐ |
| A5 por linha celular | 3, 4 | ☐ |
| B1 interchangeability | 6.1 | ☐ |
| B2 células isoladas | 6.2 | ☐ |
| B3 generalizabilidade | 6.2 | ☐ |
| B4 outputs software | 6.2 | ✅ auditado |
| B5 GitHub | 6.3 | ☐ |
| B6/R2.2 WHST | 4 | ✅ citação; ⏳ params |
| C1 exclusion rate | 6.2 | ☐ |
| C2 augmentation | 1.5 | ☐ |
| C3/R1.10 DeepScratch | 6.2 | ✅ verificado |
| C4 mAP claims | 3.1, 6.2 | ☐ |
| C5 paradigmas | 6.2 | ☐ |
| C6 mAP threshold | 3.1 | ☐ |
| C7 discordantes | 6.2 | ☐ |
| C8 pre-eclampsia | — | ✅ feito |
| C9 seed/weights | 2, 6.3 | ☐ |
| C10 editorial | 6.2 | ☐ |

---

# Gates de decisão (pontos de pausa obrigatórios)

| Gate | Após | Critério para prosseguir |
| --- | --- | --- |
| Gate 1 | Etapa 1 | Zero-overlap verificado; contagens validadas pelo Chat |
| Gate 2 | Etapa 2 | Pares de ablation diferem em exatamente um fator |
| Gate 3 | Etapa 3 | Decisão SKOV-3 tomada com base nas métricas |
| Gate 4 | Etapa 4 | Protocolo WHST congelado e documentado antes de medir |

---

# Nota sobre a resubmissão

Não é necessário nenhum email prévio ao editor. A resubmissão pelo portal (até 16 out 2026) requer: (1) o manuscrito revisado; (2) o **author response letter** point-by-point; e possivelmente (3) o manuscrito com as revisões marcadas. A magnitude da reconstrução metodológica (re-split leakage-free, re-treino local reproduzível, re-medição do reference standard, benchmark contra ferramentas automáticas) é comunicada na seção de abertura do próprio response letter, que lista as mudanças maiores — cumprindo o papel de "contar tudo que foi feito" sem correspondência separada com o editor.
