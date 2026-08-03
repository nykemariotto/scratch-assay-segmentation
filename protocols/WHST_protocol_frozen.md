# Protocolo WHST congelado — Reference Standard
## Gate 4 do protocolo experimental · Manuscript 4336348

> **Propósito.** Congelar TODOS os parâmetros de medição do Wound Healing Size Tool (WHST) ANTES de qualquer medição, para que o reference standard seja totalmente documentado e reprodutível. Isso resolve a citação, a correção de ângulo e os parâmetros de forma definitiva e blinda contra perguntas de segunda rodada.
>
> **Regra de ouro.** Uma vez congelado, os parâmetros não mudam durante a campanha de medição. Qualquer desvio é documentado e justificado por escrito.

---

## 1. Ferramenta e citação

| Item | Valor |
| --- | --- |
| Plugin | Wound Healing Size Tool (WHST) |
| Versão | *updated* (com botão de seleção manual de áreas não-identificadas = "Manual Tool") |
| Plataforma | Fiji (ImageJ) 64-bit |
| Citação | Suarez-Arnedo A, Torres Figueroa F, Clavijo C, Arbeláez P, Cruz JC, Muñoz-Camargo C. An ImageJ plugin for the high throughput image analysis of in vitro scratch wound healing assays. PLoS ONE. 2020;15(7):e0232565. doi:10.1371/journal.pone.0232565 |
| Repositório do plugin | github.com/AlejandraArnedo/Wound-healing-size-tool |

## 2. Downloads

| Item | URL |
| --- | --- |
| Fiji | https://fiji.sc/ |
| WHST updated (recomendado) | https://github.com/AlejandraArnedo/Wound-healing-size-tool/blob/6210027c8dec7a346bd0a39da37c8d8facf7270d/Wound_healing_size_tool_updated.zip?raw=true |
| Tutorial em vídeo | https://www.youtube.com/watch?v=OgzyJJi-0Ik |
| Manual (S2 File do paper) | Supplementary da publicação PLoS ONE |

## 3. Como o algoritmo funciona (contexto para documentar)

O WHST usa visão computacional clássica baseada em variância de intensidade de pixels vizinhos:
1. **Enhance Contrast** — aumenta a variância dentro do monolayer celular.
2. **Variance filter** — calcula a variância de intensidade na vizinhança de cada pixel (raio = variance window radius). Monolayer → alta variância; ferida aberta → baixa variância.
3. **Threshold** — binariza a imagem de variância (valor definido pelo usuário).
4. **Hole filling** (morphological reconstruction by erosion) — inclui células/ilhas isoladas dentro da ferida como parte da ferida.
5. **Largest connected component** — seleciona a maior região como a ferida verdadeira (elimina falsos positivos).

> **Nota crítica para B2:** o passo 4 (hole filling) faz o WHST **incluir** células isoladas dentro do gap como parte da ferida — ou seja, o WHST **também** não subtrai células migrando para dentro do gap. Isto é relevante: nosso AI (single-contour) e o reference standard (WHST) têm o mesmo comportamento nesse aspecto, o que torna a comparação justa. Documentar isto na discussão fortalece a resposta sobre células isoladas.

## 4. Os 3 parâmetros ajustáveis (a CONGELAR)

O plugin expõe exatamente três parâmetros. Ranges validados pelos autores (do paper, S5 Fig):

| Parâmetro | Função | Range validado | **HUVEC (congelado)** | **SKOV-3 (congelado)** |
| --- | --- | --- | --- | --- |
| **Variance window radius** | Raio do filtro de variância. Muito baixo → não reconhece o scratch; muito alto → subestima a área | 3 – 25 | **20** | **20** |
| **Threshold** | Valor de binarização. Aumentar → aumenta a área detectada | 50 – 150 | **100** | **100** |
| **% saturated pixels** | Contraste. Aumentar → detecta áreas menores. Deve ser > 0 | 0.001 – 0.4 | **0.001** | **0.001** |
| **Scratch is diagonal** | Ativa correção de ângulo (afeta só width, não área) | Yes/No | **Yes** (não afeta área) | **Yes** (não afeta área) |
| **Escala** | Calibração da imagem | — | **Removida → medição em pixels²** | **Removida → medição em pixels²** |

> **Parâmetros unificados HUVEC + SKOV-3 congelados.** HUVEC validado em 4 imagens do poço A2 (0/8/12/24h); SKOV-3 validado no poço 34 (Snap-34, 0/24/48h) com os **mesmos parâmetros** — série monotônica e plausível (closure 0 → 0.542 → 0.674). Usar um único conjunto de parâmetros para ambas as linhas reduz graus de liberdade e é mais defensável (menos suspeita de tuning).
>
> **Ressalva de confiabilidade:** com os parâmetros fixos, a qualidade da segmentação automática **varia por poço e por timepoint**. Poços "fáceis" (como o 34) segmentam bem até 48h; poços "difíceis" falham já a partir de 24h. A variável crítica é a dificuldade intrínseca da imagem (densidade do monolayer, contraste, quão fechada está a ferida), não o parâmetro. Consequência: espera-se **taxa de correção manual significativa no SKOV-3**, maior que no HUVEC. Reportar a taxa no Methods.


## 5. Angle correction

O WHST tem uma opção de **correção de inclinação**: se o scratch é diagonal, o plugin ajusta a medição de largura pelo ângulo de inclinação (ajustando a ROI a uma elipse, correção trigonométrica).

**Decisão congelada e justificativa:**
- A correção de ângulo afeta a medição de **largura** (width), NÃO a de **área**.
- Como nosso reference standard usa **área da ferida → closure fraction** (não width), a correção de ângulo **não afeta** a métrica que usamos.
- **Decisão:** documentar explicitamente que a métrica extraída foi a área da ferida, e que a correção de ângulo do WHST (que opera sobre width) não impacta a comparação baseada em área. Isto é preciso: a preocupação com a correção de ângulo se aplica a width; nossa comparação é por área.

> **CONFIRMADO:** a métrica extraída é a **área da ferida → closure fraction** (não width). Justificativa: (1) o modelo AI produz área (px²), então a comparação maçã-com-maçã exige área do reference standard; (2) closure fraction é a métrica que o manuscrito e toda a análise de agreement já usam; (3) a angle correction do WHST opera sobre width, portanto não afeta a área — o que fecha a questão da correção de ângulo de forma definitiva.

## 6. Calibração dos parâmetros (antes da campanha)

Seguindo a recomendação dos próprios autores (testar 2–3 imagens antes de rodar tudo):

1. Selecionar **3 imagens aleatórias** de cada linha celular (HUVEC e SKOV-3) em timepoints variados.
2. Testar combinações de parâmetros dentro dos ranges validados.
3. Escolher o conjunto que melhor segmenta a ferida por inspeção visual, **separadamente para cada linha celular** se necessário (HUVEC e SKOV-3 podem exigir parâmetros diferentes por causa da densidade/contraste).
4. **Congelar** os valores escolhidos e registrar num arquivo `whst_params.txt`.
5. A partir daí, **não mudar** durante a campanha.

> **Justificativa de calibração separada:** SKOV-3 é um monolayer mais denso; pode exigir parâmetros distintos de HUVEC. Documentar os dois conjuntos é honesto e defensável. O que NÃO se pode fazer é ajustar parâmetros imagem-a-imagem (isso seria tuning circular).

## 7. Escopo da medição (decisão A5 já tomada: HUVEC + SKOV-3)

| Item | Especificação |
| --- | --- |
| Linhas celulares | **Ambas** — HUVEC e SKOV-3 |
| Imagens a medir | Os pares necessários para method-agreement: baseline (t0) + cada timepoint pós-scratch, por well |
| HUVEC timepoints | t0, 8h, 12h, 24h |
| SKOV-3 timepoints | t0, 24h, 48h, 72h |
| Métrica extraída | Área da ferida (pixels²) → closure fraction |

### 7.1 Convenção de nomes de arquivo (para o parsing da Etapa 1)

| Linha celular | Padrão | Exemplo | Well codificado em |
| --- | --- | --- | --- |
| HUVEC | `{well}_{timepoint}hr.tiff` | `A2_24hr.tiff` | letra+número antes do `_` (ex.: `A2`) |
| SKOV-3 | `Snap-{well}-{timepoint}h.tiff` | `Snap-34-24h.tiff` | número entre `Snap-` e o timepoint (ex.: `34`) |

O well é recuperável dos nomes em ambas as linhas (sem necessidade de planilha externa). O regex de parsing da Etapa 1 deve tratar os dois padrões. Para agrupar por well: HUVEC usa o token alfanumérico (A2, B3…); SKOV-3 usa o número (34, 35…). **Atenção:** garantir que os identificadores de well das duas linhas não colidam no split (prefixar com a linha celular, ex.: `HUVEC_A2`, `SKOV3_34`).

## 8. Cálculo de closure fraction (a partir das áreas do WHST)

Para cada well, em cada timepoint pós-scratch:

```
closure_fraction(t) = (area_t0 − area_t) / area_t0
```

Onde:
- `area_t0` = área da ferida no baseline (t=0) para aquele well
- `area_t` = área da ferida no timepoint t para o mesmo well

Interpretação: closure_fraction = 0 → nenhum fechamento; = 1 → fechamento total; < 0 → ferida "cresceu" (ruído de medição ou retração — os casos negativos discutidos no C7).

> Isto reproduz exatamente a definição usada na análise pareada AI vs ImageJ.
>
> **Ponteiro corrigido, 2026-08-02.** Esta linha apontava o `paired_analysis.py` como consumidor destas closure fractions. Esse script foi **retratado**: treinava e avaliava sem filtro de partição, e a análise que produziu não consta do manuscrito. Quem consome as closure fractions é o `stage3/paired_new.py`. **O protocolo não muda** — a definição de closure fraction acima é a mesma. O que se corrige é para onde o leitor é enviado.

## 9. Protocolo de supervisão humana e correção manual (regra congelada)

### 9.1 Fluxo de decisão por imagem

| Situação | Ação |
| --- | --- |
| Automático segmenta a ferida corretamente (típico de 0h/8h/12h — ferida ampla e bem definida) | **Usar valor automático** |
| Ferida quase fechada / borda difusa (típico de 24h/48h) | **Verificar visualmente**; se o automático incluir monolayer (superestimar) ou omitir ferida, **corrigir manualmente** |
| Std deviation de width anormalmente alto vs a série do mesmo well | **Sinal de alerta** → verificar visualmente |

### 9.2 Critério objetivo de "borda da ferida"

Definição operacional (aplicar consistentemente): a borda da ferida é **onde termina o monolayer confluente** e começa a região aberta/esparsa. Ao corrigir manualmente, traçar o contorno nessa transição, usando timepoints anteriores do mesmo well como referência da geometria da ferida.

### 9.3 Ferramenta de correção manual

- **Plugins → Wound healing size manual tool**.
- Remover a escala primeiro (Analyze → Set Scale → Remove Scale) para medir em pixels².
- Traçar o contorno com Polygon ou Freehand selection, seguindo a ferida real.
- Medir (a Manual Tool ou Analyze → Measure).

### 9.4 Registro obrigatório

- Marcar no CSV `corrected_manually = yes/no` para cada imagem.
- Reportar a **taxa de correção manual** (% de imagens corrigidas) no Methods — número importante para transparência.

### 9.5 Caso documentado: poço A2 (HUVEC) — exemplo de referência

Série temporal do poço A2, ilustra o fluxo de decisão:

| Timepoint | Automático (Area %) | Corrigido? | Area % final | Closure fraction |
| --- | --- | --- | --- | --- |
| 0h | 25.986% | Não | 25.986% | 0 (baseline) |
| 8h | 15.942% | Não | 15.942% | 0.386 |
| 12h | 14.753% | Não | 14.753% | 0.432 |
| 24h | 21.189% (superestimou) | **Sim** | 2.878% | 0.889 |

O automático em 24h superestimou (incluiu monolayer esparso como ferida; Std dev de width saltou para 4.241 vs 0.25–0.67 nas demais — sinal de alerta). A correção manual recuperou a fenda estreita real. A série corrigida é monotônica e biologicamente coerente (fechamento progressivo 0 → 38.6% → 43.2% → 88.9%).

**Lição incorporada:** o WHST automático tende a superestimar em feridas quase fechadas com bordas difusas (24h/48h). Estas exigem verificação e frequente correção manual. Isto conecta diretamente com o C7 (casos discordantes) e é reportado honestamente.

## 10. Cegamento (blindagem contra viés)

- **O operador do WHST NÃO deve ver a saída do modelo AI ao medir.** Isto evita que a medição de referência seja inconscientemente enviesada em direção ao AI.
- Medir as áreas WHST **antes** de comparar com o AI, ou com a saída do AI oculta.
- Documentar que a medição foi cega no manuscrito → fortalece a validade do reference standard.

## 11. Registro e entregáveis

| Arquivo | Conteúdo |
| --- | --- |
| `whst_params.txt` | Parâmetros congelados (radius, threshold, saturated pixels) por linha celular |
| `whst_measurements.csv` | Uma linha por (well × timepoint): cell_line, well_id, timepoint, area_px, closure_fraction, corrigido_manualmente (sim/não) |
| `whst_protocol_notes.md` | Notas de operação: quem mediu, quando, critério de correção manual, confirmação de cegamento |

## 12. Texto para o Methods — versão final com números reais

> Substitui o rascunho anterior (escrito antes da execução). Todos os valores são os obtidos na Etapa 4.

### 12.1 Reference standard measurement (Methods)

Reference measurements of wound area were obtained with the Wound Healing Size Tool (WHST; Suarez-Arnedo et al., PLoS ONE 2020;15(7):e0232565), an ImageJ/Fiji plugin that segments the open wound via a variance-based algorithm followed by hole filling and selection of the largest connected component. Measurements were performed on the raw acquisition files (2452 × 2056 px) rather than on the resized copies used for model training, so that all images within a series shared an identical field of view.

Plugin parameters were calibrated on images from both cell lines and then held fixed for the entire campaign (variance window radius = 20; binarisation threshold = 100; saturated pixels = 0.001; all within the ranges validated by the plugin authors). Image calibration was removed prior to measurement so that all areas were recorded in pixels², matching the units of the model output; unit consistency was verified per image. A single parameter set was sufficient for both HUVEC and SKOV-3, avoiding cell-line-specific tuning. Because our outcome is wound **area** rather than average wound width, the plugin's angle-correction function — which adjusts width for scratch inclination — does not affect the reported values.

Because the plugin does not export segmentation geometry, the measurement macro was extended to save, for every image, the region of interest, a binary mask, the contour polygon, and an overlay for visual review. The macro is deposited with the analysis code.

### 12.2 Visual triage and manual correction

Automated segmentations were reviewed by a single observer under a rubric defined a priori: a frame was scored **OK** when a wound was identifiable, the monolayer was confluent on both margins, and the contour followed the wound; **segmentation failure** when the image was valid but the contour was incorrect (excess, deficit, or displacement); and **invalid image** when no wound was present in the field, the frame was severely out of focus, or the monolayer was not confluent. The wound border was defined as the transition between confluent monolayer and open or sparsely populated area.

Review was performed **blind**: the panels presented only the image, its contour, and the file name, with no automated quality flags, measured areas, or series statistics. Critically, the entire measurement and correction campaign was completed **before any segmentation model was trained**, so the observer could not be biased toward the model output — it did not yet exist.

Of 223 images, 77 (34.5%) were scored OK, 123 (55.2%) as segmentation failure on a valid image, and 23 (10.3%) as invalid images. The last figure constitutes the assay exclusion rate and is distinct from the method failure rate: an invalid image is removed from analysis, whereas a segmentation failure on a valid image is corrected. Among failures, excess segmentation predominated, but in a substantial fraction the mask was displaced and did not overlie the wound at all, indicating localisation error rather than imprecise delineation.

Frames scored as segmentation failures were re-segmented manually using the plugin's manual-selection tool, starting from the automated contour. A total of 140 frames were corrected; 132 of 223 images (59%) carry a manually verified measurement. Manual correction reduced the segmented area in 86% of cases, with a median change of −22.5 percentage points of image area.

### 12.3 Observer reproducibility

To quantify the subjectivity of manual correction, 15 frames were randomly selected (fixed seed) and re-corrected after an interval, with the observer blind to both the selection and the first correction. Intra-observer agreement was **IoU 0.861** (median, 95% CI 0.746–0.904) and **Lin's concordance correlation coefficient 0.996** for measured areas, with a bias of +0.10 percentage points; the categorical outcome agreed in 14 of 15 frames.

> **Correction, 2026-07-31.** The values first reported here (IoU 0.894, CCC 0.998) were computed over all 14 repeat pairs. Four of those pairs do not measure boundary reproducibility: in three the observer recorded no wound on either pass, which scores an intersection over union of 1 by the empty-mask convention, and in one the two passes returned an identical area and an exact intersection over union of 1. **The protocol itself is unchanged** — the frame selection, the fixed seed, the blinding and the interval are as frozen. What is corrected is a result computed after freezing.
>
> **Extended, 2026-08-02.** The ratio in the paragraph below was derived from the superseded IoU and was left at 3.3 when the IoU was corrected. With the clean-pair median, 0.8605 / 0.2665 = 3.23, so the figure is **3.2**, and it now reads that way. Same correction, one step further down the chain. Verifiable from `data/correction_agreement.csv` and `stage3/intraobs_ci.json`, both deposited.

For comparison, agreement between the automated and the manually corrected segmentation was **IoU 0.267**. The observer therefore agreed with themselves approximately 3.2 times more closely than the automated method agreed with the observer, indicating that the dominant source of variability is the segmentation method rather than the human operator.

### 12.4 Closure fraction and analysable series

For each acquisition field, closure fraction at time *t* was computed as (area₀ − areaₜ)/area₀, where area₀ is the baseline (0 h) measurement of the same field. Series lacking a baseline cannot yield a closure fraction and were excluded.

Of 68 candidate series, **52 (41 HUVEC, 11 SKOV-3) were analysable**. Nine were excluded for absence of a usable baseline and seven retained implausible closure trajectories after correction.

> **Correction, 2026-08-01.** The counts in the sentence above are wrong and the sentence contradicts itself. Of 69 candidate series (not 68), nine lacked a usable baseline and one contained no valid frame, leaving **59 analysable (48 HUVEC, 11 SKOV-3)** — not 52. The 7 series with implausible trajectories were *retained*, as the sentence itself states, so they belong inside the analysable count; 52 is 59 minus those 7, subtracted in error. The 187 measurements are the total over all 59 series, which is why that figure was and remains correct. Verifiable from `data/whst_series_analysis.csv` and `data/closure_final_longo.csv`, both deposited. **The protocol itself is unchanged** — the calibration, the border criterion, the blinding and the correction rule are as frozen. What is corrected is a count computed after freezing. Median closure at the final timepoint was 0.691 (IQR 0.450–0.930); 13 series reached complete closure (≥ 0.99). The final dataset comprises 187 paired measurements.

Correcting the segmentations markedly improved the biological plausibility of the series: the number of series with implausible closure trajectories fell from 15 to 4, and the category of series that were uniformly over-segmented yet internally consistent disappeared entirely. This indicates that the implausibility arose from segmentation rather than from the underlying biology. It also shows that over-segmentation is not a constant multiplicative bias — which would cancel in a ratio — but grows as the wound narrows, so that the error is largest at precisely the timepoints where closure is most informative.

### 12.5 Statement of independence

The reference standard was established independently of the models under evaluation: measurements were made on raw images with a published, previously validated tool, under frozen parameters, by a blinded observer, before any model was trained. Model outputs were never used to inform reference measurements at any stage.

## Checklist antes de abrir o Fiji

- [x] Fiji baixado e instalado
- [x] WHST *updated* instalado (Plugins → Wound healing size tool aparece)
- [x] Métrica confirmada: área → closure fraction (não width)
- [x] Parâmetros HUVEC calibrados e congelados (radius=20, threshold=100, saturated=0.001)
- [x] Parâmetros SKOV-3 confirmados (mesmos do HUVEC — validado no poço Snap-34)
- [x] Convenção de nomes documentada (HUVEC `A2_24hr`, SKOV-3 `Snap-34-24h`)
- [x] Escala removida → medição em pixels²
- [x] Regra de correção manual definida (§9) + caso A2 documentado
- [ ] Cegamento garantido (saída do AI oculta durante medição)
- [ ] Estrutura de `whst_measurements.csv` pronta
- [ ] Fluxo de medição em massa definido (manual vs macro vs stacks)
