// ============================================================================
// stage4/whst_manual_correction.ijm — Correcao manual assistida das medicoes WHST
// Manuscript 4336348, Cytometry Part A
//
// ---------------------------------------------------------------------------
// CRITERIO DE BORDA (DECLARADO ANTES DA CORRECAO, aplicado uniformemente)
// ---------------------------------------------------------------------------
//   A borda da ferida e onde TERMINA O MONOLAYER CONFLUENTE e COMECA A REGIAO
//   ABERTA/ESPARSA.
//
//   Operacionalmente:
//     - Incluir na ferida (regiao aberta): area sem cobertura celular continua,
//       incluindo trechos com celulas isoladas/migratorias esparsas que ainda
//       NAO formam monolayer confluente.
//     - Excluir da ferida (monolayer): area com cobertura celular continua e
//       confluente, mesmo que a densidade varie.
//     - A borda acompanha o limite do MONOLAYER, nao o limite das celulas mais
//       avancadas isoladas.
//     - Debris, bolhas e artefatos NAO contam como ferida; se sobrepoem a
//       borda, interpolar a borda do monolayer por baixo do artefato.
//     - Criterio identico em todos os timepoints e em ambas as passadas.
//
//   Definido ANTES do inicio da correcao e versionado junto ao script
//   (ver tambem PROTOCOLO_CORRECAO_MANUAL.md) -> rastreavel.
// ---------------------------------------------------------------------------
//
// O QUE O MACRO FAZ
//   - Le a lista da passada (ja ordenada por serie -> contexto temporal).
//   - Abre cada imagem de whst_input/ e CARREGA o ROI automatico correspondente
//     (whst_output/rois/) no ROI Manager como PONTO DE PARTIDA do ajuste.
//   - Voce ajusta o ROI e confirma.
//   - Salva ROI + mascara em pasta NOVA, sem tocar em nada do original.
//
//   O IoU (auto vs corrigido, e passada1 vs passada2) e calculado depois em
//   Python por stage4/correction_agreement.py, a partir das mascaras salvas. Isso
//   mantem o macro simples e o calculo testavel/reproduzivel.
//
// ANOTACAO — NAO e feita aqui.
//   Aqui o trabalho e so tracar o contorno correto. A caracterizacao do que o
//   WHST errou e feita a parte, sobre os OVERLAYS AUTOMATICOS, em
//   stage1/annotation_sheet.csv (stage1/build_annotation_sheet.py / stage1/read_annotations.py):
//   o comentario descreve a falha do automatico, nao a correcao manual.

// IJM: funcoes so enxergam variaveis GLOBAIS declaradas com 'var'.
var jaFeito = newArray(0);

root = getDirectory("Select the project root folder (the one containing whst_input)");
inDir   = root + "whst_input" + File.separator;
roiAuto = root + "whst_output" + File.separator + "rois" + File.separator;

Dialog.create("Correcao manual WHST");
Dialog.addChoice("Passada:", newArray("1 - correcao (worklist completa)",
                                      "2 - RE-correcao cega (subconjunto)",
                                      "3 - validacao da procedencia",
                                      "4 - completar series mistas",
                                      "5 - baselines recuperados"));
Dialog.addMessage("CRITERIO DE BORDA:\nfim do monolayer confluente / inicio da regiao aberta ou esparsa.");
Dialog.show();
passada = Dialog.getChoice();

if (startsWith(passada, "1")) {
    listCsv = root + "stage4/correction_worklist.csv";
    outDir  = root + "whst_output" + File.separator + "rois_corrected" + File.separator;
    outCsv  = root + "stage4/manual_correction_pass1.csv";
    colFile = 1;                          // coluna 'whst_input_file' na worklist
} else if (startsWith(passada, "3")) {
    listCsv = root + "stage4/validation_worklist.csv";
    outDir  = root + "whst_output" + File.separator + "rois_validation" + File.separator;
    outCsv  = root + "stage4/manual_correction_validation.csv";
    colFile = 1;                          // mesma estrutura da worklist principal
} else if (startsWith(passada, "4")) {
    listCsv = root + "stage4/completion_worklist.csv";
    outDir  = root + "whst_output" + File.separator + "rois_completion" + File.separator;
    outCsv  = root + "stage4/manual_correction_completion.csv";
    colFile = 1;
} else if (startsWith(passada, "5")) {
    listCsv = root + "stage4/baseline_worklist.csv";
    outDir  = root + "whst_output" + File.separator + "rois_baselines" + File.separator;
    outCsv  = root + "stage4/manual_correction_baselines.csv";
    colFile = 1;
} else {
    listCsv = root + "stage4/.recorrecao_oculta.csv";
    outDir  = root + "whst_output" + File.separator + "rois_blind_repeat" + File.separator;
    outCsv  = root + "stage4/manual_correction_pass2.csv";
    colFile = 0;                          // arquivo oculto tem 1 coluna
}
if (!File.exists(listCsv)) exit("Nao encontrei a lista: " + listCsv);
File.makeDirectory(outDir);
maskDir = outDir + "masks" + File.separator;
File.makeDirectory(maskDir);
if (!File.exists(outCsv))
    File.saveString("whst_input_file,area_px_corrigida,area_pct_corrigida,area_pct_auto,status\n", outCsv);


// ---- le a lista ----
lines = split(File.openAsString(listCsv), "\n");
files = newArray(0);
for (i = 1; i < lines.length; i++) {                    // pula cabecalho
    ln = replace(lines[i], "\r", "");
    if (lengthOf(ln) < 3) continue;
    cols = split(ln, ",");
    if (cols.length <= colFile) continue;
    files = Array.concat(files, cols[colFile]);
}
print("=== Correcao manual WHST | " + passada + " ===");
print("imagens na lista: " + files.length);

setOption("BlackBackground", true);
setForegroundColor(255, 255, 255);
nDone = 0; nSkip = 0; nFech = 0; nInval = 0; nJa = 0;

// ---- retomada: le o CSV de saida e marca o que ja foi RESOLVIDO ----
// 'pulada' NAO conta como resolvido (a intencao e voltar nela depois);
// 'ok', 'fechada' e 'invalida' contam.
if (File.exists(outCsv)) {
    prev = split(File.openAsString(outCsv), "\n");
    for (p = 1; p < prev.length; p++) {
        ln = replace(prev[p], "\r", "");
        if (lengthOf(ln) < 3) continue;
        pc = split(ln, ",");
        if (pc.length < 5) continue;
        st = pc[pc.length - 1];
        if (st == "ok" || st == "fechada" || st == "invalida")
            jaFeito = Array.concat(jaFeito, pc[0]);
    }
}
function resolvido(nome) {
    for (q = 0; q < jaFeito.length; q++)
        if (jaFeito[q] == nome) return true;
    return false;
}

for (i = 0; i < files.length; i++) {
    fn = files[i];
    base = fn;
    if (endsWith(toLowerCase(base), ".tiff"))     base = substring(base, 0, lengthOf(base) - 5);
    else if (endsWith(toLowerCase(base), ".tif")) base = substring(base, 0, lengthOf(base) - 4);

    outRoi = outDir + base + ".roi";
    // retomavel: ja tem ROI salvo OU ja foi resolvido como fechada/invalida
    if (File.exists(outRoi) || resolvido(fn)) { nJa++; continue; }

    imgPath = inDir + fn;
    if (!File.exists(imgPath)) { print("FALTA imagem: " + fn); continue; }

    open(imgPath);
    orig = getTitle();
    run("Set Scale...", "distance=0 known=0 unit=pixel");
    setVoxelSize(1, 1, 1, "pixel");
    W = getWidth(); H = getHeight();

    // ---- ROI automatico como ponto de partida ----
    roiManager("reset");
    autoRoiPath = roiAuto + base + ".roi";
    areaAuto = 0;
    if (File.exists(autoRoiPath)) {
        roiManager("Open", autoRoiPath);
        roiManager("select", 0);
        getRawStatistics(areaAuto);                     // nPixels do ROI automatico
    } else {
        print("sem ROI auto (comecar do zero): " + base);
        run("Select None");
    }

    // ---- ajuste manual ----
    setTool("freehand");
    waitForUser("AJUSTE O ROI  (" + (i + 1) + "/" + files.length + ")",
        "CRITERIO: a borda e onde termina o MONOLAYER CONFLUENTE\n" +
        "e comeca a regiao ABERTA/ESPARSA.\n\n" +
        "COMO AJUSTAR (clique na JANELA DA IMAGEM primeiro):\n" +
        "  - freehand para redesenhar; SHIFT = somar outra regiao;\n" +
        "    ALT = subtrair.\n" +
        "  - Ferida partida em 2 pedacos? Trace o 1o, segure SHIFT\n" +
        "    e trace o 2o: a area vira a SOMA dos dois.\n" +
        "  - ROI automatico ja correto? So clique OK.\n\n" +
        "FERIDA FECHADA / PULAR / INVALIDA:\n" +
        "  1) clique na JANELA DA IMAGEM\n" +
        "  2) Ctrl+Shift+A   (= Edit > Selection > Select None)\n" +
        "     -> o contorno amarelo some\n" +
        "  3) clique OK aqui -> abre o menu com as 3 opcoes\n\n" +
        "Imagem: " + fn);

    pctAuto = 100.0 * areaAuto / (W * H);

    // ---- sem ROI: tres significados DIFERENTES, nao confundir ----
    //   fechada  -> area = 0. E MEDIDA VALIDA (closure = 1,0). Nao e dado ausente.
    //   pulada   -> nao corrigi agora; sem medida. Volta na proxima execucao.
    //   invalida -> frame descartado (nao da para segmentar).
    if (selectionType() < 0) {
        Dialog.create("Sem ROI — o que registrar?");
        Dialog.addChoice("Registrar como:", newArray(
            "FERIDA FECHADA  ->  area = 0 (medida valida)",
            "PULAR  ->  nao corrigir agora (volta depois)",
            "IMAGEM INVALIDA  ->  descartar este frame"));
        Dialog.addMessage("Ferida fechada NAO e o mesmo que pular:\n" +
                          "fechada = 0% de ferida (fechamento total);\n" +
                          "pular = sem medida, reaparece na proxima execucao.");
        Dialog.show();
        esc = Dialog.getChoice();

        if (startsWith(esc, "FERIDA")) {
            // mascara toda preta = area 0 (mantem o IoU calculavel em Python)
            newImage("ZEROMASK", "8-bit black", W, H, 1);
            saveAs("PNG", maskDir + base + "_mask.png");
            close();
            File.append(fn + ",0,0.000," + d2s(pctAuto, 3) + ",fechada", outCsv);
            nFech++;
        } else if (startsWith(esc, "IMAGEM")) {
            File.append(fn + ",NA,NA," + d2s(pctAuto, 3) + ",invalida", outCsv);
            nInval++;
        } else {
            File.append(fn + ",NA,NA," + d2s(pctAuto, 3) + ",pulada", outCsv);
            nSkip++;
        }
        selectWindow(orig);
        close();
        continue;
    }

    // ---- mede e salva o corrigido ----
    getRawStatistics(areaCorr);
    pctCorr = 100.0 * areaCorr / (W * H);

    roiManager("reset");
    roiManager("add");
    roiManager("select", 0);
    roiManager("save selected", outRoi);                // ROI corrigido (pasta nova)

    // mascara binaria do ROI corrigido -> insumo do IoU em Python
    newImage("CORRMASK", "8-bit black", W, H, 1);
    roiManager("select", 0);                            // aplica o ROI na mascara
    run("Fill", "slice");
    run("Select None");
    saveAs("PNG", maskDir + base + "_mask.png");
    close();                                            // fecha CORRMASK

    File.append(fn + "," + d2s(areaCorr, 0) + "," + d2s(pctCorr, 3) + "," +
                d2s(pctAuto, 3) + ",ok", outCsv);
    nDone++;

    selectWindow(orig);
    run("Select None");
    close();
}

roiManager("reset");
print("=== concluido ===");
print("corrigidas (ROI):   " + nDone);
print("ferida fechada(0):  " + nFech);
print("invalidas:          " + nInval);
print("puladas (voltam):   " + nSkip);
print("ja feitas antes:    " + nJa);
print("ROIs:      " + outDir);
print("Mascaras:  " + maskDir);
print("CSV:       " + outCsv);
print("");
print("Proximo passo: python stage4/correction_agreement.py");
