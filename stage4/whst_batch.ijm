// ============================================================================
// stage4/whst_batch.ijm — Medição WHST em lote, sem diálogo
// Manuscript 4336348, Cytometry Part A
//
// Replica a lógica do Wound Healing Size Tool (Suarez-Arnedo et al. 2020)
// com os parâmetros CONGELADOS, sem abrir o diálogo interativo.
//
// DIFERENÇAS DELIBERADAS EM RELAÇÃO AO ORIGINAL:
//   1. Não chama 'Wound healing size tool options' -> sem diálogo, roda em lote
//   2. FORÇA medição em pixels (o original restaura a escala da imagem antes
//      de medir, na linha 304 do original -- é a causa do output em inches)
//   3. Grava CSV incremental com diagnóstico (n_rois, status)
//
// PARÂMETROS CONGELADOS (protocolo protocols/WHST_protocol_frozen.md):
//   variance radius = 20 | threshold = 100 | saturated = 0.001
//   diagonal = No (não afeta área) | escala removida -> pixels²
//
// USO:
//   Plugins -> Macros -> Run...  -> selecionar este arquivo
//   Escolher a pasta de imagens e o destino do CSV quando solicitado.
// ============================================================================

// ---- Parâmetros congelados ----
var filter_radius = 20;
var threshold     = 100;
var sat_pix       = 0.001;
var min_area      = 100;
var diagonal      = "No";   // área não é afetada; mantido por fidelidade

// ---- Funções auxiliares (copiadas do WHST original) ----
function edge_min_coordinates(arr_ROI) {
    arr_min = newArray();
    arr_min = Array.concat(arr_min, arr_ROI[0]);
    for (x = 1; x < arr_ROI.length - 2; x++) {
        if (arr_ROI[x-1] > arr_ROI[x] && arr_ROI[x] < arr_ROI[x+1]) {
            arr_min = Array.concat(arr_min, arr_ROI[x]);
        }
    }
    return arr_min;
}

function edge_max_coordinates(arr_ROI) {
    arr_max = newArray();
    for (x = 1; x < arr_ROI.length - 2; x++) {
        if (arr_ROI[x-1] < arr_ROI[x] && arr_ROI[x] > arr_ROI[x+1]) {
            arr_max = Array.concat(arr_max, arr_ROI[x]);
        }
    }
    arr_max = Array.concat(arr_max, arr_ROI[arr_ROI.length - 1]);
    return arr_max;
}

function diff_arrays(arr_max, arr_min) {
    arr_diff = newArray();
    n = minOf(arr_max.length, arr_min.length);
    for (x = 0; x < n - 1; x++) {
        arr_diff = Array.concat(arr_diff, arr_max[x] - arr_min[x]);
    }
    return arr_diff;
}

// ============================================================================
// PRINCIPAL
// ============================================================================
inputDir  = getDirectory("Pasta com as imagens a medir");
outputDir = getDirectory("Onde salvar o CSV");
csvPath   = outputDir + "data/whst_batch_results.csv";

// Subpastas para os contornos (criadas se não existirem)
roiDir  = outputDir + "rois" + File.separator;
maskDir = outputDir + "masks" + File.separator;
polyDir = outputDir + "polygons" + File.separator;
ovlDir  = outputDir + "overlays" + File.separator;
File.makeDirectory(roiDir);
File.makeDirectory(maskDir);
File.makeDirectory(polyDir);
File.makeDirectory(ovlDir);
run("Input/Output...", "jpeg=92");

// Cabeçalho
File.saveString("filename,area_px,area_pct,width_px,width_sd,n_rois,status,unit_check,pw_check\n", csvPath);

list = getFileList(inputDir);
setBatchMode(true);
run("Options...", " black");
setForegroundColor(0, 0, 0);
setBackgroundColor(255, 255, 255);

nOK = 0; nFail = 0;

for (i = 0; i < list.length; i++) {
    fn = list[i];
    low = toLowerCase(fn);
    if (!(endsWith(low, ".tif") || endsWith(low, ".tiff") ||
          endsWith(low, ".png") || endsWith(low, ".jpg") || endsWith(low, ".jpeg")))
        continue;

    open(inputDir + fn);
    orig = getTitle();

    // ---- FORÇA pixels (dupla proteção) ----
    // setVoxelSize é mais confiável que run("Set Scale...") em batch mode.
    // Mesmo assim, a área é lida com getRawStatistics (imune a calibração).
    run("Set Scale...", "distance=0 known=0 unit=pixel");
    setVoxelSize(1, 1, 1, "pixel");

    run("Select None");
    roiManager("reset");

    // ---- Pipeline de segmentação (idêntico ao WHST) ----
    run("Duplicate...", "title=WORK duplicate");
    selectWindow("WORK");
    run("8-bit");
    run("Set Scale...", "distance=0 known=0 unit=pixel");
    setVoxelSize(1, 1, 1, "pixel");
    run("Enhance Contrast...", "saturated=" + sat_pix + " normalize");
    run("Variance...", "radius=" + filter_radius);
    setThreshold(0, threshold);
    run("Convert to Mask", "black");
    run("Fill Holes");
    run("Select All");
    run("Analyze Particles...",
        "size=" + min_area + "-Infinity circularity=0.00-1.00 show=Nothing add");
    close();  // fecha WORK -> volta para a original

    selectWindow(orig);
    nR = roiManager("count");

    if (nR == 0) {
        // nenhuma partícula: registra e segue
        File.append(fn + ",NA,NA,NA,NA,0,no_roi,NA,NA", csvPath);
        nFail++;
        close();
        continue;
    }

    // ---- Selecionar o MAIOR componente (idêntico ao WHST) ----
    if (nR > 1) {
        areas = newArray(nR);
        for (k = 0; k < nR; k++) {
            roiManager("select", k);
            getRawStatistics(nPix, mean, mn, mx, sd);
            areas[k] = nPix;
        }
        largest = 0; large = 0;
        for (k = 0; k < nR; k++) {
            if (areas[k] > largest) { largest = areas[k]; large = k; }
        }
        sel_idx = large;
    } else {
        sel_idx = 0;
    }
    roiManager("select", sel_idx);

    // ---- SALVAR O CONTORNO (ROI já ativo na imagem) ----
    stem = File.nameWithoutExtension;

    // 1) ROI no formato ImageJ -> permite reabrir e EDITAR no Manual Tool
    saveAs("Selection", roiDir + stem + ".roi");

    // 2) Polígono (vértices) -> comparável com a saída polygonal do YOLO-seg
    // Usa File.open/print (bufferizado). NÃO usar concatenação de string em
    // loop: é O(n^2) no ImageJ macro e trava com ~7000 vértices por imagem.
    Roi.getCoordinates(px, py);
    fpoly = File.open(polyDir + stem + "_polygon.csv");
    print(fpoly, "x,y");
    for (k = 0; k < px.length; k++) print(fpoly, "" + px[k] + "," + py[k]);
    File.close(fpoly);

    // 3) Máscara binária -> para IoU/Dice fora do ImageJ
    run("Create Mask");
    saveAs("PNG", maskDir + stem + "_mask.png");
    close();               // fecha a máscara
    selectWindow(orig);          // volta para a imagem original

    // 4) OVERLAY para verificação visual (crua + contorno ciano)
    // A máscara binária serve para IoU/Dice; o overlay é o que permite
    // julgar visualmente se o contorno traça a ferida real.
    run("Select None");                       // sem isso, Duplicate recorta na seleção
    run("Duplicate...", "title=OVERLAY");
    selectWindow("OVERLAY");
    run("RGB Color");
    roiManager("select", sel_idx);
    setForegroundColor(0, 255, 255);          // ciano
    run("Line Width...", "line=6");
    run("Draw", "slice");
    run("Select None");
    saveAs("Jpeg", ovlDir + stem + "_overlay.jpg");
    close();
    setForegroundColor(0, 0, 0);              // restaura para o pipeline
    selectWindow(orig);
    roiManager("select", sel_idx);            // restaura a seleção

    // ---- Largura média e desvio (flag de instabilidade) ----
    Roi.getContainedPoints(xp, yp);
    min_x = edge_min_coordinates(xp);
    max_x = edge_max_coordinates(xp);
    dv = diff_arrays(max_x, min_x);
    Array.getStatistics(dv, lo, hi, avg_width, std_dist);

    // ---- Área em PIXELS via contagem bruta ----
    // getRawStatistics devolve nPixels (contagem literal), imune a qualquer
    // calibração embutida no TIFF (DPI). Corrige o bug em que imagens com DPI
    // embutido eram medidas em inches^2 (ex.: A2 0hr -> 142.149 em vez de 1310043).
    getRawStatistics(area, mean, mn, mx, sd);
    total_area = getWidth() * getHeight();
    area_pct = (area / total_area) * 100;

    getPixelSize(u_chk, pw_chk, ph_chk);
    File.append(fn + "," + d2s(area, 3) + "," + d2s(area_pct, 3) + "," +
                d2s(avg_width, 3) + "," + d2s(std_dist, 3) + "," + nR + ",ok," +
                u_chk + "," + d2s(pw_chk, 6), csvPath);
    nOK++;

    run("Select None");
    close();
}

setBatchMode(false);
print("=== WHST batch concluido ===");
print("Medidas OK: " + nOK);
print("Sem ROI:    " + nFail);
print("CSV: " + csvPath);
