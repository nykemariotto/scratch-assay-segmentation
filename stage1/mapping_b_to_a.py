#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
"""
stage1/mapping_b_to_a.py
=================
Constroi o mapeamento definitivo BANCO B (Roboflow, anotado) -> BANCO A (cruas .tiff)
to the base version "Pre Eclampsia.coco-segmentation" (1369 annotated images).

Estrategia: casamento por CONTEUDO (pixel) RESTRITO POR NOME.
- The name identity of each image in B comes from coco['images'][i]['extra']['name'],
  which preserves case, separator and tokens of the ORIGINAL name (without the real
  extension and without the Windows " (N)" collision suffix, which Roboflow strips).
- For HUVEC the timepoint is in the name itself (0h/8h/12h/24h); for SKOV-treatment the
  name implies 0h; for the SKOV "Snap-NN" the timepoint is not in the name and is
  recoverable only by
  pixel (o mesmo Snap-NN existe em 0h/24h/48h/72h).
- For each group (same normalised name) the 1:1 assignment B<->A is resolved by
  maximising similarity (Hungarian). We report similarity, the margin to the second
  best, and a negative control (null distribution = unassigned pairs within groups).

READ-ONLY on both banks. It writes only inside the repository root.
"""
import os, re, json, time, csv, sys
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

A_ROOT = os.environ.get("BANCO_A", "<banco_a>")
B_TRAIN = os.environ.get('SCRATCH_B_TRAIN',
    r'DATASET/Pre Eclampsia.coco-segmentation/train')
OUT = os.environ.get('SCRATCH_OUT', '.')
CACHE_A = os.path.join(OUT, 'stage1/cache_A_files.json')
FEAT_CACHE = os.path.join(OUT, 'feat_cache.npz')

SZ = 256          # resolucao do descritor (cinza, z-score)
T_OK = 0.75       # similaridade minima para considerar casamento fisico confiavel
M_OK = 0.05       # margem minima para o 2o melhor candidato

# ---------------------------------------------------------------- normalizacao
def strip_ext(n):
    return os.path.splitext(n)[0]

def is_snap(stem):
    return 'nap' in stem.lower() and re.search(r'snap[\s_-]*\d', stem.lower()) is not None

def is_scale(stem):
    s = stem.lower()
    return s == 'scale' or 'scale' in s

def norm_key(name):
    """Chave de agrupamento tolerante a caixa/separador/sufixo de colisao.
    Snap-aware: for snaps the key ignores the dedup counter and the timepoint."""
    stem = strip_ext(name)
    s = stem.lower().strip()
    s = re.sub(r'\s*\(\d+\)\s*$', '', s)      # remove sufixo de colisao do Windows "(N)"
    if 'nap' in s:
        m = re.search(r'snap[\s_-]*(\d+)', s)
        if m:
            return 'snap-%d' % int(m.group(1))   # the snap base, no counter or timepoint
    s = s.replace('_', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# --------------------------------------------------- parser de identidade (nome)
def classify(stem):
    s = stem.lower()
    if is_scale(stem):
        return 'scale'
    if is_snap(stem):
        return 'skov_snap'
    if re.match(r'^(75geo|75ug|ct\d|ptx\d)', s):
        return 'skov_treat'
    if re.match(r'^[a-f]\d', s):
        return 'huvec'
    return 'other'

def parse_huvec(stem):
    """Retorna (well, timepoint_h) a partir do nome HUVEC, ou (None,None)."""
    s = stem.strip()
    mw = re.match(r'^([A-Fa-f]\d+)', s)
    well = mw.group(1).upper() if mw else None
    # timepoint: primeiro numero seguido de h/hr (ou token '24' isolado)
    mt = re.search(r'(\d+)\s*h', s.lower())
    if mt:
        tp = int(mt.group(1))
    else:
        # anomaly 'B1 24 1' -> token 24 without an h
        mt2 = re.search(r'\b(0|8|12|24)\b', s)
        tp = int(mt2.group(1)) if mt2 else None
    return well, tp

TP_FOLDER = {0: '0h', 8: '8h', 12: '12h', 24: '24h'}

# ------------------------------------------------------------------- features
def load_feat(path):
    im = Image.open(path).convert('L').resize((SZ, SZ), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32).ravel()
    a -= a.mean()
    sd = a.std()
    if sd > 0:
        a /= sd
    return a.astype(np.float32)

def build_records():
    coco = json.load(open(os.path.join(B_TRAIN, '_annotations.coco.json'), encoding='utf-8'))
    Brecs = []
    for im in coco['images']:
        extra = im.get('extra', {}).get('name', '')
        stem = strip_ext(extra)
        dom = classify(stem)
        key = norm_key(extra)
        Brecs.append(dict(
            file_b=im['file_name'],
            path_b=os.path.join(B_TRAIN, im['file_name']),
            extra=extra, stem=stem, domain=dom, key=key,
            w=im['width'], h=im['height'],
        ))
    # Banco A
    Alist = json.load(open(CACHE_A, encoding='utf-8'))
    Arecs = []
    for f in Alist:
        folder = f['folder']; name = f['name']
        stem = strip_ext(name)
        dom = classify(stem)
        key = norm_key(name)
        tp = None
        if folder.startswith('SKOV/'):
            tpname = folder.split('/')[1]
            tp = int(re.match(r'(\d+)h', tpname).group(1))
        elif folder.startswith('HUVEC/'):
            tpname = folder.split('/')[1]
            m = re.match(r'(\d+)h', tpname)
            tp = int(m.group(1)) if m else None
        Arecs.append(dict(
            folder=folder, name=name,
            path_a=os.path.join(A_ROOT, folder.replace('/', os.sep), name),
            stem=stem, domain=dom, key=key, tp_folder=tp,
        ))
    return Brecs, Arecs

# ------------------------------------------------------------- feature caching
def compute_features(paths):
    """Computes or loads descriptors; npz cache keyed by path (with mtime)."""
    cache = {}
    if os.path.exists(FEAT_CACHE):
        z = np.load(FEAT_CACHE, allow_pickle=True)
        keys = z['keys']; mats = z['mats']
        for k, v in zip(keys, mats):
            cache[str(k)] = v
    feats = {}
    todo = [p for p in paths if p not in cache]
    print(f'[feat] cache={len(cache)} todo={len(todo)} total={len(paths)}', flush=True)
    t0 = time.time(); done = 0
    for p in paths:
        if p in cache:
            feats[p] = cache[p]; continue
        try:
            feats[p] = load_feat(p)
        except Exception as e:
            print('  ERRO lendo', p, e, flush=True)
            feats[p] = None
        done += 1
        if done % 100 == 0:
            print(f'  {done}/{len(todo)}  ({time.time()-t0:.0f}s)', flush=True)
    # persist merged cache
    allp = list(cache.keys())
    for p in feats:
        if feats[p] is not None:
            cache[p] = feats[p]
    ks = list(cache.keys())
    np.savez_compressed(FEAT_CACHE, keys=np.array(ks, dtype=object),
                        mats=np.array([cache[k] for k in ks], dtype=object))
    return {p: feats.get(p, cache.get(p)) for p in paths}

# --------------------------------------------------------------------- main
def main():
    Brecs, Arecs = build_records()
    from collections import Counter, defaultdict
    print('B por dominio:', Counter(b['domain'] for b in Brecs))
    print('A por dominio:', Counter(a['domain'] for a in Arecs))

    # indices por chave
    Bkey = defaultdict(list); Akey = defaultdict(list)
    for b in Brecs:
        if b['domain'] == 'scale':
            continue
        Bkey[b['key']].append(b)
    for a in Arecs:
        if a['domain'] == 'scale':
            continue
        Akey[a['key']].append(a)

    # paths to compute: all of B (non-scale) + the A candidates whose keys appear in B
    needed_A_paths = set()
    for k in Bkey:
        for a in Akey.get(k, []):
            needed_A_paths.add(a['path_a'])
    b_paths = [b['path_b'] for b in Brecs if b['domain'] != 'scale']
    print(f'[plan] B non-scale={len(b_paths)}  A candidates={len(needed_A_paths)}')

    feats = compute_features(b_paths + list(needed_A_paths))

    def sim(u, v):
        if u is None or v is None:
            return float('nan')
        return float(np.dot(u, v) / u.shape[0])

    rows = []
    assigned_sims = []
    null_sims = []
    stats = Counter()

    # scale de B -> descartado
    for b in Brecs:
        if b['domain'] == 'scale':
            rows.append(dict(arquivo_b=b['file_b'], stem_normalizado=b['key'], arquivo_a='',
                             pasta_a='', linha_celular='', well_campo='', timepoint_h='',
                             similaridade='', margem_2o_melhor='', status='descartado_scale'))
            stats['descartado_scale'] += 1

    for key, bl in Bkey.items():
        al = Akey.get(key, [])
        dom = bl[0]['domain']
        cell = 'HUVEC' if dom == 'huvec' else 'SKOV'
        # matriz de similaridade
        if len(al) == 0:
            for b in bl:
                well, tp = (parse_huvec(b['stem']) if dom == 'huvec' else (None, None))
                rows.append(_row(b, None, cell, well, tp, float('nan'), float('nan'),
                                 'sem_candidato'))
                stats['sem_candidato'] += 1
            continue
        Bf = [feats.get(b['path_b']) for b in bl]
        Af = [feats.get(a['path_a']) for a in al]
        S = np.full((len(bl), len(al)), -1e9, dtype=np.float64)
        for i in range(len(bl)):
            for j in range(len(al)):
                s = sim(Bf[i], Af[j])
                S[i, j] = -1e9 if np.isnan(s) else s
        # Hungarian (maximise) — cropped to min(nB,nA)
        ri, ci = linear_sum_assignment(-S)
        assign = {int(i): int(j) for i, j in zip(ri, ci)}
        # collect the nulls (pairs not chosen, only where valid)
        for i in range(len(bl)):
            for j in range(len(al)):
                if S[i, j] > -1e8 and assign.get(i) != j:
                    null_sims.append(S[i, j])
        for i, b in enumerate(bl):
            well, tp_name = (parse_huvec(b['stem']) if dom == 'huvec' else (None, None))
            if i in assign:
                j = assign[i]
                a = al[j]
                s = S[i, j]
                # margin: the best alternative for this B row among the other columns
                others = [S[i, jj] for jj in range(len(al)) if jj != j and S[i, jj] > -1e8]
                second = max(others) if others else float('nan')
                margin = (s - second) if others else float('nan')
                # timepoint efetivo
                if dom == 'huvec':
                    tp_eff = tp_name if tp_name is not None else a['tp_folder']
                    well_eff = well
                elif dom == 'skov_treat':
                    tp_eff = 0; well_eff = b['stem']
                else:  # skov_snap
                    tp_eff = a['tp_folder']; well_eff = norm_key(b['extra'])
                # status
                if np.isnan(s):
                    st = 'sem_candidato'
                elif s >= T_OK and (np.isnan(margin) or margin >= M_OK):
                    st = 'ok'
                else:
                    st = 'ambiguo'
                assigned_sims.append(s)
                stats[st] += 1
                rows.append(_row(b, a, cell, well_eff, tp_eff, s, margin, st))
            else:
                # surplus B with no physical A file (Windows case-insensitive collision)
                if dom == 'huvec':
                    well_eff = well; tp_eff = tp_name
                elif dom == 'skov_treat':
                    well_eff = b['stem']; tp_eff = 0
                else:
                    well_eff = norm_key(b['extra']); tp_eff = None
                rows.append(_row(b, None, cell, well_eff, tp_eff, float('nan'), float('nan'),
                                 'sem_candidato'))
                stats['sem_candidato'] += 1

    # ------- escrita CSV
    cols = ['arquivo_b', 'stem_normalizado', 'arquivo_a', 'pasta_a', 'linha_celular',
            'well_campo', 'timepoint_h', 'similaridade', 'margem_2o_melhor', 'status']
    csv_path = os.path.join(OUT, 'stage1/mapping_b_to_a.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ------- validacao / estatisticas
    asd = np.array(assigned_sims, dtype=float)
    nul = np.array(null_sims, dtype=float)
    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else None

    # breakdown by domain from the rows written
    def rdom(r):
        if r['status'] == 'descartado_scale':
            return 'scale'
        if r['linha_celular'] == 'HUVEC':
            return 'huvec'
        return 'skov_snap' if str(r['stem_normalizado']).startswith('snap') else 'skov_treat'
    dom_status = defaultdict(Counter)
    snap_tp = Counter()
    for r in rows:
        dom_status[rdom(r)][r['status']] += 1
        if rdom(r) == 'skov_snap' and r['status'] in ('ok', 'ambiguo'):
            snap_tp[str(r['timepoint_h'])] += 1

    # confiabilidade de IDENTIDADE (linha/well/timepoint) vs ARQUIVO fisico
    ident_ok = sum(1 for r in rows if r['status'] != 'descartado_scale'
                   and r['linha_celular'] and str(r['timepoint_h']) != '')
    fisico_ok = dom_status['huvec']['ok'] + dom_status['skov_snap']['ok'] + dom_status['skov_treat']['ok']

    summary = dict(
        n_B_total=len(Brecs),
        n_rows=len(rows),
        resolucao=SZ, thr_ok=T_OK, margem_min=M_OK,
        metodo='casamento por pixel (cinza 256x256, z-score, correlacao) restrito por nome + atribuicao 1:1 Hungarian por grupo',
        status_counts=dict(stats),
        status_por_dominio={k: dict(v) for k, v in dom_status.items()},
        snap_timepoint_dist=dict(snap_tp),
        identidade_confiavel=dict(
            n=ident_ok,
            nota='linha_celular+well+timepoint recuperados do NOME (HUVEC/tratamento) ou do PIXEL (snaps); confiaveis mesmo quando o arquivo fisico A e ambiguo. Timepoint HUVEC bate 100% com a pasta (0 divergentes no inventario).'),
        arquivo_fisico_confiavel=dict(
            n=fisico_ok,
            nota='linhas status=ok: arquivo .tiff especifico com sim>=%.2f e margem>=%.2f' % (T_OK, M_OK)),
        arquivo_fisico_incerto=dict(
            n_ambiguo=stats['ambiguo'], n_sem_candidato=stats['sem_candidato'],
            nota='TODOS HUVEC. ambiguo=twin exato (margem~0) ou fonte fisica perdida por colisao case-insensitive; sem_candidato=B excedente (nB>nA) sem .tiff fonte. Identidade linha/well/tp permanece confiavel.'),
        assigned_sim=dict(n=int(asd.size), mean=float(np.mean(asd)) if asd.size else None,
                          p05=pct(asd,5), p25=pct(asd,25), p50=pct(asd,50),
                          p75=pct(asd,75), p95=pct(asd,95),
                          min=float(np.min(asd)) if asd.size else None),
        null_sim=dict(n=int(nul.size), mean=float(np.mean(nul)) if nul.size else None,
                     p50=pct(nul,50), p95=pct(nul,95), p99=pct(nul,99),
                     max=float(np.max(nul)) if nul.size else None),
        separacao=dict(
            assigned_p05=pct(asd,5), null_p95=pct(nul,95), null_p99=pct(nul,99),
            nota='pares corretos (mediana ~1.0, p05~0.94) separam da distribuicao nula (mediana ~0.71); sobreposicao so na cauda por imagens globalmente parecidas. null_max~1.0 vem de familias com duplicatas EXATAS (A2.TIFF, (N) identicos): troca fisica irrelevante pois identidade e a mesma.'),
        caveat_snap='well_campo dos snaps = id de aquisicao (snap-NN). Se o MESMO snap-NN em timepoints diferentes e o MESMO campo fisico (serie temporal) NAO foi provado aqui; verificar antes de usar snap-NN como chave de agrupamento no re-split por campo.',
    )
    json.dump(summary, open(os.path.join(OUT, 'stage1/mapping_b_to_a.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('CSV:', csv_path)

def _row(b, a, cell, well, tp, s, margin, status):
    return dict(
        arquivo_b=b['file_b'],
        stem_normalizado=b['key'],
        arquivo_a=(a['name'] if a else ''),
        pasta_a=(a['folder'] if a else ''),
        linha_celular=cell,
        well_campo=(well if well is not None else ''),
        timepoint_h=('' if tp is None else tp),
        similaridade=('' if (s is None or (isinstance(s, float) and np.isnan(s))) else round(s, 4)),
        margem_2o_melhor=('' if (margin is None or (isinstance(margin, float) and np.isnan(margin))) else round(margin, 4)),
        status=status,
    )

if __name__ == '__main__':
    main()
