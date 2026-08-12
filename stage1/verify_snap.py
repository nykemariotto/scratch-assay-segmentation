# -*- coding: utf-8 -*-
import os
# Independent verification of SKOV snap timepoint matching.
# Re-reads pixels from disk; uses two independent descriptors.
import os, re, json, csv, random, collections, sys
import numpy as np
from PIL import Image

A_ROOT = os.environ.get("BANCO_A", "<banco_a>")
B_TRAIN = os.environ.get('SCRATCH_B_TRAIN',
    r'DATASET/Pre Eclampsia.coco-segmentation/train')
CACHE_A = 'stage1/cache_A_files.json'

def norm_key_snap(name):
    s=os.path.splitext(name)[0].lower().strip()
    s=re.sub(r'\s*\(\d+\)\s*$','',s)
    if 'nap' in s:
        m=re.search(r'snap[\s_-]*(\d+)',s)
        if m: return 'snap-%d'%int(m.group(1))
    return None

# descriptor 1: replicate original (gray 256, bilinear, zscore, dot/N corr)
def feat256(path):
    im=Image.open(path).convert('L').resize((256,256),Image.BILINEAR)
    a=np.asarray(im,dtype=np.float32).ravel(); a-=a.mean(); s=a.std()
    if s>0: a/=s
    return a
# descriptor 2: independent -> gray 96, LANCZOS, then zscore Pearson
def feat96(path):
    im=Image.open(path).convert('L').resize((96,96),Image.LANCZOS)
    a=np.asarray(im,dtype=np.float32).ravel(); a-=a.mean(); s=a.std()
    if s>0: a/=s
    return a
def corr(u,v):
    return float(np.dot(u,v)/u.shape[0])

# A snap candidates grouped
CA=json.load(open(CACHE_A,encoding='utf-8'))
Agrp=collections.defaultdict(list)
for f in CA:
    k=norm_key_snap(f['name'])
    if k: Agrp[k].append((f['folder'],f['name']))

# B snap images from COCO
coco=json.load(open(os.path.join(B_TRAIN,'_annotations.coco.json'),encoding='utf-8'))
Bsnaps=[]
for im in coco['images']:
    extra=im.get('extra',{}).get('name','')
    k=norm_key_snap(extra)
    if k and 'scale' not in extra.lower():
        Bsnaps.append((im['file_name'],extra,k))

# load mapping to compare
mp={}
for r in csv.DictReader(open('stage1/mapping_b_to_a.csv',encoding='utf-8')):
    mp[r['arquivo_b']]=r

random.seed(7)
sample=random.sample(Bsnaps, 18)
print('Verifying %d snaps independently\n'%len(sample))
disagree=0
adj_flags=[]
for fb,extra,k in sample:
    bpath=os.path.join(B_TRAIN,fb)
    u256=feat256(bpath); u96=feat96(bpath)
    # per-timepoint best sim under each descriptor
    perTP={}
    for fold,name in Agrp[k]:
        tp=fold.split('/')[1]
        ap=os.path.join(A_ROOT,fold.replace('/',os.sep),name)
        try:
            s256=corr(u256,feat256(ap)); s96=corr(u96,feat96(ap))
        except Exception as e:
            print('  ERR',ap,e); continue
        cur=perTP.get(tp,(-9,-9,''))
        if s256>cur[0]: perTP[tp]=(s256,s96,name)
    order=sorted(perTP.items(), key=lambda kv:-kv[1][0])
    best_tp,(bs256,bs96,bname)=order[0]
    second=order[1] if len(order)>1 else None
    margin=bs256-second[1][0] if second else float('nan')
    # descriptor2 winner
    best_tp96=max(perTP.items(), key=lambda kv:kv[1][1])[0]
    mapped=mp[fb]['timepoint_h']+'h'
    ok = (best_tp==mapped)
    ok96 = (best_tp96==mapped)
    if not ok: disagree+=1
    # 2nd best adjacency
    tp_num={'0h':0,'24h':24,'48h':48,'72h':72}
    flag=''
    print('%-40s map=%s  win256=%s(%.4f) win96=%s  2nd=%s(%.4f) margin=%.4f %s%s'%(
        extra, mapped, best_tp,bs256, best_tp96, second[0] if second else '-',
        second[1][0] if second else float('nan'), margin,
        '' if ok else '<<< DISAGREE256', '' if ok96 else ' <<<DISAGREE96'))
print('\nDisagreements with mapping (256):', disagree)
