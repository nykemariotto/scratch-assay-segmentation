import json, os, csv, collections, glob
ROOT = os.environ.get("ROBOFLOW_EXPORT", "<roboflow_export>")
B = os.environ.get("SCRATCH_ASSAY_ROOT", ".")
with open(os.path.join(B,"data/mapping_huvec_final.csv"), encoding='utf-8-sig', newline='') as f:
    fin = list(csv.DictReader(f))
csv_names = set(r['arquivo_b'] for r in fin)
print("csv arquivo_b unicos:", len(csv_names))

for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT,d)
    if not os.path.isdir(p): continue
    tot_img=0; tot_ann=0; names=set(); per={}
    for j in glob.glob(os.path.join(p,"*","_annotations.coco.json")):
        data=json.load(open(j,encoding='utf-8'))
        sp=os.path.basename(os.path.dirname(j))
        per[sp]=(len(data['images']),len(data['annotations']))
        tot_img+=len(data['images']); tot_ann+=len(data['annotations'])
        for im in data['images']: names.add(im['file_name'])
    inter = names & csv_names
    print(f"{d}: imgs={tot_img} uniq_names={len(names)} anns={tot_ann} per={per}")
    print(f"   cobertos_pelo_csv={len(inter)}  orfaos={len(names-csv_names)}")
    if names-csv_names and len(names-csv_names)<=10:
        print("   orfaos:", sorted(names-csv_names))
