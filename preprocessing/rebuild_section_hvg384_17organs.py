#!/usr/bin/env python3

raise RuntimeError('ARCHIVAL_ONLY: historical in-place/site-specific protocol; use documented current CLI stages, not direct execution')
import csv, json, hashlib, time
from pathlib import Path
import torch

BASE = Path((__import__('os').environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data') + ''))
RUN = BASE/"new_pipeline_800gb/full_run_firstv1_rawlog1p_20260703_110639"
MANIFEST = RUN/"step2_gene_rawlog1p/gene_label_chunks_manifest.tsv"
OUT = BASE/"deepseek_outcome/spagenx_v3_17organs_continue_20260819/section_hvg384_17organs"
OUT.mkdir(parents=True, exist_ok=True)

rows=[]
with open(MANIFEST, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

section_map={}
summary=[]; fail=[]
for i,r in enumerate(rows):
    sid = str(r.get("sid") or r.get("sample_id") or r.get("section_id") or r.get("gene_chunk_id"))
    cp = Path(r["chunk_path"])
    if not cp.exists():
        fail.append({"sid":sid,"chunk_path":str(cp),"error":"missing_chunk"})
        continue
    try:
        obj=torch.load(cp,map_location="cpu",weights_only=False)
        csr=obj["target_genes_raw_local_csr"]
        gids=obj["local_to_global_gene_idx"]
        mask=obj.get("observed_gene_mask_local", None)
        n=int(csr.shape[0])
        if mask is not None:
            mask_np = mask.cpu().numpy().astype(bool) if hasattr(mask,"cpu") else mask.astype(bool)
        else:
            mask_np = None
        s1=csr.sum(axis=0).A1; s2=csr.multiply(csr).sum(axis=0).A1
        mean=s1/max(n,1); var=s2/max(n,1)-mean*mean
        if mask_np is not None:
            var=var.copy(); var[~mask_np]=-1.0
        gid_np = gids.cpu().long().numpy() if hasattr(gids,"cpu") else gids.numpy()
        valid_idx=[j for j,v in enumerate(var) if v>=0]
        valid_idx.sort(key=lambda j:(-float(var[j]), int(gid_np[j])))
        top_idx=valid_idx[:384]
        top_gids=torch.as_tensor([int(gid_np[j]) for j in top_idx], dtype=torch.long)
        section_map[sid]=top_gids
        summary.append({
            "sid":sid,"gene_chunk_id":r.get("gene_chunk_id",""),"chunk_path":str(cp),
            "n_rows":n,"n_local_genes":int(csr.shape[1]),"n_observed_genes":len(valid_idx),
            "n_hvg384":int(top_gids.numel()),"top10_gene_ids":",".join(str(int(x)) for x in top_gids[:10].tolist()),
        })
    except Exception as e:
        fail.append({"sid":sid,"chunk_path":str(cp),"error":repr(e)})
    if (i+1)%100==0:
        print("processed", i+1, flush=True)

pt=OUT/"section_hvg384_ids.pt"
torch.save(section_map, pt)
with open(OUT/"section_hvg384_summary.tsv","w",newline="") as f:
    fields=["sid","gene_chunk_id","chunk_path","n_rows","n_local_genes","n_observed_genes","n_hvg384","top10_gene_ids"]
    w=csv.DictWriter(f, delimiter="\t", fieldnames=fields); w.writeheader(); w.writerows(summary)
with open(OUT/"section_hvg384_failures.tsv","w",newline="") as f:
    fields=["sid","chunk_path","error"]
    w=csv.DictWriter(f, delimiter="\t", fieldnames=fields); w.writeheader(); w.writerows(fail)
report={
    "task_status":"PASS" if not fail and section_map else "FAIL",
    "created_at":time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_manifest":str(MANIFEST),
    "source_manifest_sha256":sha256_file(MANIFEST),
    "n_manifest_rows":len(rows),
    "n_sections_with_hvg384":len(section_map),
    "n_failures":len(fail),
    "hvg_per_section":384,
    "hvg_method":"per-section variance over current raw_counts_log1p CSR local observed genes; tie-break global_gene_id ascending",
    "output_pt":str(pt),
}
(OUT/"section_hvg384_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report, indent=2, sort_keys=True))
