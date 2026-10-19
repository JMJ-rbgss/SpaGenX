#!/usr/bin/env python3
from __future__ import annotations

raise RuntimeError('ARCHIVAL_ONLY: historical in-place/site-specific protocol; use documented current CLI stages, not direct execution')
import argparse,csv,gc,hashlib,json,os,sys,time
from pathlib import Path
import torch

ROOT=Path((__import__('os').environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data') + ''))
RUN=ROOT/"deepseek_outcome/result5_latest_m2_17organ_revalidation_20260921_051035"
TRANSFER=ROOT/"deepseek_outcome/m2_tcga_brca_molecular_transfer_validation_20260720_233651"
CACHE=TRANSFER/"03_m2_prediction"
CKPT=ROOT/"deepseek_outcome/m2_lineage_differential_audit_20260918_024756/early_divergence_audit/recovery_gate_4ddp_20260918_075646/continue_full_40k_patience/best_per_gene_pcc_checkpoint.pt"
PROGRAM=ROOT/"deepseek_outcome/m2_v3_17organ_gene_identity_fix_20260909_225119/result/program256_canonical_identity_m16/gene_to_program_m16_overlay.pt"
GENES=TRANSFER/"01_contract/frozen_m2_rna_gene_universe.tsv"
OLD_PANEL=ROOT/"deepseek_outcome/result5_primary16_sensitivity32_20260724_234510/panel_definition_frozen.tsv"
EXPECTED="0bcb29527826a54f6be4ffd1e9140fc35167d6b48b9ced5ff21ab98e60cc7603"

def parse():
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path);p.add_argument("--device",required=True);p.add_argument("--worker-index",type=int,default=0);p.add_argument("--tile-batch",type=int,default=64);p.add_argument("--gene-chunk",type=int,default=1024);p.add_argument("--smoke",action="store_true");return p.parse_args()
def read_tsv(p):
 with open(p,newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def sha(p,block=16<<20):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(block),b""):h.update(b)
 return h.hexdigest()
def atomic_save(obj,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);q=RUN/"tmp"/(path.name+f".{os.getpid()}.partial");torch.save(obj,q);os.replace(q,path)
def atomic_json(obj,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);q=RUN/"tmp"/(path.name+f".{os.getpid()}.partial");q.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n");os.replace(q,path)
def load_model(device):
 if sha(CKPT)!=EXPECTED:raise RuntimeError("CHECKPOINT_HASH_MISMATCH")
 ov=torch.load(PROGRAM,map_location="cpu",weights_only=False);pids=ov["gene_to_program_ids"];pmask=ov["gene_to_program_mask"]
 sys.path.insert(0,str(RUN/"scripts"));from m2_v3_model import M2Pool32TypedEvidenceRawLog1pV3
 model=M2Pool32TypedEvidenceRawLog1pV3(n_genes=115055,n_organs=17,n_techs=4,gene_to_program_ids=pids,gene_to_program_mask=pmask,n_programs=256,feature_dim=1536,d_model=512,alpha_init=.1,gene_bias=True,dropout=.1).to(device)
 ck=torch.load(CKPT,map_location="cpu",weights_only=False)
 if int(ck["step"])!=42000:raise RuntimeError("STEP_MISMATCH:%s"%ck["step"])
 inc=model.load_state_dict(ck["model"],strict=True)
 if inc.missing_keys or inc.unexpected_keys:raise RuntimeError(f"STRICT_LOAD_FAIL:{inc}")
 model.eval();return model

def load_genes(device):
 rows=read_tsv(GENES);g=torch.tensor([int(x["global_gene_id"]) for x in rows],dtype=torch.long,device=device);s=[x["gene_symbol"] for x in rows]
 if len(g)!=38330 or len(set(g.tolist()))!=38330:raise RuntimeError("GENE_UNIVERSE_FAIL")
 return g,s

def load_patient(pid,device):
 fp=CACHE/"slide_feature_cache"/f"{pid}.pt";op=CACHE/"m2_neighbor_overlay"/f"{pid}.pt"
 f=torch.load(fp,map_location="cpu",weights_only=False);o=torch.load(op,map_location="cpu",weights_only=False)
 if o["patch_coordinates_sha256"]!=f["patch_coordinates_sha256"]:raise RuntimeError(f"COORD_HASH_MISMATCH:{pid}")
 feats=f["features"].float()
 if not torch.isfinite(feats).all():raise RuntimeError(f"FEATURE_NONFINITE:{pid}")
 data={"features":feats.to(device)}
 for name in ("candidate","homo","hetero"):
  data[name+"_ids"]=o[name+"_local_indices"].long().to(device);data[name+"_mask"]=o[name+"_valid_mask"].bool().to(device);data[name+"_dist"]=o[name+"_distance"].float().to(device);data[name+"_cos"]=o[name+"_cosine"].float().to(device)
 return f,o,data,fp,op

def forward(model,data,idx,gids,device):
 idx=torch.as_tensor(list(idx),dtype=torch.long,device=device);B=len(idx);feats=data["features"];kw={}
 for name in ("candidate","homo","hetero"):
  ids=data[name+"_ids"][idx];kw[name+"_img"]=feats[ids];kw[name+"_mask"]=data[name+"_mask"][idx];kw[name+"_dist"]=data[name+"_dist"][idx];kw[name+"_cos"]=data[name+"_cos"][idx]
 tg=gids[None,:].expand(B,-1);eg=torch.empty((B,0),dtype=torch.long,device=device);ev=torch.empty((B,0),dtype=torch.float32,device=device);em=torch.empty((B,0),dtype=torch.bool,device=device)
 with torch.inference_mode():
  out=model(center_img=feats[idx],target_gene_ids=tg,organ_id=torch.full((B,),2,dtype=torch.long,device=device),tech_id=torch.zeros(B,dtype=torch.long,device=device),p_gids=eg,p_log1p=ev,p_mask=em,global_step=42000,ablation={"tech_zero":True},**kw)
 return out["pred_log1p"].float()

def smoke(model,gids,device):
 pid="SYNTHETIC_PATIENT";f,o,data,fp,op=load_patient(pid,device);idx=range(min(64,len(data["features"])))
 panel=read_tsv(OLD_PANEL);p32=[int(x["global_gene_id"]) for x in panel if x["panel"]=="Sensitivity32"]
 p32=list(dict.fromkeys(p32));pos={int(g):i for i,g in enumerate(gids.tolist())};pi=[pos[x] for x in p32]
 filler=[i for i,g in enumerate(gids.tolist()) if int(g) not in set(p32)][:1024-len(p32)]
 chunk_idx=pi+filler;direct=forward(model,data,idx,gids[pi],device);chunk=forward(model,data,idx,gids[chunk_idx],device)[:,:len(pi)]
 diff=float((direct-chunk).abs().max());res={"status":"PASS" if diff<=1e-5 else "FAIL","patient_id":pid,"patches":len(idx),"genes_direct":32,"genes_chunk":1024,"max_abs_diff":diff,"threshold":1e-5,"finite":bool(torch.isfinite(direct).all() and torch.isfinite(chunk).all()),"gpu_peak_memory_bytes":int(torch.cuda.max_memory_allocated()),"checkpoint_sha256":EXPECTED,"precision":"fp32"}
 atomic_json(res,RUN/"smoke/chunk_vs_direct32.json");print(json.dumps(res,indent=2));return res["status"]=="PASS"

def main():
 a=parse();device=torch.device(a.device)
 if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
 torch.cuda.reset_peak_memory_stats();model=load_model(device);gids,symbols=load_genes(device)
 if a.smoke:raise SystemExit(0 if smoke(model,gids,device) else 20)
 if not a.manifest:raise RuntimeError("MANIFEST_REQUIRED")
 rows=read_tsv(a.manifest);pred_dir=RUN/"00_inference/patient_gene_m2_pseudobulk";audit_dir=RUN/"00_inference/patient_audits"
 print(json.dumps({"event":"START","worker":a.worker_index,"patients":len(rows),"device":a.device,"tile_batch":a.tile_batch,"gene_chunk":a.gene_chunk}),flush=True)
 for ordinal,row in enumerate(rows,1):
  pid=row["patient_id"];outp=pred_dir/f"{pid}.pt";ap=audit_dir/f"{pid}.json"
  if outp.exists() and ap.exists() and json.loads(ap.read_text()).get("status")=="PASS":print(f"SKIP {pid}",flush=True);continue
  t0=time.time();torch.cuda.reset_peak_memory_stats();f,o,data,fp,op=load_patient(pid,device);n=len(data["features"]);G=len(gids);sumv=torch.zeros(G,dtype=torch.float64);sumsq=torch.zeros(G,dtype=torch.float64);calls=0;pmin=float("inf");pmax=float("-inf")
  for ts in range(0,n,a.tile_batch):
   te=min(n,ts+a.tile_batch)
   for gs in range(0,G,a.gene_chunk):
    ge=min(G,gs+a.gene_chunk);p=forward(model,data,range(ts,te),gids[gs:ge],device)
    if not torch.isfinite(p).all():raise RuntimeError(f"NONFINITE:{pid}:{ts}:{gs}")
    d=p.double();sumv[gs:ge]+=d.sum(0).cpu();sumsq[gs:ge]+=(d*d).sum(0).cpu();calls+=1;pmin=min(pmin,float(p.min()));pmax=max(pmax,float(p.max()))
  mean=sumv/n;std=(sumsq/n-mean*mean).clamp_min(0).sqrt();fsha=sha(fp);osha=sha(op)
  obj={"patient_id":pid,"global_gene_ids":gids.cpu(),"gene_symbols":symbols,"predicted_mean":mean.float(),"predicted_spatial_std":std.float(),"n_patches":n,"scale":"M2_native_raw_counts_log1p_output","checkpoint_sha256":EXPECTED,"checkpoint_step":42000,"organ_id":2,"organ":"Breast","tech_id_runtime":0,"tech_zero":True,"native_empty_prompt":True,"candidate_k":32,"homo_k":12,"hetero_k":8,"tile_batch":a.tile_batch,"gene_chunk":a.gene_chunk,"precision":"fp32","feature_sha256":fsha,"neighbor_overlay_sha256":osha,"patch_coordinates_sha256":f["patch_coordinates_sha256"],"tile_predictions_saved":False,"outcome_files_read":False,"training_started":False,"schema_version":"latest_m2_17_result5_pseudobulk_v1"}
  atomic_save(obj,outp);elapsed=time.time()-t0;au={"status":"PASS","worker":a.worker_index,"patient_id":pid,"ordinal":ordinal,"assigned":len(rows),"n_patches":n,"n_genes":G,"forward_calls":calls,"elapsed_seconds":elapsed,"pred_min":pmin,"pred_max":pmax,"gpu_peak_memory_bytes":int(torch.cuda.max_memory_allocated()),"feature_sha256":fsha,"neighbor_overlay_sha256":osha,"checkpoint_sha256":EXPECTED,"outcome_read":False,"wsi_read":False,"finite":True}
  atomic_json(au,ap);print(json.dumps(au),flush=True);del f,o,data,obj,mean,std,sumv,sumsq;gc.collect();torch.cuda.empty_cache()
 print(json.dumps({"event":"DONE","worker":a.worker_index,"patients":len(rows)}),flush=True)
if __name__=="__main__":main()
