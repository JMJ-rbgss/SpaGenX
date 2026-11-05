from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
import torch

R=Path(os.environ.get('RUN_ROOT',Path(__file__).resolve().parents[1]))
D=Path(os.environ.get('SPAGENX_DATA_ROOT','/external/spagenx_data')); DEEP=D
PIPE=D/'pipeline'
ROWMAP=PIPE/'step3_rowmap_17organs'
FIX=D/'canonical_assets'
MAN=FIX/'corrected_gene_assets/gene_label_chunks_manifest.tsv'
OVER=FIX/'program256_canonical_identity_m16/gene_to_program_m16_overlay.pt'
VOCAB=PIPE/'step2_gene_rawlog1p/global_gene_vocab.tsv'
SPAGENX_CODE=Path(os.environ.get('SPAGENX_CODE_ROOT','/external/spagenx_code/training'))
SPAGENX_CHECKPOINT=D/'weights/SpaGenX_weights.pt'
SPAGENX_SHA256='0bcb29527826a54f6be4ffd1e9140fc35167d6b48b9ced5ff21ab98e60cc7603'
STCK=D/'comparison/stpath_weights.pth'
STSHA='10fecc5c43ca81aaf92c67b1fabd54f216d2baa864533eb24816c0d8dc2a39fc'
STROOT=D/'comparison/stpath'
STCODE=STROOT/'official_stpath_py39_compat'
STVOC=STROOT/'vocab_assets_v3/spagenx_canonical_symbol_to_stable_id.json'
CAN=D/'test_contract'
DIAG=D/'frozen_panels'
JULY=D/'comparison/gigapath/master_feature_store.pt'
SHARDS=CAN/'stpath_shards_combined'
BENCH=D/'comparison/hest_bench'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def js(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str)+'\n')
def table(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);(x if isinstance(x,pd.DataFrame) else pd.DataFrame(x)).to_csv(p,sep='\t',index=False)
def require(v,s):
 if not v:raise RuntimeError(s)
def normspot(sid,x):
 x=str(x).strip();prefix=sid+'_';return x[len(prefix):] if x.startswith(prefix) else x
def xy_normalized(x):
 x=np.asarray(x,dtype=np.float64);require(x.ndim==2 and x.shape[1]==2 and np.isfinite(x).all(),'COORD_NONFINITE_OR_SHAPE')
 span=np.ptp(x,axis=0);require((span>0).all(),'CONSTANT_COORD_AXIS');return (x-x.min(0))/span
def st_agent(device):
 import sys
 sys.path.insert(0,str(STCODE))
 from stpath.app.pipeline.inference import STPathInference
 if str(device)=='cpu':
  old=torch.load
  def cpu_load(*a,**kw):kw['map_location']='cpu';kw['weights_only']=False;return old(*a,**kw)
  torch.load=cpu_load
  try:agent=STPathInference(str(STVOC),str(STCK),device=torch.device('cpu'))
  finally:torch.load=old
 else:agent=STPathInference(str(STVOC),str(STCK),device=device)
 agent.model.eval();agent.model.requires_grad_(False)
 return agent
def giga_reader():
 store=torch.load(JULY,map_location='cpu',weights_only=False)
 by={}
 for i,s in enumerate(store['sid']):by.setdefault(str(s),[]).append(i)
 spot_lookup={}
 for sid,indices in by.items():
  names=[normspot(sid,store['spot_id'][i]) for i in indices]
  require(len(set(names))==len(names),'GIGAPATH_MASTER_DUPLICATE_SPOT:'+sid)
  spot_lookup[sid]=dict(zip(names,indices))
 def read(sid,plan):
  ii=np.asarray([r['feature_index'] for r in plan['rows']],dtype=int)
  if sid in by:
   expected=[normspot(sid,r['spot']) for r in plan['rows']]
   require(all(s in spot_lookup[sid] for s in expected),'GIGAPATH_MASTER_MISSING_SPOT:'+sid)
   resolved=np.array([spot_lookup[sid][s] for s in expected],dtype=int)
   global_match=np.array_equal(ii,resolved)
   local_match=bool(np.all(ii<len(by[sid]))) and np.array_equal(np.asarray(by[sid])[ii],resolved)
   require(global_match or local_match,'GIGAPATH_PLAN_INDEX_IDENTITY_CONFLICT:'+sid)
   ix=torch.as_tensor(resolved); f=store['features'][ix].float().numpy(); xy=store['coords'][ix].float().numpy();source=JULY
   sp=np.asarray(store['spot_id'],dtype=str)[ix.numpy()] if 'spot_id' in store else None
  else:
   source=SHARDS/(sid+'.pt');a=torch.load(source,map_location='cpu',weights_only=False);ix=torch.as_tensor(ii)
   f=a['features'][ix].float().numpy();xy=a['coords'][ix].float().numpy();sp=np.asarray(a['spot_id'],dtype=str)[ii] if 'spot_id' in a else None
  require(f.shape==(len(ii),1536) and np.isfinite(f).all(),'GIGAPATH_FEATURE_FAIL:'+sid)
  if sp is not None:require([normspot(sid,s) for s in sp]==[normspot(sid,r['spot']) for r in plan['rows']],'GIGAPATH_SPOT_ID_MISMATCH:'+sid)
  return f,xy,source
 return read
def original_coordinates(sid):
 roots=[D/'coordinates_primary',D/'coordinates_secondary']
 for root in roots:
  for tech in ['ST','Visium','VisiumHD','Xenium']:
   p=root/tech/'coord'/(sid+'_coord.csv')
   if p.exists():
    a=pd.read_csv(p,index_col=0);return a.index.astype(str).tolist(),a[['xaxis','yaxis']].to_numpy(float),p
 for root in roots:
  for tech in ['st','ST','Visium','VisiumHD','Xenium']:
   p=root/tech/(sid+'.h5ad')
   if p.exists():
    import anndata as ad
    a=ad.read_h5ad(p,backed='r');names=a.obs_names.astype(str).tolist();xy=np.asarray(a.obsm['spatial']);a.file.close();return names,xy,p
 raise RuntimeError('NO_COORD_SOURCE:'+sid)
def pcc(x,y):
 from scipy.stats import pearsonr
 if not np.isfinite(x).all() or not np.isfinite(y).all():return float('nan'),'nonfinite'
 if len(x)<2:return float('nan'),'too_few_spots'
 if np.ptp(y)==0:return float('nan'),'constant_measured'
 if np.ptp(x)==0:return float('nan'),'constant_predicted'
 return float(pearsonr(y.astype(float),x.astype(float))[0]),''
