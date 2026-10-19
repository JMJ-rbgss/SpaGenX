#!/usr/bin/env python3
"""Portable single-slide SpaGenX inference. No training or web service."""
from pathlib import Path
import os,sys,json,csv,hashlib,argparse,gc,importlib.util,time
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('TMPDIR',str(ROOT/'work_tmp'))
os.environ.setdefault('PYTHONPYCACHEPREFIX',str(ROOT/'work_cache/pycache'))
os.environ.setdefault('HF_HOME',str(ROOT/'work_cache/huggingface'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'work_cache/torch'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'work_cache'))
import numpy as np,torch
from PIL import Image
from scipy.spatial import cKDTree

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(8<<20),b''):h.update(v)
 return h.hexdigest()
def js(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str)+'\n')
def require(x,s):
 if not x:raise RuntimeError(s)
def config(root):return json.loads((Path(root)/'bundle_config.json').read_text())
def asset(root,key):
 p=Path(config(root)['paths'][key]);return p if p.is_absolute() else Path(root)/p
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def load_m2(root,device):
 cfg=config(root);p=asset(root,'m2_weights');require(sha(p)==cfg['m2_sha256'],'M2_WEIGHT_SHA256')
 sys.path.insert(0,str(asset(root,'model_code').parent));m=module(asset(root,'model_code'),'portable_m2_definition')
 ov=torch.load(asset(root,'program'),map_location='cpu',weights_only=False)
 model=m.M2Pool32TypedEvidenceRawLog1pV3(n_genes=115055,n_organs=17,n_techs=4,gene_to_program_ids=ov['gene_to_program_ids'],gene_to_program_mask=ov['gene_to_program_mask'],n_programs=256,feature_dim=1536,d_model=512,alpha_init=.1,gene_bias=True,dropout=.1)
 ck=torch.load(p,map_location='cpu',weights_only=False);require(int(ck['step'])==42000,'CHECKPOINT_STEP')
 model.load_state_dict(ck['model'],strict=True)
 for name in ('gene_to_program_ids','gene_to_program_mask'):require(torch.equal(ck['model']['program_prior.'+name],ov[name]),'PROGRAM_BUFFER_IDENTITY:'+name)
 del ck,ov;model.eval();model.requires_grad_(False);model.to(device)
 return model
def parameter_hash(model):
 h=hashlib.sha256()
 for n,p in model.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def query(root,items):
 with asset(root,'vocab').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
 by_id={int(r['global_gene_id']):r['gene_symbol'] for r in rows};by_symbol={}
 for g,s in by_id.items():by_symbol.setdefault(s,[]).append(g)
 gids=[];symbols=[]
 for item in items:
  if item.isdigit():g=int(item);require(g in by_id,'UNKNOWN_GENE_ID:'+item)
  else:require(len(by_symbol.get(item,[]))==1,'UNKNOWN_OR_AMBIGUOUS_GENE:'+item);g=by_symbol[item][0]
  require(0<=g<115055 and g not in gids,'GENE_RANGE_OR_DUPLICATE:'+item);gids.append(g);symbols.append(by_id[g])
 require(len(gids)>0,'EMPTY_GENE_QUERY');return np.asarray(gids,np.int64),symbols
def organ_id(root,organ):
 cb=json.loads(asset(root,'codebooks').read_text());names=cb['organ_id'];matched=[(n,k) for n,k in names.items() if n.lower()==organ.lower()]
 require(len(matched)==1,'ORGAN_MUST_BE_ONE_OF_17:'+organ);return int(matched[0][1])
def neighbors(coords,features):
 n=len(coords);require(n>0 and features.shape==(n,1536) and np.isfinite(coords).all(),'NEIGHBOR_INPUT')
 require(len(np.unique(coords,axis=0))==n,'DUPLICATE_PATCH_COORDINATES')
 ci=np.full((n,32),-1,np.int64);mask=np.zeros((n,32),bool);dist=np.zeros((n,32),np.float32);cos=np.zeros((n,32),np.float32)
 if n>1:
  d,idx=cKDTree(coords).query(coords,k=min(n,33))
  for i in range(n):
   pairs=[(int(j),float(dd)) for j,dd in zip(np.atleast_1d(idx[i]),np.atleast_1d(d[i])) if int(j)!=i and int(j)<n][:32]
   for k,(j,dd) in enumerate(pairs):ci[i,k]=j;dist[i,k]=dd;mask[i,k]=True
 scale=max(float(np.median(dist[:,0][mask[:,0]])),1e-6) if mask[:,0].any() else 1.
 dist/=scale;norm=torch.nn.functional.normalize(torch.as_tensor(features).float(),dim=1)
 for st in range(0,n,512):
  stop=min(st+512,n);ix=torch.as_tensor(ci[st:stop].clip(0));cos[st:stop]=(norm[st:stop,None,:]*norm[ix]).sum(-1).numpy();cos[st:stop][~mask[st:stop]]=0
 out={'candidate_local_indices':torch.as_tensor(ci),'candidate_valid_mask':torch.as_tensor(mask),'candidate_distance':torch.as_tensor(dist),'candidate_cosine':torch.as_tensor(cos)}
 for prefix,k in [('homo',12),('hetero',8)]:
  inds=np.full((n,k),-1,np.int64);dd=np.zeros((n,k),np.float32);cc=np.zeros((n,k),np.float32);valid=np.zeros((n,k),bool)
  for i in range(n):
   present=np.flatnonzero(mask[i]);ranked=present[np.argsort(-cos[i,present],kind='stable')];chosen=ranked[:12] if prefix=='homo' else ranked[12:][np.argsort(cos[i,ranked[12:]],kind='stable')][:8]
   z=len(chosen);inds[i,:z]=ci[i,chosen];dd[i,:z]=dist[i,chosen];cc[i,:z]=cos[i,chosen];valid[i,:z]=True
  out[prefix+'_local_indices']=torch.as_tensor(inds);out[prefix+'_valid_mask']=torch.as_tensor(valid);out[prefix+'_distance']=torch.as_tensor(dd);out[prefix+'_cosine']=torch.as_tensor(cc)
 out.update(distance_unit_scale_level0_pixels=scale,neighbor_mode='slide_local_candidate32_homo12_hetero8',candidate_k=32,homo_k=12,hetero_k=8,patch_coordinates_sha256=hashlib.sha256(np.ascontiguousarray(coords,np.float32).tobytes()).hexdigest())
 return out
def predict(root,features,overlay,gids,organ,device='cpu',batch=16,chunk=256,rows=None):
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dev=torch.device(device);model=load_m2(root,dev);before=parameter_hash(model)
 f=torch.as_tensor(features).float().to(dev);indices=np.arange(len(f)) if rows is None else np.asarray(rows,np.int64);results=[]
 require(f.ndim==2 and f.shape[1]==1536 and torch.isfinite(f).all(),'FEATURE_SHAPE_OR_FINITE')
 for prefix,k in [('candidate',32),('homo',12),('hetero',8)]:
  ix=overlay[prefix+'_local_indices'];mask=overlay[prefix+'_valid_mask'];require(ix.shape==(len(f),k) and mask.shape==ix.shape,'NEIGHBOR_SHAPE')
  require(bool(((ix[mask]>=0)&(ix[mask]<len(f))).all()),'NEIGHBOR_INDEX_RANGE')
  for field in ('distance','cosine'):require(torch.isfinite(overlay[prefix+'_'+field]).all(),'NEIGHBOR_NONFINITE')
 oid=organ_id(root,organ);g=torch.as_tensor(gids,dtype=torch.long,device=dev)
 with torch.inference_mode():
  for st in range(0,len(indices),batch):
   ids=indices[st:st+batch];b=len(ids);kw={}
   for prefix in ('candidate','homo','hetero'):
    ix=overlay[prefix+'_local_indices'][ids].long().to(dev);valid=overlay[prefix+'_valid_mask'][ids].bool().to(dev)
    kw[prefix+'_img']=f[ix.clamp(min=0)]*valid[...,None];kw[prefix+'_mask']=valid
    kw[prefix+'_dist']=overlay[prefix+'_distance'][ids].float().to(dev);kw[prefix+'_cos']=overlay[prefix+'_cosine'][ids].float().to(dev)
   pieces=[]
   for lo in range(0,len(g),chunk):
    z=model(center_img=f[torch.as_tensor(ids,device=dev)],target_gene_ids=g[lo:lo+chunk][None,:].expand(b,-1),organ_id=torch.full((b,),oid,device=dev,dtype=torch.long),tech_id=torch.zeros(b,device=dev,dtype=torch.long),p_gids=torch.empty(b,0,device=dev,dtype=torch.long),p_log1p=torch.empty(b,0,device=dev),p_mask=torch.empty(b,0,device=dev,dtype=torch.bool),global_step=42000,ablation={'tech_zero':True,'disable_prompt':True},**kw)
    require(not z['prompt_enabled_runtime'] and z['gene_bias_enabled_runtime'] and float(z['p_ctx_norm'])==0,'FORWARD_PROTOCOL')
    value=z['pred_log1p'].float().cpu().numpy();require(np.isfinite(value).all(),'NONFINITE_PREDICTION');pieces.append(value)
   results.append(np.concatenate(pieces,axis=1))
 require(before==parameter_hash(model),'PARAMETERS_CHANGED');del model,f;gc.collect()
 if dev.type=='cuda':torch.cuda.empty_cache()
 return np.concatenate(results),{'global_step':42000,'no_prompt':True,'tech_zero':True,'gene_bias':True,'organ':organ,'precision':'FP32','patch_batch':batch,'gene_chunk':chunk,'parameters_unchanged':True,'checkpoint_sha256':config(root)['m2_sha256']}

def preprocess(root,image_path,out,mpp=None,max_patches=16000,device='cuda:0'):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);image_path=Path(image_path);model=None;slide=None;small=None
 try:
  require(image_path.is_file(),'IMAGE_NOT_FOUND');require(1<=max_patches<=16000,'INVALID_PATCH_LIMIT')
  try:
   import openslide
   slide=openslide.OpenSlide(str(image_path))
  except Exception:slide=None
  if slide:
   vals=[]
   for key in ('openslide.mpp-x','aperio.MPP'):
    try:v=float(slide.properties[key]);require(np.isfinite(v) and v>0,'INVALID_MPP');vals.append(v)
    except KeyError:pass
   source_mpp=float(mpp) if mpp is not None else (vals[0] if vals else None);width,height=slide.dimensions
   if 'openslide.mpp-y' in slide.properties and mpp is None and source_mpp is not None:require(abs(float(slide.properties['openslide.mpp-y'])/source_mpp-1)<.01,'ANISOTROPIC_MPP_NEEDS_EXPLICIT_POLICY')
  else:
   small=Image.open(image_path);width,height=small.size;require(width*height<=64000000,'REGULAR_IMAGE_TOO_LARGE_USE_PYRAMIDAL_WSI');source_mpp=float(mpp) if mpp is not None else None
  require(source_mpp is not None and np.isfinite(source_mpp) and source_mpp>0,'MPP_REQUIRED')
  footprint=max(1,int(round(224*.5/source_mpp)));require(width>=footprint and height>=footprint,'IMAGE_TOO_SMALL')
  level=slide.get_best_level_for_downsample(.5/source_mpp) if slide else 0;ds=float(slide.level_downsamples[level]) if slide else 1.;readsize=max(1,int(round(footprint/ds)))
  ref=module(asset(root,'preprocess_code'),'uni2_preprocess_reference');patches=[];coords=[];features=[];ratios=[];candidate_count=0;rejected=0
  def flush():
   nonlocal model,patches
   if not patches:return
   if model is None:
    require(sha(asset(root,'uni2_weights'))==config(root)['uni2_sha256'],'UNI2_WEIGHT_SHA256')
    model=ref.load_uni2h(str(asset(root,'uni2_weights')),torch.device(device),1536);model.requires_grad_(False)
    state=torch.load(asset(root,'uni2_weights'),map_location='cpu',weights_only=False);state=state.get('state_dict',state);state={k.replace('module.',''):v for k,v in state.items()}
    require(set(state)==set(model.state_dict()),'UNEXPLAINED_UNI2_STATE_KEYS');del state
   v=ref.infer_batch(model,patches,torch.device(device),amp=True);features.append(v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v,np.float32));patches=[]
  for y in range(0,height-footprint+1,footprint):
   for x in range(0,width-footprint+1,footprint):
    candidate_count+=1
    patch=slide.read_region((x,y),level,(readsize,readsize)).convert('RGB') if slide else small.crop((x,y,x+footprint,y+footprint)).convert('RGB')
    if patch.size!=(224,224):patch=patch.resize((224,224),Image.Resampling.BICUBIC)
    arr=np.asarray(patch,np.uint8);white=float((arr.mean(2)>=205).mean())
    if white>.8:rejected+=1;continue
    require(len(coords)<max_patches,'PATCH_LIMIT_EXCEEDED_NO_SAMPLING')
    coords.append((x+footprint/2,y+footprint/2));ratios.append(white);patches.append(arr)
    if len(patches)>=8:flush()
  require(len(coords)>0,'NO_VALID_TISSUE_PATCHES');flush();feat=np.concatenate(features);crd=np.asarray(coords,np.float32)
  require(feat.shape==(len(coords),1536) and np.isfinite(feat).all(),'ENCODER_OUTPUT_SHAPE_OR_FINITE')
  torch.save({'features':torch.as_tensor(feat),'coords_level0_center':torch.as_tensor(crd),'white_ratio':torch.as_tensor(ratios),'feature_dim':1536,'source_mpp':source_mpp,'target_mpp':.5,'patch_size':224,'auto_grid':True,'patch_coordinates_sha256':hashlib.sha256(crd.tobytes()).hexdigest()},out/'features.pt')
  torch.save(neighbors(crd,feat),out/'neighbors.pt')
  with (out/'patches.tsv').open('w') as f:
   f.write('patch_id\tx_level0_center\ty_level0_center\twhite_ratio\n')
   for i,((x,y),w) in enumerate(zip(coords,ratios)):f.write(f'patch_{i:06d}\t{x}\t{y}\t{w}\n')
  js(out/'preprocess.json',{'status':'PASS','source_image':str(image_path.resolve()),'source_sha256':sha(image_path),'source_mpp':source_mpp,'target_mpp':.5,'patch_size':224,'stride_target_pixels':224,'source_footprint_pixels':footprint,'white_gray_threshold':205,'white_ratio_threshold':.8,'candidates':candidate_count,'valid_patches':len(coords),'rejected_white':rejected,'partial_edges_omitted':True,'max_patches':max_patches,'encoder_sha256':config(root)['uni2_sha256'],'new_application_sampling_protocol':True})
 except Exception as e:js(out/'FAILED.json',{'status':'BLOCKED','reason':str(e)});raise
 finally:
  if slide:slide.close()
  if small:small.close()
  del model;gc.collect()
  if str(device).startswith('cuda'):torch.cuda.empty_cache()
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('preprocess');a.add_argument('--image',required=True);a.add_argument('--out',required=True);a.add_argument('--mpp',type=float);a.add_argument('--max-patches',type=int,default=16000);a.add_argument('--device',default='cuda:0')
 a=sub.add_parser('infer');a.add_argument('--features',required=True);a.add_argument('--neighbors',required=True);a.add_argument('--genes-file',required=True);a.add_argument('--organ',required=True);a.add_argument('--out',required=True);a.add_argument('--device',default='cuda:0');a.add_argument('--patch-batch',type=int,default=16);a.add_argument('--gene-chunk',type=int,default=256)
 a=p.parse_args();torch.set_num_threads(4)
 if a.command=='preprocess':preprocess(a.bundle,a.image,a.out,a.mpp,a.max_patches,a.device)
 else:
  out=Path(a.out);out.mkdir(parents=True,exist_ok=False);ft=torch.load(a.features,map_location='cpu',weights_only=False);ov=torch.load(a.neighbors,map_location='cpu',weights_only=False);require(ft.get('patch_coordinates_sha256')==ov.get('patch_coordinates_sha256') and ft.get('patch_coordinates_sha256') is not None,'FEATURE_NEIGHBOR_COORDINATE_IDENTITY');items=[s.strip() for s in Path(a.genes_file).read_text().splitlines() if s.strip()];gids,symbols=query(a.bundle,items)
  y,meta=predict(a.bundle,ft['features'],ov,gids,a.organ,a.device,a.patch_batch,a.gene_chunk)
  np.savez_compressed(out/'predictions.npz',pred_log1p=y,global_gene_ids=gids,gene_symbols=np.asarray(symbols),coords_level0_center=ft['coords_level0_center'].numpy(),patch_ids=np.asarray([f'patch_{i:06d}' for i in range(len(y))]))
  meta.update(features_sha256=sha(a.features),neighbors_sha256=sha(a.neighbors),gene_order=symbols,n_patches=len(y),n_genes=len(gids),status='PASS');js(out/'inference.json',meta)
if __name__=='__main__':main()
