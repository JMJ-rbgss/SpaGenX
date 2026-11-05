import argparse,sys,time,traceback,os
from common import *

def parameter_hash(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['SpaGenX','STPath'],required=True);a=ap.parse_args();name=a.model
 require(json.loads((R/'01_audit/preflight_status.json').read_text())['status']=='PASS','PREFLIGHT_NOT_PASS')
 cohort=pd.read_csv(R/'02_cohort/common_test_sections.tsv',sep='\t');device=torch.device('cuda:0');torch.set_num_threads(4)
 torch.cuda.reset_peak_memory_stats();started=time.time()
 if name=='SpaGenX':
  sys.path.insert(0,str(SPAGENX_CODE));import spagenx_trainer as T
  cb=T.load_codebooks(str(ROWMAP));rm=T.load_rowmap_tensors(str(ROWMAP));ov=torch.load(OVER,map_location='cpu',weights_only=False)
  model=T.SpaGenXModel(n_genes=115055,n_organs=cb['n_organs'],n_techs=cb['n_techs'],gene_to_program_ids=ov['gene_to_program_ids'],gene_to_program_mask=ov['gene_to_program_mask'],n_programs=256,feature_dim=1536,d_model=512,gene_bias=True,dropout=.1)
  ck=torch.load(SPAGENX_CHECKPOINT,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck
  model.eval();model.requires_grad_(False);before=parameter_hash(model);model=model.to(device)
  feat=T.load_feature_store(str(PIPE/'step1_features/master_feature_store.pt'),device,half=True)
  nbr=T.load_spagenx_neighbor_overlay(PIPE/'step4_overlays_17organs/spagenx_pool32_homo_hetero_overlay.pt',len(rm['row_to_feature_index']),device)
  rf=rm['row_to_feature_index'].long().to(device);rorg=rm['row_to_organ_id'].long().to(device);rtech=rm['row_to_tech_id'].long().to(device)
  def forward(rec,rr,gids):
   panels=torch.as_tensor(gids,device=device,dtype=torch.long)
   result=[]
   with torch.inference_mode():
    for start in range(0,len(rr),64):
     ids=torch.as_tensor(rr[start:start+64],device=device,dtype=torch.long);b=len(ids)
     nb=T.gather_spagenx_neighbors(nbr,ids,feat)
     zg=torch.zeros(b,0,dtype=torch.long,device=device);zv=torch.zeros(b,0,device=device);zm=torch.zeros(b,0,dtype=torch.bool,device=device)
     with torch.autocast('cuda',dtype=torch.bfloat16):
      z=model(center_img=feat[rf[ids]].float(),target_gene_ids=panels[None,:].expand(b,-1),organ_id=rorg[ids],tech_id=rtech[ids],p_gids=zg,p_log1p=zv,p_mask=zm,global_step=42000,ablation={'disable_prompt':True},**nb)
     require(not z['prompt_enabled_runtime'] and z['gene_bias_enabled_runtime'],'SpaGenX_RUNTIME_PROMPT_OR_BIAS_FAIL')
     require(z['p_ctx_norm'].item()==0,'SpaGenX_PROMPT_CONTEXT_NONZERO')
     result.append(z['pred_log1p'].float().cpu().numpy())
   return np.concatenate(result)
 else:
  agent=st_agent(0);model=agent.model;before=parameter_hash(model);readg=giga_reader();plan=json.loads((CAN/'result/stpath_c162_plan.json').read_text())
  mask_token=agent.tokenizer.ge_tokenizer.mask_token
  require(int(torch.count_nonzero(mask_token))==1 and float(mask_token.flatten()[1])==1,'STPATH_MASK_TOKEN_NOT_NATIVE')
  def fast(feats,xy,gids,rec):
   with torch.inference_mode():
    f=torch.as_tensor(np.ascontiguousarray(feats),device=device);c=agent._normalize_coords(torch.as_tensor(np.array(xy,copy=True),device=device))
    ie=model.input_encoder;n=len(f);x=ie.image_embed(f)+ie.gene_embed.weight[:,1].unsqueeze(0)
    tid=agent.tokenizer.tech_tokenizer.encode(str(rec.technology),align_first=True);oid=agent.tokenizer.organ_tokenizer.encode(str(rec.organ),align_first=True)
    x=x+ie.tech_embed(torch.full((n,),int(tid),device=device,dtype=torch.long))+ie.organ_embed(torch.full((n,),int(oid),device=device,dtype=torch.long))
    h=model.model(x,c,torch.zeros(n,dtype=torch.long,device=device))
    normalized=model.gene_exp_head[0](h);idx=torch.as_tensor(gids,device=device,dtype=torch.long)
    # Restrict only the final linear output rows, after the complete native spatial encoder.
    return torch.nn.functional.linear(normalized,model.gene_exp_head[1].weight[idx],model.gene_exp_head[1].bias[idx]).float().cpu().numpy()
  def forward(rec,rr,gids):
   f,xy,_=readg(rec.section_id,plan[rec.section_id]);require(len(f)==len(rr),'STPATH_ROW_COUNT_FAIL');return fast(f,xy,gids,rec)
 require(not model.training and all(not p.requires_grad for p in model.parameters()),'MODEL_NOT_FROZEN')
 smoke_sections=cohort.drop_duplicates('technology',keep='first').section_id.tolist();smoke=[]
 for sid in smoke_sections:
  rec=next(r for r in cohort.itertuples(index=False) if r.section_id==sid);z=np.load(R/'02_cohort/prepared'/(sid+'.npz'))
  rr=z['row_ids'];gid=z['gene_ids'] if name=='SpaGenX' else z['stpath_gene_ids']
  pred=forward(rec,rr[:32] if name=='SpaGenX' else rr,gid)
  require(pred.shape==((min(32,len(rr)) if name=='SpaGenX' else len(rr)),50) and np.isfinite(pred).all(),'SMOKE_SHAPE_OR_FINITE_FAIL:'+sid)
  record={'section_id':sid,'technology':rec.technology,'model':name,'n_spots':len(pred),'all_finite':True,'prompt_enabled':False,'mean_prediction':float(pred.mean())}
  if name=='STPath':
   f,xy,_=readg(sid,plan[sid]);ii=np.linspace(0,len(f)-1,min(32,len(f)),dtype=int)
   native=agent.inference(coords=np.array(xy[ii],copy=True),img_features=f[ii],context_ids=None,context_gene_exps=None,context_gene_names=None,organ_type=rec.organ,tech_type=rec.technology,save_gene_names=z['gene_symbols'].tolist()).X
   optimized=fast(f[ii],xy[ii],gid,rec);err=float(np.max(np.abs(native-optimized)))
   require(np.allclose(native,optimized,atol=1e-4,rtol=1e-4),'STPATH_NATIVE_HELPER_EQUIVALENCE_FAIL:'+sid+':'+str(err));record['native_helper_max_abs_error']=err
  else:
   old=np.load(DIAG/'predictions/section_hvg50_primary.npz');oldids=old['row_ids'];secids=old['section_ids'];sgi=old['selected_gene_ids'];secs=[str(s) for s in secids]
   if sid in secs:
    ix=secs.index(sid);oldgenes=sgi[ix];common=np.intersect1d(gid,oldgenes);rowindex={int(r):i for i,r in enumerate(oldids)}
    if all(int(r) in rowindex for r in rr[:len(pred)]) and len(common):
     ii=np.array([rowindex[int(r)] for r in rr[:len(pred)]]);aa=np.array([np.flatnonzero(gid==g)[0] for g in common]);bb=np.array([np.flatnonzero(oldgenes==g)[0] for g in common])
     expected=old['pred'][ii][:,bb]+model.gene_bias(torch.as_tensor(common,device=device)).detach().float().cpu().numpy().reshape(1,-1)
     err=float(np.max(np.abs(pred[:,aa]-expected)));record['bias_addition_common_genes']=len(common);record['bias_addition_max_abs_error']=err
     require(np.allclose(pred[:,aa],expected,atol=.02,rtol=.005),'SpaGenX_BIAS_ADDITION_DIAGNOSTIC_FAIL:'+sid+':'+str(err))
  smoke.append(record);print('SMOKE PASS',name,sid,flush=True)
 table(R/'01_audit'/('smoke_'+name+'.tsv'),smoke)
 js(R/'logs'/('smoke_'+name+'.json'),{'status':'PASS','sections':smoke_sections,'model':name,'parameter_hash_before':before,'prompt_forbidden':True,'smoke_seconds':time.time()-started})
 while not (R/'gate_allow_full.json').exists():
  if (R/'gate_abort.json').exists():raise RuntimeError('PEER_SMOKE_FAILED')
  time.sleep(1)
 done=[]
 for j,rec in enumerate(cohort.itertuples(index=False)):
  sid=rec.section_id;z=np.load(R/'02_cohort/prepared'/(sid+'.npz'));rr=z['row_ids'];gid=z['gene_ids'] if name=='SpaGenX' else z['stpath_gene_ids']
  start=time.time();pred=forward(rec,rr,gid);require(pred.shape==(len(rr),50) and np.isfinite(pred).all(),'FULL_PREDICTION_FAIL:'+sid)
  np.savez_compressed(R/'predictions'/name/(sid+'.npz'),row_ids=rr,spot_ids=z['spot_ids'],gene_ids=z['gene_ids'],pred=pred,prompt_enabled=np.array(False))
  done.append({'section_id':sid,'rows':len(rr),'seconds':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'model':name})
  table(R/'logs'/('progress_'+name+'.tsv'),done)
  print('FULL',name,j+1,'/162',sid,'rows',len(rr),flush=True)
 require(parameter_hash(model)==before,'PARAMETERS_CHANGED')
 require(sha(SPAGENX_CHECKPOINT if name=='SpaGenX' else STCK)==(SPAGENX_SHA256 if name=='SpaGenX' else STSHA),'CHECKPOINT_CHANGED')
 js(R/'logs'/('complete_'+name+'.json'),{'status':'PASS','model':name,'sections':len(done),'rows':sum(r['rows'] for r in done),'seconds':time.time()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'parameters_unchanged':True,'prompt_enabled':False,'checkpoint_unchanged':True})

if __name__=='__main__':
 try:main()
 except Exception as e:
  model=sys.argv[sys.argv.index('--model')+1];js(R/'logs'/('failed_'+model+'.json'),{'status':'BLOCKED','reason':str(e),'traceback':traceback.format_exc()});raise
