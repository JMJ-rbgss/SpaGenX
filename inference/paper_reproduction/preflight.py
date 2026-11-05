import sys, json, time, traceback
from common import *

def main():
 for s in ['01_audit','02_cohort/prepared','02_cohort/truth','03_metrics','04_figures','05_report','predictions/SpaGenX','predictions/STPath','logs','cache']:(R/s).mkdir(parents=True,exist_ok=True)
 js(R/'05_report/final_status.json',{'status':'RUNNING','stage':'preflight','no_training':True,'prompt_forbidden_both_models':True})
 hashes={str(p):sha(p) for p in [SPAGENX_CHECKPOINT,STCK,VOCAB,STVOC,OVER,MAN,CAN/'result/section_manifest.tsv',CAN/'result/stpath_c162_plan.json',DIAG/'analysis/section_hvg50_selection_primary.tsv']}
 require(hashes[str(SPAGENX_CHECKPOINT)]==SPAGENX_SHA256 and hashes[str(STCK)]==STSHA,'CHECKPOINT_SHA_MISMATCH')
 require(hashes[str(STVOC)]=='94736c62b58a5c7560adbd52f96db9148a60bbe73bbcf8704fa38a38527f682c','STPATH_VOCAB_SHA_MISMATCH')
 require(hashes[str(DIAG/'analysis/section_hvg50_selection_primary.tsv')]=='2559fa196f7fd36c876531257478f1e3e40acbb28bfd1851b19f389c5bf13095','HVG_FREEZE_CHANGED')
 sys.path.insert(0,str(SPAGENX_CODE));import spagenx_trainer as T
 v=pd.read_csv(VOCAB,sep='\t');require(np.array_equal(v.global_gene_id,np.arange(len(v))),'NONCONTIGUOUS_SpaGenX_VOCAB')
 ov=torch.load(OVER,map_location='cpu',weights_only=False);ck=torch.load(SPAGENX_CHECKPOINT,map_location='cpu',weights_only=False)
 require(int(ck['step'])==42000,'SPAGENX_CHECKPOINT_STEP_MISMATCH')
 for k in ['gene_to_program_ids','gene_to_program_mask']:require(torch.equal(ck['model']['program_prior.'+k],ov[k]),'PROGRAM_BUFFER_MISMATCH')
 m=T.SpaGenXModel(n_genes=115055,n_organs=17,n_techs=4,gene_to_program_ids=ov['gene_to_program_ids'],gene_to_program_mask=ov['gene_to_program_mask'],n_programs=256,feature_dim=1536,d_model=512,gene_bias=True,dropout=.1)
 m.load_state_dict(ck['model'],strict=True);m.eval();m.requires_grad_(False);del m,ck
 agent=st_agent('cpu');require(not agent.model.training and all(not p.requires_grad for p in agent.model.parameters()),'STPATH_NOT_FROZEN')
 cohort=pd.read_csv(CAN/'result/section_manifest.tsv',sep='\t');require(len(cohort)==162 and cohort.section_id.nunique()==162 and cohort.organ.nunique()==17 and int(cohort.n_rows.sum())==329612,'FROZEN_COHORT_FAIL')
 require(cohort.groupby('dataset_id').size().to_dict()=={'HEST':75,'STimage':87},'DATASET_GROUP_FAIL')
 plan=json.loads((CAN/'result/stpath_c162_plan.json').read_text());require(set(plan)==set(cohort.section_id),'STPATH_PLAN_SECTION_FAIL')
 canonical=torch.load(CAN/'result/canonical162_row_ids.pt',map_location='cpu',weights_only=False).numpy().astype('int64');require(len(np.unique(canonical))==329612,'ROW_UNIQUENESS_FAIL')
 rcid=torch.load(ROWMAP/'row_to_gene_chunk_id.pt',map_location='cpu',weights_only=False).numpy();rlocal=torch.load(ROWMAP/'row_to_gene_row_index.pt',map_location='cpu',weights_only=False).numpy()
 rmspots=pd.read_csv(ROWMAP/'row_to_sid.tsv',sep='\t',usecols=['row_id','sid','spot_id']).set_index('row_id')
 chunks=pd.read_csv(MAN,sep='\t').set_index('gene_chunk_id');selection=pd.read_csv(DIAG/'analysis/section_hvg50_selection_primary.tsv',sep='\t')
 bench={}
 for ds in sorted(BENCH.iterdir()):
  if not ds.is_dir() or not (ds/'var_50genes.json').exists():continue
  genes=json.loads((ds/'var_50genes.json').read_text())['genes']
  for sp in sorted((ds/'splits').glob('test_*.csv')):
   tab=pd.read_csv(sp,dtype=str)
   col=next((c for c in ['sample_id','section_id','id'] if c in tab),None)
   if col is None:continue
   for sid in tab[col].dropna():
    if sid in bench:require(bench[sid][0]==ds.name,'HEST_BENCH_AMBIGUOUS_SECTION:'+sid)
    bench[sid]=(ds.name,genes,ds/'var_50genes.json')
 readg=giga_reader();covered=[];alignment=[];geneaudit=[];sels=[];comparison=[];input_sources={}
 for j,rec in enumerate(cohort.itertuples(index=False)):
  sid=rec.section_id;p=plan[sid];rr=np.array([int(z['global_row']) for z in p['rows']],dtype='int64')
  require(len(rr)==rec.n_rows and len(set(rr))==len(rr),'PLAN_ROWS_FAIL:'+sid)
  require(set(rr).issubset(set(canonical)),'ROW_OUTSIDE_FROZEN:'+sid)
  require(np.array_equal(np.array([int(z['local_row']) for z in p['rows']]),rlocal[rr]),'LOCAL_ROW_ALIGNMENT_FAIL:'+sid)
  cc=np.unique(rcid[rr]);require(len(cc)==1,'MULTIPLE_CHUNKS:'+sid)
  ch=torch.load(chunks.loc[int(cc[0]),'chunk_path'],map_location='cpu',weights_only=False)
  spots=[str(ch['spot_id'][i]) for i in rlocal[rr]]
  require([normspot(sid,z['spot']) for z in p['rows']]==[normspot(sid,z) for z in spots],'PLAN_SPOT_MISMATCH:'+sid)
  require(rmspots.loc[rr,'sid'].eq(sid).all() and [normspot(sid,z) for z in rmspots.loc[rr,'spot_id']]==[normspot(sid,z) for z in spots],'ROWMAP_SPOT_MISMATCH:'+sid)
  require(len(set(spots))==len(spots),'DUPLICATE_SPOT:'+sid)
  sf=selection.loc[selection.section_id.eq(sid)].sort_values('variance_rank');require(len(sf)==50 and sf.panel_definition.eq('pure_top50').all(),'FROZEN_HVG_FAIL:'+sid)
  origin='frozen_section_pure_top50';panel_source=DIAG/'analysis/section_hvg50_selection_primary.tsv'
  if sid in bench:
   _,syms,panel_source=bench[sid];lookup={s:int(g) for s,g in zip(v.gene_symbol.astype(str),v.global_gene_id)}
   require(all(s in lookup for s in syms),'HEST_GENE_MAPPING_FAIL:'+sid);gids=np.array([lookup[s] for s in syms],dtype='int64');origin='HEST_Bench_official_var50'
  else:gids=sf.global_gene_id.to_numpy('int64');syms=sf.gene.astype(str).tolist()
  require(len(gids)==len(set(gids))==50 and len(set(syms))==50,'GENE_PANEL_DUPLICATE:'+sid)
  require(v.gene_symbol.iloc[gids].astype(str).tolist()==syms,'SYMBOL_GLOBAL_ID_CONFLICT:'+sid)
  require(not any(s.startswith('ENSG') and s[4:].isdigit() for s in syms),'ENSG_PSEUDO_TARGET:'+sid)
  native_ids=agent.tokenizer.ge_tokenizer.symbol2id(syms)
  require(len(native_ids)==len(set(native_ids))==50 and min(native_ids)>=2,'STPATH_HVG_COVERAGE_OR_IDENTITY_FAIL:'+sid)
  globalids=ch['local_to_global_gene_idx'].numpy();obs=ch['observed_gene_mask_local'].numpy().astype(bool);cols=[]
  for g in gids:
   hit=np.flatnonzero(globalids==g);require(len(hit)==1 and obs[hit[0]],'MEASURED_GENE_MISSING_OR_AMBIGUOUS:'+sid+':'+str(g));cols.append(int(hit[0]))
  y=ch['target_genes_raw_local_csr'][rlocal[rr]][:,cols].toarray().astype('float32');require(np.isfinite(y).all(),'TRUTH_NONFINITE:'+sid)
  require(ch['target_raw_scale']=='raw_counts_log1p' and not bool(ch.get('targets_are_zscored',False)),'TRUTH_SCALE_FAIL:'+sid)
  feats,xy,feature_source=readg(sid,p)
  source_spots,source_xy,coord_source=original_coordinates(sid);xyby={normspot(sid,x):i for i,x in enumerate(source_spots)}
  require(len(xyby)==len(source_spots),'SOURCE_COORD_DUPLICATE:'+sid)
  require(all(normspot(sid,x) in xyby for x in spots),'COORD_SPOT_MISSING:'+sid)
  refxy=source_xy[np.array([xyby[normspot(sid,x)] for x in spots])]
  err=float(np.max(np.abs(xy_normalized(xy)-xy_normalized(refxy))))
  require(err<=1e-3,'SPATIAL_COORD_MISMATCH:'+sid+':'+str(err))
  for path in [Path(chunks.loc[int(cc[0]),'chunk_path']),feature_source,coord_source,panel_source]:
   if str(path) not in input_sources:input_sources[str(path)]=sha(path)
  np.savez_compressed(R/'02_cohort/prepared'/(sid+'.npz'),row_ids=rr,spot_ids=np.array(spots),gene_ids=gids,gene_symbols=np.array(syms),stpath_gene_ids=np.array(native_ids),coords=xy.astype('float32'))
  np.save(R/'02_cohort/truth'/(sid+'.npy'),y)
  covered.append(dict(section_id=sid,n_hvg=50,measured=50,spagenx=50,stpath=50,strict_hvg50=True,source=origin))
  comparison.append(dict(section_id=sid,patient_id='',patient_status=rec.patient_status,dataset_id=rec.dataset_id,organ=rec.organ,technology=rec.tech_id,n_spots=len(rr),measured_gene_count=int(obs.sum()),spagenx_available=True,stpath_available=True,included_in_comparison=True,exclusion_reason='',bootstrap_cluster='section:'+sid))
  alignment.append(dict(section_id=sid,rows=len(rr),spot_id_unique=True,spot_id_plan_rowmap_chunk_equal=True,normalized_coordinate_max_error=err,feature_source=str(feature_source),coordinate_source=str(coord_source),row_order_sha256=hashlib.sha256(rr.tobytes()).hexdigest()))
  for k,(g,s,tok) in enumerate(zip(gids,syms,native_ids)):
   geneaudit.append(dict(section_id=sid,rank=k+1,gene_symbol=s,spagenx_global_id=int(g),stpath_token_id=int(tok),measured_local_column=cols[k],unique_identity=True))
   sels.append(dict(section_id=sid,rank=k+1,gene_symbol=s,global_gene_id=int(g),source=origin,source_path=str(panel_source),source_sha256=input_sources[str(panel_source)]))
  print('PREFLIGHT',j+1,'/162',sid,'spots',len(rr),'coord_error',err,flush=True)
  del ch,feats,y
 table(R/'02_cohort/common_test_sections.tsv',comparison);table(R/'02_cohort/section_spot_alignment.tsv',alignment);table(R/'02_cohort/gene_mapping_audit.tsv',geneaudit);table(R/'02_cohort/hvg50_gene_lists.tsv',sels);table(R/'02_cohort/hvg50_selection_provenance.tsv',sels);table(R/'02_cohort/hvg50_coverage_audit.tsv',covered)
 split=pd.read_csv(D/'split/section_split_17organs_dedup.tsv',sep='\t');train=set(split.loc[split.split.eq('train'),'section_id']);test=set(cohort.section_id)
 require(not test.intersection(train),'SpaGenX_CURRENT_TRAIN_TEST_SECTION_OVERLAP')
 table(R/'01_audit/test_overlap_audit.tsv',[{'model':'SpaGenX','stage':'current frozen training split','status':'verified','overlap_sections':len(test&train)}, {'model':'SpaGenX','stage':'all historical initialization exposures','status':'unknown','overlap_sections':None},{'model':'STPath pretrained_v3','stage':'official initialization plus continuation patient-level exposures','status':'unknown','overlap_sections':None}])
 js(R/'01_audit/expression_scale_audit.json',{'truth':'corrected CSR raw_counts_log1p, measured zeros retained','library_size_normalization_added':False,'additional_log1p':False,'SpaGenX_output':'raw log1p','STPath_output':'log1p','output_transformation':'none'})
 hashes.update(input_sources);js(R/'01_audit/input_hashes.json',hashes)
 table(R/'01_audit/checkpoint_sha256.tsv',[{'model':'SpaGenX','checkpoint':str(SPAGENX_CHECKPOINT),'sha256':SPAGENX_SHA256},{'model':'STPath_pretrained_v3','checkpoint':str(STCK),'sha256':STSHA}])
 js(R/'01_audit/checkpoint_manifest.json',{'SpaGenX':{'path':str(SPAGENX_CHECKPOINT),'sha256':SPAGENX_SHA256,'strict_load':True,'checkpoint_step':42000,'forward_global_step':42000,'gene_bias':True,'disable_prompt':True,'empty_prompt':True,'encoder':'UNI2-h'},'STPath':{'path':str(STCK),'sha256':STSHA,'strict_load':True,'version':'adapted/continued pretrained_v3, not official release','vocab':str(STVOC),'context_args':None,'encoder':'Gigapath'},'sections':162,'spots':329612,'resource_collections':{'HEST':75,'STimage':87},'patient_identity':'unknown in frozen canonical manifest; section bootstrap; do not interpret section IDs as patient IDs'})
 js(R/'01_audit/preflight_status.json',{'status':'PASS','all_162_sections_aligned':True,'strict_hvg50_sections':162,'rows':329612,'no_model_forward':True,'prompt_forbidden':True,'pretraining_independence':'unknown; not proven external benchmark'})

if __name__=='__main__':
 try:main()
 except Exception as e:
  js(R/'01_audit/preflight_status.json',{'status':'BLOCKED','reason':str(e),'traceback':traceback.format_exc()});js(R/'05_report/final_status.json',{'status':'BLOCKED','stage':'Stage0-3','reason':str(e),'inference_started':False});raise
