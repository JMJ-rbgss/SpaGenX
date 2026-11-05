import traceback
from common import *

def main():
 for model in ['SpaGenX','STPath']:require(json.loads((R/'logs'/('complete_'+model+'.json')).read_text())['status']=='PASS','INCOMPLETE_INFERENCE:'+model)
 manifest=pd.read_csv(R/'02_cohort/common_test_sections.tsv',sep='\t');gene_rows=[];section_rows=[];paired_rows=[]
 for rec in manifest.itertuples(index=False):
  sid=rec.section_id;z=np.load(R/'02_cohort/prepared'/(sid+'.npz'));y=np.load(R/'02_cohort/truth'/(sid+'.npy'));pred={}
  for model in ['SpaGenX','STPath']:
   a=np.load(R/'predictions'/model/(sid+'.npz'));require(np.array_equal(a['row_ids'],z['row_ids']) and np.array_equal(a['spot_ids'],z['spot_ids']) and np.array_equal(a['gene_ids'],z['gene_ids']),'PREDICTION_ALIGNMENT_FAIL:'+model+':'+sid)
   require(not bool(a['prompt_enabled']),'PROMPT_ENABLED');pred[model]=a['pred']
  rs={m:[] for m in pred}
  for j,(g,sym) in enumerate(zip(z['gene_ids'],z['gene_symbols'])):
   for model in pred:
    r,reason=pcc(pred[model][:,j],y[:,j]);rs[model].append(r)
    gene_rows.append(dict(section_id=sid,dataset_id=rec.dataset_id,organ=rec.organ,gene_symbol=str(sym),global_gene_id=int(g),model=model,n_spots=len(y),PCC=r,invalid_reason=reason,MAE=float(np.mean(np.abs(pred[model][:,j]-y[:,j])))))
  joint=np.isfinite(rs['SpaGenX'])&np.isfinite(rs['STPath']);sens={m:float(np.mean(np.asarray(rs[m])[joint])) if joint.any() else np.nan for m in rs}
  for model in rs:
   r=np.asarray(rs[model]);section_rows.append(dict(section_id=sid,dataset_id=rec.dataset_id,organ=rec.organ,technology=rec.technology,model=model,n_spots=len(y),n_hvg=50,valid_gene_count=int(np.isfinite(r).sum()),excluded_gene_count=int((~np.isfinite(r)).sum()),strict_HVG50_PCC=float(np.mean(r)),paired_valid_PCC=sens[model],paired_valid_gene_count=int(joint.sum()),MAE=float(np.mean(np.abs(pred[model]-y))),prompt_enabled=False))
  paired_rows.append(dict(section_id=sid,dataset_id=rec.dataset_id,organ=rec.organ,n_spots=len(y),SpaGenX_PCC=float(np.mean(rs['SpaGenX'])),STPath_PCC=float(np.mean(rs['STPath'])),delta_PCC=float(np.mean(rs['SpaGenX'])-np.mean(rs['STPath'])),paired_valid_SpaGenX_PCC=sens['SpaGenX'],paired_valid_STPath_PCC=sens['STPath'],paired_valid_delta_PCC=sens['SpaGenX']-sens['STPath'],n_paired_valid_genes=int(joint.sum())))
 genes=pd.DataFrame(gene_rows);sections=pd.DataFrame(section_rows);paired=pd.DataFrame(paired_rows)
 table(R/'03_metrics/section_gene_pcc.tsv',genes);table(R/'03_metrics/section_hvg50_pcc.tsv',sections);table(R/'03_metrics/paired_delta_pcc.tsv',paired)
 dataset=[];organ=[];overall=[]
 for model in ['SpaGenX','STPath']:
  mm=sections.loc[sections.model.eq(model)]
  for ds,g in mm.groupby('dataset_id',sort=True):dataset.append(dict(Dataset=ds,model=model,N_sections=len(g),strict_PCC=float(np.mean(g.strict_HVG50_PCC.to_numpy())),paired_valid_PCC=float(np.mean(g.paired_valid_PCC.to_numpy())),n_strict_defined=int(g.strict_HVG50_PCC.notna().sum())))
  for o,g in mm.groupby('organ',sort=True):organ.append(dict(organ=o,model=model,N_sections=len(g),strict_PCC=float(np.mean(g.strict_HVG50_PCC.to_numpy())),paired_valid_PCC=float(np.mean(g.paired_valid_PCC.to_numpy()))))
  dm=[r['strict_PCC'] for r in dataset if r['model']==model];sv=[r['paired_valid_PCC'] for r in dataset if r['model']==model]
  overall.extend([dict(model=model,metric='resource_collection_macro_PCC',strict_value=float(np.mean(dm)),paired_valid_sensitivity_value=float(np.mean(sv)),N_sections=len(mm),N_collections=len(dm)),dict(model=model,metric='section_weighted_PCC',strict_value=float(np.mean(mm.strict_HVG50_PCC.to_numpy())),paired_valid_sensitivity_value=float(np.mean(mm.paired_valid_PCC.to_numpy())),N_sections=len(mm),N_collections=len(dm))])
 table(R/'03_metrics/dataset_mean_pcc.tsv',dataset);table(R/'03_metrics/organ_mean_pcc.tsv',organ);table(R/'03_metrics/overall_metrics.tsv',overall)
 rng=np.random.default_rng(42);boots=[];groups={d:g for d,g in paired.groupby('dataset_id',sort=True)}
 for b in range(100):
  draw=[]
  for ds,g in groups.items():
   draw.append(g.iloc[rng.integers(0,len(g),len(g))])
  dsmeans=[];dssens=[]
  for ds,g in zip(groups,draw):
   vals=g[['SpaGenX_PCC','STPath_PCC']].to_numpy();vv=np.mean(vals,axis=0);ss=np.mean(g[['paired_valid_SpaGenX_PCC','paired_valid_STPath_PCC']].to_numpy(),axis=0);dsmeans.append(vv);dssens.append(ss)
   boots.append(dict(replicate=b,scope='collection:'+ds,SpaGenX_PCC=vv[0],STPath_PCC=vv[1],delta=vv[0]-vv[1],sensitivity_SpaGenX=ss[0],sensitivity_STPath=ss[1],sensitivity_delta=ss[0]-ss[1]))
  for scope,arrays,sarrays in [('resource_collection_macro',np.array(dsmeans),np.array(dssens)),('section_weighted',pd.concat(draw)[['SpaGenX_PCC','STPath_PCC']].to_numpy(),pd.concat(draw)[['paired_valid_SpaGenX_PCC','paired_valid_STPath_PCC']].to_numpy())]:
   vv=np.mean(arrays,axis=0);ss=np.mean(sarrays,axis=0);boots.append(dict(replicate=b,scope=scope,SpaGenX_PCC=vv[0],STPath_PCC=vv[1],delta=vv[0]-vv[1],sensitivity_SpaGenX=ss[0],sensitivity_STPath=ss[1],sensitivity_delta=ss[0]-ss[1]))
 br=pd.DataFrame(boots);table(R/'03_metrics/bootstrap_replicates.tsv',br);cis=[]
 for scope,g in br.groupby('scope',sort=True):
  for col in ['SpaGenX_PCC','STPath_PCC','delta','sensitivity_SpaGenX','sensitivity_STPath','sensitivity_delta']:
   x=g[col].to_numpy();finite=np.isfinite(x);q=np.quantile(x,[.025,.975]) if finite.all() else [np.nan,np.nan]
   cis.append(dict(scope=scope,metric=col,CI_low=q[0],CI_high=q[1],replicates=100,defined_replicates=int(finite.sum()),seed=42,unit='paired sections stratified by resource collection; patient identity unknown',CI_method='percentile'))
 table(R/'03_metrics/bootstrap_ci.tsv',cis)
 final=[]
 for ds,g in groups.items():
  ci=next(r for r in cis if r['scope']=='collection:'+ds and r['metric']=='delta');final.append(dict(Dataset=ds,N_sections=len(g),SpaGenX_PCC=float(np.mean(g.SpaGenX_PCC.to_numpy())),STPath_PCC=float(np.mean(g.STPath_PCC.to_numpy())),Delta_PCC=float(np.mean(g.delta_PCC.to_numpy())),CI_low=ci['CI_low'],CI_high=ci['CI_high']))
 table(R/'03_metrics/final_dataset_summary.tsv',final)
 n_defined=int(paired[['SpaGenX_PCC','STPath_PCC']].notna().all(axis=1).sum());valid=paired.loc[paired[['SpaGenX_PCC','STPath_PCC']].notna().all(axis=1)]
 result={'status':'METRICS_READY' if n_defined==162 else 'PARTIAL','sections':162,'spots':329612,'organs':17,'strict_defined_sections':n_defined,'resource_collections':2,'runtime_SpaGenX_global_step':42000,'gene_bias_enabled':True,'prompt_enabled_both':False,'section_win_rate':float(np.mean(valid.delta_PCC>0)),'section_win_rate_denominator':len(valid),'paired_mean_delta':float(np.mean(paired.delta_PCC.to_numpy())),'paired_median_delta':float(np.median(paired.delta_PCC.to_numpy())),'collection_win_rate':float(np.mean(np.array([r['Delta_PCC'] for r in final])>0)),'formal_external_benchmark_eligible':False,'eligibility_reason':'STPath pretrained_v3 adapted/continued; historical pretraining independence and patient identities not established; resource collections are not original study datasets.','overall':overall,'dataset_summary':final,'weights_updated':False,'figures_pending':True}
 js(R/'05_report/final_status.json',result)
 text='# SpaGenX vs STPath pretrained_v3\n\n```text\n'+pd.DataFrame(final).to_string(index=False)+'\n```\n\n```text\n'+pd.DataFrame(overall).to_string(index=False)+'\n```\n\nBoth models used native image encoders and zero real-expression prompts. SpaGenX runtime step42000 enables the trained gene bias; old step0 predictions were not substituted. PCC was calculated per gene within each section and then averaged over the frozen section-specific 50 genes. HEST/STimage are resource collections, not the original HEST-Bench ten cohorts. The pretrained_v3 checkpoint is not the official release. Patient-level and historical pretraining independence remain unverified.\n'
 (R/'05_report/final_comparison_report.md').write_text(text)
 (R/'05_report/manuscript_ready_methods.txt').write_text('A frozen SpaGenX-current checkpoint at training step 42000 and the frozen adapted STPath pretrained_v3 checkpoint were compared on 162 fixed test sections (329612 spots). No model was trained or fine-tuned during evaluation. SpaGenX used its native UNI2-h image features and neighbor overlay, runtime step42000 with learned gene bias enabled and expression prompts explicitly disabled. STPath used native Gigapath features, coordinate normalization, organ/technology tokens and all-expression-mask inputs. The same pre-frozen section-specific 50-gene panel and corrected raw-log1p measured expression, including measured zeros, were used for both methods. Gene-wise Pearson correlations were calculated over spots within each section, followed by a strict arithmetic mean over 50 genes. Collection-level means were calculated separately for HEST and STimage; their equal-weight mean is a resource-collection macro and is reported separately from the equal-weight section average. Paired percentile bootstrap intervals used 100 section resamples stratified by resource collection with seed42; patient identities were not established. Invalid correlations were retained in strict results, with a separately labeled paired-valid sensitivity analysis. This comparison is not an official STPath-release or proven external zero-shot HEST-Bench replication.\n')
 (R/'05_report/manuscript_ready_results.txt').write_text('Results for this adapted-checkpoint comparison:\n'+pd.DataFrame(final).to_string(index=False)+'\n\n'+pd.DataFrame(overall).to_string(index=False)+'\n\nStrict defined paired sections: '+str(n_defined)+'/162. No external-generalization or causality claim is justified by this comparison.\n')
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:js(R/'05_report/final_status.json',{'status':'BLOCKED','stage':'metrics','reason':str(e),'traceback':traceback.format_exc()});raise
