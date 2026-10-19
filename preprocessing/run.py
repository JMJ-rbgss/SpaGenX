"""Explicit CLI dispatch; historical in-place migrations are not auto-executable."""
from pathlib import Path
import argparse,json,os,sys,subprocess,shlex
STAGES={'discover':'step0_sample_discovery.py','features':'step1_extract_uni2h_features.py','merge':'step1_merge_features.py','expression':'step2_gene_expression_preprocessing.py','rowmap':'step3_build_rowmap_threaded.py','neighbors':'step4_build_m1_m2_overlays.py','canonical':'fix_m2_gene_identity_assets.py','hvg384':'rebuild_active_row_hvg384.py'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 c=json.loads(a.config.read_text());args=c['stages'][a.stage];assert '--overwrite' not in args,'OVERWRITE_FORBIDDEN'
 outputs=[]
 for flag in ['--output_dir','--output-dir','--output_root','--out','--output_pt','--output_tsv','--manifest_tsv']:
  if flag in args:outputs.append(Path(args[args.index(flag)+1]))
 assert outputs,'EXPLICIT_OUTPUT_FLAG_REQUIRED'
 for out in outputs:assert not out.exists(),'OUTPUT_ALREADY_EXISTS'
 env=os.environ.copy();env['SPAGENX_DATA_ROOT']=str(Path(c['data_root']).resolve());env['CUDA_VISIBLE_DEVICES']=','.join(map(str,c.get('gpus',[4])))
 cmd=[sys.executable,str(Path(__file__).with_name(STAGES[a.stage]))]+args;print(shlex.join(cmd))
 if a.execute:subprocess.run(cmd,env=env,check=True)
 else:print('DRY_RUN: no preprocessing performed')
if __name__=='__main__':main()
