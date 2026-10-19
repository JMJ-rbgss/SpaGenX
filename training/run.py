"""Configuration-based, dry-run-by-default launcher for the unchanged native trainer."""
from pathlib import Path
import argparse,json,os,sys,subprocess,shlex
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 c=json.loads(a.config.read_text());args=c['native_args'];assert '--output_root' in args
 out=Path(args[args.index('--output_root')+1]);assert not out.exists(),'OUTPUT_ALREADY_EXISTS'
 assert '--train_ablation' not in args and '--disable_homo' not in args and '--disable_hetero' not in args
 env=os.environ.copy();env['SPAGENX_DATA_ROOT']=str(Path(c['data_root']).resolve());env['CUDA_VISIBLE_DEVICES']=','.join(map(str,c['gpus']))
 for flag in ['--rowmap_root','--v3_split_root','--feature_store','--gene_chunk_manifest','--neighbor_overlay','--program_overlay_path','--section_hvg_path']:
  assert flag in args,flag
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node='+str(len(c['gpus'])),str(Path(__file__).with_name('train_spagenx.py'))]+args
 print(shlex.join(cmd))
 if not a.execute:print('DRY_RUN: no training or directory creation');return
 for flag in ['--rowmap_root','--v3_split_root','--feature_store','--gene_chunk_manifest','--neighbor_overlay','--program_overlay_path','--section_hvg_path','--resume_checkpoint']:
  if flag in args:assert Path(args[args.index(flag)+1]).exists(),'MISSING_ASSET:'+flag
 subprocess.run(cmd,env=env,check=True)
if __name__=='__main__':main()
