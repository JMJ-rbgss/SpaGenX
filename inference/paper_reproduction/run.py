import sys,os,subprocess,time
from common import *

require(json.loads((R/'01_audit/preflight_status.json').read_text())['status']=='PASS','PRE_STAGE_NOT_PASS')
(R/'work_tmp').mkdir(exist_ok=True)
ps=[]
for name,gpu in [('SpaGenX','6'),('STPath','7')]:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,RUN_ROOT=str(R),TMPDIR=str(R/'work_tmp'),PYTHONPYCACHEPREFIX=str(R/'cache'))
 f=open(R/'logs'/('worker_'+name+'.log'),'w');p=subprocess.Popen([sys.executable,str(R/'code/worker.py'),'--model',name],env=env,stdout=f,stderr=subprocess.STDOUT);ps.append((name,p,f));print('START',name,'physical_GPU',gpu,'pid',p.pid,flush=True)
while True:
 failed=[n for n,p,f in ps if (R/'logs'/('failed_'+n+'.json')).exists() or (p.poll() is not None and p.returncode!=0)]
 if failed:
  js(R/'gate_abort.json',{'failed':failed});js(R/'05_report/final_status.json',{'status':'BLOCKED','stage':'smoke','failed_models':failed,'formal_started':False});raise SystemExit(1)
 if all((R/'logs'/('smoke_'+n+'.json')).exists() for n,p,f in ps):
  require(all(json.loads((R/'logs'/('smoke_'+n+'.json')).read_text())['status']=='PASS' for n,p,f in ps),'SMOKE_GATE_NOT_PASS');js(R/'gate_allow_full.json',{'status':'PASS_BOTH_SMOKE','prompt_forbidden':True});print('SMOKE GATE PASS; FULL RELEASED',flush=True);break
 time.sleep(1)
for n,p,f in ps:p.wait();f.close()
if any(p.returncode for n,p,f in ps):js(R/'05_report/final_status.json',{'status':'PARTIAL','stage':'inference','failed_models':[n for n,p,f in ps if p.returncode]});raise SystemExit(1)
subprocess.run([sys.executable,str(R/'code/evaluate.py')],check=True,env=os.environ)
