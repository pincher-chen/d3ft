"""Read-only Slurm-step boundary inspection; no CUDA initialization."""
import datetime,json,subprocess
from pathlib import Path
r=Path(json.loads(Path('validation/results/response_design_20261021_internal_review/current.json').read_text())['root'])
code="""import os,json
rows=[]
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 try:
  args=open('/proc/'+pid+'/cmdline','rb').read().decode().split('\\x00')
  if len(args)>1 and 'python' in os.path.basename(args[0]) and any(args[1].endswith('/'+x) for x in ['m4_worker.py','e3_mg_fd_point.py','e6_k4_spectra.py']) and any('approved_20261021T144349' in a for a in args[1:]):
   env=dict(x.split('=',1) for x in open('/proc/'+pid+'/environ','rb').read().decode().split('\\x00') if '=' in x)
   rows.append({'pid':int(pid),'script':os.path.basename(args[1]),'CUDA_VISIBLE_DEVICES':env.get('CUDA_VISIBLE_DEVICES'),'SLURM_STEP_ID':env.get('SLURM_STEP_ID')})
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
print(json.dumps(rows))
"""
a=subprocess.run(['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','-N1','-n1','/usr/bin/python3','-c',code],capture_output=True,text=True,timeout=45,check=True);rows=json.loads(a.stdout);assert len(rows)==8 and sorted(int(q['CUDA_VISIBLE_DEVICES']) for q in rows)==list(range(8));obj={'time':datetime.datetime.now().astimezone().isoformat(),'passed':True,'rows':rows,'scope':'Actual Python /proc environ after Slurm env reset; introductory shell GPU line is before override and not authoritative'};(r/'live_binding_verified_r2.json').write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps(obj))
