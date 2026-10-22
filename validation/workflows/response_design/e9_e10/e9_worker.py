"""Resumable E9 GPU worker. New evaluations only; old E4 data define the model."""
import argparse,os,time
from pathlib import Path
import numpy as np
from common import read,dump,sha,unique
from relaxed_design import run

def spent(root):
 b=read(root/'GPU_budget.json');return sum(a['GPU_hours'] if a.get('GPU_hours') is not None else max(0,time.time()-a['start_epoch'])/3600 for a in b['attempts'])

class Store:
 def __init__(self,root,backend):
  self.root=root;self.backend=backend;self.kind='dft';self.callsfile=root/'evaluations.json';self.calls=read(self.callsfile) if self.callsfile.exists() else {}
 def geometry(self,theta):return self.backend.geometry(theta)
 def spent(self):return spent(self.root)
 def evaluate(self,theta,label,parent,spectrum=True):
  label=Path(label);label.mkdir(parents=True,exist_ok=True);key=str(label.relative_to(self.root));entry=self.calls.setdefault(key,dict(theta=np.asarray(theta).tolist(),spectrum=bool(spectrum),attempts=[]))
  if not np.allclose(entry['theta'],theta,rtol=0,atol=1e-13) or entry['spectrum']!=spectrum:raise ValueError('Changed logical E9 evaluation on resume')
  resume=[]
  for folder in entry['attempts']:
   folder=Path(folder)
   if (folder/'readout.json').exists():return read(folder/'readout.json')
   for latest in folder.glob('checkpoints/*/latest.json'):
    state=read(latest);model=read(latest.parent/'model.json');done=int(model['config'].get('d3ft',{}).get('checkpoint_resume',{}).get('completed_steps',0))+state['iteration']
    if state.get('complete') and sha(latest.parent/state['state_file'])==state['state_sha256']:resume.append((done,latest.stat().st_mtime,latest.parent))
  folder=label/('attempt_'+str(len(entry['attempts'])))
  entry['attempts'].append(str(folder));dump(self.callsfile,self.calls)
  checkpoint=max(resume,key=lambda x:x[:2])[2] if resume else None
  if len(entry['attempts'])>1 and checkpoint is None:raise RuntimeError('Incomplete evaluation without validated checkpoint; no silent cold restart')
  dump(self.root/'live_evaluation.json',dict(time=__import__('datetime').datetime.now().astimezone().isoformat(),label=key,attempt=str(folder),spectrum=spectrum,resume_checkpoint=None if checkpoint is None else str(checkpoint)))
  return self.backend.evaluate(theta,folder,parent,spectrum=spectrum,resume=checkpoint)
 def gradient(self,theta,current,label):
  label=Path(label);label.mkdir(parents=True,exist_ok=True)
  for response in label.glob('attempt_*/response.json'):
   r=read(response)
   if r['adjoint_record']['converged']:return np.array(r['gradient_eV'])
  folder=label/('attempt_'+str(len(list(label.glob('attempt_*')))))
  g=self.backend.gradient(theta,current,folder)
  if not read(folder/'response.json')['adjoint_record']['converged']:raise ValueError('E9 adjoint did not converge')
  return g
 def first_step_report(self,metrics):
  metrics['physical_evaluation_attempts_for_first_step']=sum(len(v['attempts']) for k,v in self.calls.items() if k.startswith('design/outer_0_'))
  metrics.update(time=__import__('datetime').datetime.now().astimezone().isoformat(),measured_GPU_hours_not_estimate=True)
  remaining=metrics['remaining_max_updates'];cost=metrics['step_GPU_hours']
  metrics['remaining_ETA_hours_range']=[.7*remaining*cost,2*remaining*cost+cost]
  dump(self.root/'first_design_step_measurement.json',metrics)
  c=read(self.root/'registered_protocol.json');message=('E9首个设计步实测 '+str(metrics)+'；ETA按首步外推，未放宽预注册力/电子/谱门。')
  for f in [Path(c['project'])/'paper/response_design_EXPERIMENTS_20261014.md',Path(c['project'])/'response_design_to_do_list.md']:
   with f.open('a') as stream:stream.write('\n- '+metrics['time']+' '+message+'\n')
  print(message,flush=True)

def main(root):
 from e9_backend import E9Backend
 p=read(root/'backend_protocol.json');backend=Store(root,E9Backend(p));registered=read(root/'registered_protocol.json')
 result=run(root/'design',p['theta0'],backend,registered['E9']);dump(root/'acceptance.json',result)
 print('E9_COMPLETE '+str(dict(passed=result['passed'],accepted_updates=result['accepted_updates'],gap=result['final_gap_eV'],max_force=result['max_force_norm_Ha_Bohr'])),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);main(a.parse_args().root)
