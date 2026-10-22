"""One local path-edge refinement from real base spectra; no full-BZ claim."""
import concurrent.futures,copy,datetime,hashlib,json,subprocess
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[3];O=P/'validation/results/response_design_20261005_band_structure'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def stage(name,**kw):
 status=read(O/'status.json');status.update(stage=name,time=datetime.datetime.now().isoformat(),**kw);dump(O/'status.json',status)
def main():
 if (O/'local_refinement_audit.json').exists():
  assert read(O/'local_refinement_audit.json')['passed'];return
 base=read(O/'bands_complete.json');protocol=read(O/'path_protocol.json');oldn=len(protocol['k_fractional']);edges=set()
 for s in base.values():
  assert s['full_path'] and len(s['points'])==oldn
  eigen=np.array([r['eigenvalues_Ha'] for r in s['points']]);extrema={int(np.argmax(eigen[:,47])),int(np.argmin(eigen[:,48]))}
  for seg in protocol['segments']:
   for pos,i in enumerate(seg['indices']):
    if i in extrema:
     for j in [pos-1,pos+1]:
      if 0<=j<len(seg['indices']):edges.add(tuple(sorted([i,seg['indices'][j]])))
 refined=copy.deepcopy(protocol);frac=refined['k_fractional'];lookup={tuple(np.round(f,12)):i for i,f in enumerate(frac)};edgeids={}
 for i,j in sorted(edges):
  ids=[]
  for t in [.25,.5,.75]:
   f=(1-t)*np.array(frac[i])+t*np.array(frac[j]);key=tuple(np.round(f,12))
   if key not in lookup:lookup[key]=len(frac);frac.append(f.tolist())
   ids.append((t,lookup[key]))
  edgeids[(i,j)]=ids
 for seg in refined['segments']:
  ind,x=seg['indices'],seg['x'];outind=[ind[0]];outx=[x[0]]
  for pos,(i,j) in enumerate(zip(ind[:-1],ind[1:])):
   key=tuple(sorted([i,j]));rows=edgeids.get(key,[])
   ts=sorted((t if i<j else 1-t,index) for t,index in rows)
   for t,index in ts:outind.append(index);outx.append((1-t)*x[pos]+t*x[pos+1])
   outind.append(j);outx.append(x[pos+1])
  seg['indices'],seg['x']=outind,outx
 refined['local_refinement']=dict(original_unique_points=oldn,inserted_unique_points=len(frac)-oldn,method='quarter/midpoint/three-quarter of neighboring base intervals at each sampled VBM and CBM; union shared by all states',base_protocol_sha256=sha(O/'path_protocol.json'),base_spectrum_sha256=sha(O/'bands_complete.json'),scope='One local path check, not a continuum or full-BZ convergence certificate')
 dump(O/'path_protocol.refined.json',refined);dump(O/'bands_base_complete.json',base)
 shards=read(O/'all_shards.json');dump(O/'base_shards.json',shards)
 stage('edge_refinement_running',original_path_points=oldn,refined_path_points=len(frac))
 tasks=[]
 for state in base:
  parents=sorted((O).glob('path_'+state+'*_command.json'));assert len(parents)==4
  for g,parent in enumerate(parents):
   indices=list(range(oldn,len(frac)))[g::4]
   if indices:tasks.append((state,indices,parent))
 def execute(task):
  state,indices,parent=task;cmd=read(parent);name='refine_'+parent.name.removesuffix('_command.json');out=O/'shards'/name;oldtmp=next(x[7:] for x in cmd if x.startswith('TMPDIR='));tmp='/dev/shm/d3ft_response_design_20261005_bands_'+name
  cmd=[x.replace(oldtmp,tmp) for x in cmd];cmd[-4:]=[str(O/'path_protocol.refined.json'),state,str(out),json.dumps(indices)]
  dump(O/(name+'_command.json'),cmd)
  with (O/(name+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(f'{name}: RC{r.returncode}')
  return dict(state=state,indices=indices,path=str(out))
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:extra=list(pool.map(execute,tasks))
 audit=dict(passed=True,refinement=refined['local_refinement'],states={})
 merged=copy.deepcopy(base)
 for name,s in merged.items():
  rows=[read(Path(t['path'])/f'k_{i:04d}.json') for t in extra if t['state']==name for i in t['indices']];s['points']+=rows;s['points'].sort(key=lambda r:r['index']);assert [r['index'] for r in s['points']]==list(range(len(frac)))
  s['max_eigen_residual_Ha']=max(max(r['residuals_Ha']) for r in s['points']);assert s['max_eigen_residual_Ha']<=1e-7
  olde=np.array([r['eigenvalues_Ha'] for r in base[name]['points']]);newe=np.array([r['eigenvalues_Ha'] for r in s['points']]);oldgap=(olde[:,48].min()-olde[:,47].max())*27.211386245988;newgap=(newe[:,48].min()-newe[:,47].max())*27.211386245988
  audit['states'][name]=dict(base_gap_eV=float(oldgap),refined_gap_eV=float(newgap),gap_difference_meV=float((newgap-oldgap)*1000),max_residual_Ha=s['max_eigen_residual_Ha'])
  s['local_path_refinement']=audit['states'][name]
 dump(O/'all_shards.json',shards+extra);dump(O/'bands_complete.json',merged);dump(O/'local_refinement_audit.json',audit);stage('edge_refinement_complete',refined_path_points=len(frac),local_refinement_results=audit['states'])
if __name__=='__main__':main()
