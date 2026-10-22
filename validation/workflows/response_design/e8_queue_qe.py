"""Fresh CPU-only QE SCF then NSCF; no GPU resource consumed."""
from pathlib import Path
import datetime,hashlib,json,subprocess,time,xml.etree.ElementTree as ET
import numpy as np
import argparse
P=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=P/'validation/results/response_design_20261015_e8_bands/inputs');args=ap.parse_args();A=args.root.resolve()
def dump(name,value):
 p=A/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def now():return datetime.datetime.now().isoformat()
try:
 for kind in ['scf','nscf']:
  cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=16','bash',str(A/'run_stage.sh'),str(A/(kind+'.in'))]
  rec=dict(start=now(),jobid=250285,gpus=[],GPU_hours=0,command=cmd,input_sha256=hashlib.sha256((A/(kind+'.in')).read_bytes()).hexdigest(),status='running')
  dump('status.json',dict(time=now(),stage=kind+'_running',GPU_hours=0))
  with (A/(kind+'.log')).open('xb') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(kind+'_launch.json',rec);code=proc.wait()
  rec.update(end=now(),returncode=code);dump(kind+'_launch.json',rec)
  if code:raise RuntimeError(kind+' failed; no next stage submitted')
  xml=A/'scratch/MgSiN2_e8.save/data-file-schema.xml';content=xml.read_bytes();(A/(kind+'_data-file-schema.xml')).write_bytes(content)
  out=ET.fromstring(content).find('output');convergence=out.findtext('convergence_info/'+('scf_conv' if kind=='scf' else 'nscf_conv')+'/convergence_achieved')
  if convergence is not None and convergence.strip().lower()=='false':raise RuntimeError('QE unconverged '+kind)
  if 'JOB DONE.' not in (A/(kind+'.log')).read_text():raise RuntimeError('QE completion marker absent')
 if kind=='nscf':
  path=json.loads((A/'path.json').read_text());nodes=out.findall('band_structure/ks_energies');st=out.find('atomic_structure');alat=float(st.attrib['alat'])
  cell=np.array([np.fromstring(st.findtext('cell/'+s),sep=' ') for s in ['a1','a2','a3']]);b=2*np.pi*np.linalg.inv(cell).T
  frac=np.array([np.fromstring(n.findtext('k_point'),sep=' ')*2*np.pi/alat@np.linalg.inv(b) for n in nodes]);values=np.array([np.fromstring(n.findtext('eigenvalues'),sep=' ') for n in nodes])
  if frac.shape!=np.shape(path['k_fractional']) or np.max(abs(frac-np.array(path['k_fractional'])))>1e-8:raise RuntimeError('QE path ordering mismatch')
  if values.shape[1]!=56 or abs(float(out.findtext('band_structure/nelec'))-96)>1e-8:raise RuntimeError('QE bands/charge mismatch')
  np.savez_compressed(A/'fresh_QE_bands.npz',energies_Ha=values,k_fractional=frac,cell_bohr=cell)
 dump('status.json',dict(time=now(),stage='QE_path_complete',passed=True,GPU_hours=0,points=len(nodes),comparison_with_D3FT_complete=False))
 for doc in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
  with doc.open('a') as f:f.write('\n- '+now()+' E8新MgSiN2 CPU QE SCF/路径NSCF完成；同UPF/截止输入独立生成，D3FT新路径比较尚待GPU资源。\n')
 print('E8 新QE SCF/NSCF路径完成。\n完整精度XML与本征值已保存，GPU机时0。\nD3FT新路径比较尚待验收。',flush=True)
except Exception as exc:
 dump('status.json',dict(time=now(),stage='failed',passed=False,error=str(exc),GPU_hours=0));raise
