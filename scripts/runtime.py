"""Hash-verified version launcher; no global environment or source mutation."""
from pathlib import Path
import hashlib,json,os,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
DEP=ROOT/'third_party/jrystal_dependency'

def source_path(name):
 if name=='current':return ROOT,ROOT/'reproducibility/versions/current_manifest.json'
 if name not in ['paper_frozen','ph_prefix','ph_prefix_nlcc']:raise ValueError(name)
 path=ROOT/'reproducibility/versions'/name
 return path,path/'manifest.json'

def verify(name):
 path,mp=source_path(name);m=json.loads(mp.read_text())
 for f,h in m['files'].items():
  if hashlib.sha256((path/f).read_bytes()).hexdigest()!=h:raise RuntimeError('Solver source hash changed: '+f)
 return path,m

def run(name,driver,args,run_root=None):
 source,m=verify(name)
 env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join(map(str,[source,DEP,ROOT/'reproducibility/scripts',ROOT/'reproducibility/implementation'])),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',JCTC_FROZEN_SOURCE='d3ft_fd_raw_v2_roundoff' if name.startswith('ph_prefix') else 'd3ft_fd_raw_v2_stable_exact001',JCTC_CANDIDATE_TREE_SHA256=m['tree_sha256'])
 manifest=json.loads((ROOT/'provenance/baseline_manifest.json').read_text())
 env['JCTC_FROZEN_JRYSTAL_ARCHIVE_SHA256']=manifest['sources']['jrystal_d3ft_dependency']['archive_sha256']
 if run_root:env['D3FT_RUN_ROOT']=str(run_root)
 return subprocess.run([sys.executable,'-u',str(driver),*map(str,args)],env=env,check=True)

if __name__=='__main__':
 if len(sys.argv)<3:raise SystemExit('Usage: python scripts/runtime.py VERSION DRIVER.py [arguments]')
 run(sys.argv[1],sys.argv[2],sys.argv[3:])
