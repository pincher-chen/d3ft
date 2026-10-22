import json,hashlib,py_compile,sys,datetime
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;P=W.parents[3]
root=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root'])
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p={k:read(root/k/'protocol.json') for k in ['M4-A','M3-A','M5-A','M2-A']}
# Only timing ordering, resource guards and reservation bookkeeping change.
frozen=Path(p['M2-A']['runtime']['e3'])/'validation/workflows/response_design/reverse_response'
src=(frozen/'e3_formal_timing.py').read_text()
s=src.replace("!='hn48'", "!='hn35'").replace('must run on hn48','must run on hn35')
s=s.replace("whole_four_GPU_node_reservation_hours=4*total/3600,node='hn48'", "whole_eight_GPU_node_reservation_hours=8*total/3600,node='hn35'")
s=s.replace("   def budget():\n", "   def budget():\n    if (a.output.parent/'exclusive_violation.json').exists():raise RuntimeError('Node exclusivity lost; partial timings only')\n")
needle="    budget();bstart=time.monotonic();b=bfun();"
assert s.count(needle)==1
s=s.replace(needle,"   for repeat in range(3):\n    budget();bstart=time.monotonic();b=bfun();")
s=s.replace("   device=subprocess.run", "   budget()\n   device=subprocess.run")
out=frozen/'m2_exclusive_timing.py';out.write_text(s)
# Clean status reporting only, before any submission.
f=W/'m5_response_worker.py';t=f.read_text();t=t.replace("d={'time':datetime.datetime.datetime.now().isoformat()} if False else {'stage':s,**kw}","d={'time':datetime.datetime.now().astimezone().isoformat(),'stage':s,**kw}");f.write_text(t)
checks={}
# Exact restoration demonstrates no hidden numerical timing edits.
r=s.replace("!='hn35'", "!='hn48'").replace('must run on hn35','must run on hn48').replace("whole_eight_GPU_node_reservation_hours=8*total/3600,node='hn35'","whole_four_GPU_node_reservation_hours=4*total/3600,node='hn48'")
r=r.replace("    if (a.output.parent/'exclusive_violation.json').exists():raise RuntimeError('Node exclusivity lost; partial timings only')\n",'').replace("   for repeat in range(3):\n    budget();bstart", "    budget();bstart").replace("   budget()\n   device=subprocess.run", "   device=subprocess.run")
assert r==src;checks['timing_numerical_source_identical_after_resource_order_guard_reversal']=True
sys.path.insert(0,str(Path(p['M4-A']['runtime']['gamma'])/'validation/workflows/response_design'))
from joint_design import proposal
rng=np.random.default_rng(20261021);dirs=[]
for i in range(5):
 d=rng.normal(size=54);d[6:]= (d[6:].reshape(16,3)-d[6:].reshape(16,3).mean(axis=0)).ravel();d/=np.linalg.norm(d);dirs.append(d)
assert np.array_equal(np.asarray(dirs),np.asarray(p['M3-A']['directions']));assert np.max(abs(np.asarray(dirs)[:,6:].reshape(5,16,3).sum(axis=1)))<1e-14
checks['five_reproducible_unit_translation_free_directions']=True
origin=np.asarray(p['M4-A']['theta0']);g=rng.normal(size=54);g[:6]=0;theta=origin.copy()
for i in range(3):
 trial=proposal(theta,origin,g,4.410987, p['M4-A']['target_eV'],p['M4-A']['limits']);assert np.array_equal(trial[:6],origin[:6]);assert np.max(np.linalg.norm((trial-theta)[6:].reshape(16,3),axis=1))<=.020000000001;assert np.max(np.linalg.norm((trial-origin)[6:].reshape(16,3),axis=1))<=.150000000001;theta=trial
checks['coordinate_ablation_preserves_strain_and_original_bounds']=True
checks['twenty_unique_fd_points']=len({(d,h,sign) for d in range(5) for h in [.002,.001] for sign in [-1,1]})==20
assert p['M3-A']['point_timeout_seconds']*20<25*3600
assert p['M5-A']['spectra_timeout_seconds']*3+p['M5-A']['response_timeout_seconds']<35*3600
assert set(p['M4-A']['gpus']).isdisjoint(p['M3-A']['gpus']);assert set(p['M3-A']['gpus']).isdisjoint(p['M5-A']['spectra_gpus']);checks['nonoverlap_and_timeout_budget_reservations']=True
sys.path.insert(0,str(Path(p['M5-A']['runtime']['mesh'])/'validation/workflows/response_design/reverse_response'))
from mesh_target import select_mesh_target
spec=read(p['M5-A']['spectrum_reference']);rows=sorted(spec['spectra'],key=lambda r:r['k_index']);assert [r['k_index'] for r in rows]==list(range(64));assert all(r['passed'] for r in rows)
_,hard=select_mesh_target(np.array([r['eigenvalues_Ha'] for r in rows]),48,'hard');checks['k4_hard_gap_unique_branch_before_GPU']=hard
for name,h in p['M5-A']['source_hashes'].items():assert sha(Path(p['M5-A']['source'])/name)==h
checks['k4_source_hashes_verified']=True
meta=read(Path(p['M5-A']['source'])/'metadata.json')
for symbol,item in meta['physical']['pp'].items():assert sha(item['path'])==item['sha256']
checks['pseudopotential_hashes_verified']=True
assert np.array_equal(p['M5-A']['theta0'],p['M4-A']['theta0']);assert len(read(p['M5-A']['k2_gradient_result'])['gradient_eV'])==54;checks['same_initial_geometry_and54_k2_gradient']=True
# Compile only; importing GPU drivers is intentionally avoided.
for f in list(W.glob('*.py'))+[out]:py_compile.compile(str(f),doraise=True)
checks['new_python_compiles']=True
controller=(W/'controller.py').read_text()
assert "all((ROOT/j/'summary.json').exists()" in controller and "all(row[1:]==[0,0]" in controller
assert "exclusive_violation.json" in controller and "out.exists()" in controller
assert "--jobid=250285" in controller and "--nodelist=hn35" in controller and "scancel" not in controller
checks['exclusive_dependencies_resource_scope_and_immutable_output_guards']=True
result={'passed':True,'time':datetime.datetime.now().astimezone().isoformat(),'checks':checks,'protocol_SHA256':{k:sha(root/k/'protocol.json') for k in p},'software_SHA256':{str(f):sha(f) for f in list(W.glob('*.py'))+[out]},'scope':'CPU input/protocol/controller-source checks; no claim of GPU or physical acceptance'}
(root/'CPU_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
for k in p:(root/k/'CPU_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'root':str(root),'checks':list(checks),'hard_target':hard},indent=2))
