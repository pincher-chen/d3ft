"""Independent degeneracy/soft trace algebra, separate from electronic FD."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
from mesh_target import select_mesh_target
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();checks={}
v=np.array([[-2.,-1.,.2,.2]])
try:select_mesh_target(v,2,'hard');checks['hard_rejects_band_degeneracy']=False
except ValueError:checks['hard_rejects_band_degeneracy']=True
try:select_mesh_target(np.array([[-2.,-1.,.2,.4],[-2.,-1.,.2,.4]]),2,'hard');checks['hard_rejects_k_degeneracy']=False
except ValueError:checks['hard_rejects_k_degeneracy']=True
w,info=select_mesh_target(v,2,'soft',.001)
checks['soft_weights_normalized']=abs(w.sum())<1e-14 and abs(w[0,2:].sum()-1)<1e-14
checks['equal_degenerate_weights']=w[0,2]==w[0,3]
rng=np.random.default_rng(20261015);u=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
pert=np.array([[.3,.2+.1j],[.2-.1j,-.1]])
checks['degenerate_trace_basis_invariant']=abs(np.trace(pert).real*.5-np.trace(u.conj().T@pert@u).real*.5)<1e-14
checks['soft_bias_bound']=0<=info['hard_gap_Ha']-info['target_Ha']<=info['softening_bias_bound_Ha']+1e-14
checks={k:bool(v) for k,v in checks.items()}
r=dict(time=datetime.datetime.now().isoformat(),passed=all(checks.values()),checks=checks,
 scope='Target/trace algebra only; soft objective full electronic FD not certified')
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if not r['passed']:raise RuntimeError('Degeneracy policy algebra failed')
