"""Owned Si k2 spectrum: serial versus eight CPU-device real-DFT response."""
import argparse,json,pickle,sys,datetime
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import sha,validated_radial_splines,HA_EV
from kmesh_response import KMeshResponse
from sharded_kmesh_response import ShardedKMeshResponse
from bounded_xc import memory_bounded_xc
P=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(P/'validation/workflows/response_design'));sys.path.insert(0,str(P/'validation/workflows/phase3'))
from radial_quadrature import local_simpson_patch
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--devices',type=int,default=8,choices=[2,4,8]);a=ap.parse_args();a.output.mkdir(exist_ok=False)
if jax.default_backend()!='cpu' or len(jax.devices())!=a.devices:raise ValueError('Requested CPU device count required')
E=P/'validation/results/response_design_20261015_e5_kmesh';source=E/'si_k2_base';orb=E/'si_k2_spectrum'
m=json.loads((source/'metadata.json').read_text());om=json.loads((orb/'metadata.json').read_text());s=json.loads((orb/'summary.json').read_text());protocol=json.loads((E/'protocol.json').read_text())
for n,h in om['source_hashes'].items():
 if sha(source/n)!=h:raise ValueError('Changed source '+n)
if sha(orb/'orbitals.npy')!=s['saved_orbitals_sha256']:raise ValueError('Changed spectrum')
with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
c=ConfigDict(m['config']);c.epoch=1;c.d3ft.optim.lr_schedule='constant';c.optimizer_args.learning_rate=0.;c.d3ft.parallel.k_pool=False;c.crystal_file_path_path=str(source/'input.xyz')
errors={}
def check(name,x,y,tol):
 error=float(np.max(abs(np.asarray(x)-np.asarray(y))));errors[name]=error
 (a.output/'partial.json').write_text(json.dumps(dict(stage=name,errors=errors),indent=2)+'\n')
 if error>tol:raise ValueError(name+' mismatch '+str(error))
with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc(),validated_radial_splines():
 out=calc(c,init_params=state,freq_mask=np.load(source/'mask.npy'))
 args=(out,np.load(orb/'orbitals.npy'),np.asarray(s['eigenvalues_Ha']),protocol['theta0'])
 r=KMeshResponse(*args,include_coordinates=True);d=ShardedKMeshResponse(*args,include_coordinates=True)
 check('physical_density',r.rho,d.rho,1e-12);check('geometry_diagonal',r.diagonal,d.diagonal,1e-12)
 check('KB_projectors',r.projectors,d.projectors,1e-11);check('common_HXC',r.veff,d.veff,1e-11)
 b=r.project(r.apply_local(r.q,r.kernel(r.edge_density)));db=d.project(d.apply_local(d.q,d.kernel(d.edge_density)))
 check('adjoint_rhs',b,db,1e-10)
 probe=r.project(b/(1+r.diagonal[:,:,None]));check('coupled_Hessian',r.hessian(probe),d.hessian(d.put_k(np.asarray(probe))),1e-10)
 lam,rec=r.solve(b,None);dl,dc=d.solve(db,None);check('adjoint_solution',lam,dl,1e-8)
 check('full_12_reverse_gradient_eV',r.reverse_gradient(lam)*HA_EV,d.reverse_gradient(dl)*HA_EV,5e-6)
 for axis in [0,6]:check('geometry_JVP_axis'+str(axis),r.external(jnp.eye(12)[axis]),d.external(jnp.eye(12)[axis]),1e-9)
 result=dict(time=datetime.datetime.now().isoformat(),passed=True,backend='cpu',devices=a.devices,nk=8,errors=errors,
             serial_record=rec,sharded_record=dc,source=str(source),source_hashes=om['source_hashes'],
             scope='Owned real Si PBE k2 CPU correctness only; no Mg/GPU memory or performance claim')
 (a.output/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
 print('E5 真实Si k2八CPU分片：PASS。\n密度/KB/公共HXC/伴随及完整12梯度与串行一致。\nMgSiN2 GPU物理/显存验收仍待进行。',flush=True)
