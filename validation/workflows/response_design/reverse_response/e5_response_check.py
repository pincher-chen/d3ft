"""Small-system coupled response: all modes, cross-k coupling, Gamma limit."""
import argparse,json,pickle,sys,time,datetime
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import GammaResponse,sha,dump,validated_radial_splines,HA_EV
from kmesh_response import KMeshResponse
from projected_solver import real_inner
from bounded_xc import memory_bounded_xc
from memory_probe import MemoryProbe
P=Path(__file__).resolve().parents[4];sys.path.insert(0,str(P/'validation/workflows/response_design'))
from radial_quadrature import local_simpson_patch
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--source',required=True,type=Path)
ap.add_argument('--orbitals',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--gamma-equivalence',action='store_true');a=ap.parse_args()
cfg=json.loads(a.protocol.read_text());a.output.mkdir(exist_ok=False);sampler=MemoryProbe(a.output).start()
def stage(name,**kw):
 d=dict(time=datetime.datetime.now().isoformat(),stage=name,**kw);dump(a.output/'status.json',d);print('E5_RESPONSE',json.dumps(d),flush=True)
try:
 m=json.loads((a.source/'metadata.json').read_text());s=json.loads((a.orbitals/'summary.json').read_text());om=json.loads((a.orbitals/'metadata.json').read_text())
 for name,h in om['source_hashes'].items():
  if sha(a.source/name)!=h:raise ValueError('Source changed '+name)
 if sha(a.orbitals/'orbitals.npy')!=s['saved_orbitals_sha256']:raise ValueError('Spectrum changed')
 with (a.source/'state.pkl').open('rb') as f:state=pickle.load(f)
 c=ConfigDict(m['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.d3ft.parallel.k_pool=False;c.crystal_file_path_path=str(a.source/'input.xyz')
 with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc():
  stage('geometry_reconstruction');out=calc(c,init_params=state,freq_mask=np.load(a.source/'mask.npy'))
  vals=np.asarray(s['eigenvalues_Ha']);vectors=np.load(a.orbitals/'orbitals.npy')
  if a.gamma_equivalence:vals=vals[None,:];vectors=vectors[None,...]
  with validated_radial_splines():
   stage('coupled_response_construction');r=KMeshResponse(out,vectors,vals,cfg['theta0'],include_coordinates=True,objective='hard' if a.gamma_equivalence else cfg['objective'],tau_Ha=cfg.get('soft_tau_Ha'))
   dump(a.output/'target.json',r.target_info)
   charge=float(r.rho.mean()*r.volume);res=float(jnp.max(jnp.linalg.norm(r.ham(r.q)-r.q*r.values[:,None,:],axis=1)))
   linear=float(r.explicit_gap(r.zero));expected=r.target_info['linearized_HF_value_Ha']
   if abs(charge-2*r.n_occupied)>1e-8 or res>1e-7 or abs(linear-expected)*HA_EV>3e-6:raise ValueError('Coupled rebuilt primal mismatch')
   stage('primal_gate_PASS',charge=charge,residual_Ha=res,nk=r.nk)
   b=r.project(r.apply_local(r.q,r.kernel(r.edge_density)));jax.block_until_ready(b)
   x=b/jnp.linalg.norm(b);y=r.project(b/(1+r.diagonal[:,:,None]));y/=jnp.linalg.norm(y)
   lhs=float(real_inner(x,r.hessian(y)));rhs=float(real_inner(r.hessian(x),y));sym=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
   if sym>1e-8:raise ValueError('Coupled Hessian symmetry failed')
   lam,record=r.solve(b,lambda row:stage('adjoint_iteration',**row));gradient=r.reverse_gradient(lam)*HA_EV;jax.block_until_ready(gradient)
   bare=r.bare(r.zero);forward=[]
   for axis in range(r.n_variables):
    d=jnp.eye(r.n_variables)[axis];u,rec=r.solve(-r.project(r.external(d)),lambda row:stage('tangent_iteration',axis=axis,**row))
    value=float((bare[axis]+4*r.w*real_inner(b,u))*HA_EV);forward.append(value)
    dump(a.output/'partial_results.json',dict(reverse=np.asarray(gradient).tolist(),forward=forward,completed_axes=list(range(axis+1))))
   forward=np.asarray(forward);gradient=np.asarray(gradient);difference=abs(gradient-forward)
   checks=dict(full_mode=bool(np.all(difference<=5e-6+1e-6*np.maximum(abs(gradient),abs(forward)))),
    translation=bool(max(abs(gradient[6:].reshape(2,3).sum(axis=0)))<=1e-6),projected_adjoint=bool(float(jnp.max(abs(jnp.einsum('kga,kgb->kab',jnp.conj(r.q),lam))))<1e-9))
   extra={}
   if a.gamma_equivalence:
    g=GammaResponse(out,vectors[0],vals[0],cfg['theta0'],include_coordinates=True)
    bg=g.project(g.apply_local(g.q,g.kernel(g.edge_density)));lg,rg=g.solve(bg,lambda row:stage('Gamma_adjoint',**row))
    gg=np.asarray(g.reverse_gradient(lg))*HA_EV
    rng=np.random.default_rng(20261015);z=g.project(jnp.asarray(rng.normal(size=g.q.shape)+1j*rng.normal(size=g.q.shape)))
    hdiff=float(jnp.max(abs(g.hessian(z)-r.hessian(z[None,...])[0])))
    d=jnp.arange(12,dtype=float)/12;ediff=float(jnp.max(abs(g.external(d)-r.external(d)[0])))
    checks['Gamma_full_gradient']=bool(np.all(abs(gradient-gg)<=5e-6));checks['Gamma_Hessian_and_external']=max(hdiff,ediff)<=1e-10
    extra=dict(Gamma_gradient=gg.tolist(),Gamma_gradient_max_error=float(max(abs(gradient-gg))),Gamma_Hessian_max_error=hdiff,Gamma_external_max_error=ediff)
   else:
    rng=np.random.default_rng(20261015);probe=jnp.zeros_like(r.q).at[0].set(jnp.asarray(rng.normal(size=r.q[0].shape)+1j*rng.normal(size=r.q[0].shape)));probe=r.project(probe)
    cross=float(jnp.linalg.norm(r.hessian(probe)[1:]));checks['nonzero_public_density_cross_k_coupling']=cross>1e-10;extra['cross_k_action_norm']=cross
   result=dict(time=datetime.datetime.now().isoformat(),passed=all(checks.values()),checks=checks,gradient_eV=gradient.tolist(),forward_eV=forward.tolist(),
    maximum_mode_error=float(max(difference)),primal_residual_Ha=res,target=r.target_info,kernel_symmetry_relative_error=sym,adjoint_record=record,**extra,
    scope='Single-GPU coupled prototype; FD/full production acceptance separate')
   dump(a.output/'result.json',result)
   if not result['passed']:raise ValueError('E5 response/Gamma gates failed')
   stage('completed',passed=True)
except Exception as exc:stage('failed',error=str(exc));raise
finally:sampler.finish()
