"""One actual N-dimensional Gamma graph, three fresh tangent/adjoint timings.

No memory sampler, file I/O or iteration callback inside measured intervals.
Forward explicit term is directional JVP, not a precomputed reverse gradient.
Warmups/compilation and allocated step walltime are reported separately.
"""
import argparse,json,pickle,sys,time,os,datetime,subprocess
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import GammaResponse,sha,dump,validated_radial_splines,HA_EV
from bounded_xc import memory_bounded_xc
from projected_solver import real_inner
from memory_probe import MemoryProbe
P=Path(__file__).resolve().parents[4];sys.path.insert(0,str(P/'validation/workflows/response_design'))
from radial_quadrature import local_simpson_patch
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--N',type=int,choices=[1,6,24,54],required=True)
ap.add_argument('--output',type=Path,required=True);ap.add_argument('--budget-hours',type=float,required=True);a=ap.parse_args();start=time.monotonic();a.output.mkdir(exist_ok=False)
warm_memory_sampler=MemoryProbe(a.output).start()
def stage(name,**kw):
 d=dict(stage=name,time=datetime.datetime.now().isoformat(),jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),N=a.N,**kw);dump(a.output/'status.json',d);print('E3_TIMING',json.dumps(d),flush=True)
try:
 if not json.loads((a.root/'formal_timing_20261015/active_geometry_checks.json').read_text())['passed']:
  raise ValueError('Actual N input-map/Jacobian CPU gate failed')
 for name in ['si_acceptance_refined.json','mg_mode_acceptance.json','mg_fd_acceptance.json']:
  if not json.loads((a.root/name).read_text())['passed']:raise ValueError('Derivative gate failed '+name)
 cfg=json.loads((a.root/'mg_response_protocol.json').read_text());source=Path(cfg['source']);orb=Path(cfg['orbitals']);m=json.loads((source/'metadata.json').read_text());s=json.loads((orb/'summary.json').read_text());om=json.loads((orb/'metadata.json').read_text())
 for n,h in om['source_hashes'].items():
  if sha(source/n)!=h:raise ValueError('Source changed '+n)
 if sha(source/'state.pkl')!=m['state_sha256'] or sha(orb/'orbitals.npy')!=s['saved_orbitals_sha256']:raise ValueError('Owned state/orbitals changed')
 if os.environ.get('SLURMD_NODENAME')!='hn48':raise ValueError('Formal timing must run on hn48')
 with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
 c=ConfigDict(m['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.d3ft.parallel.k_pool=False;c.crystal_file_path_path=str(source/'input.xyz')
 with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc():
  out=calc(c,init_params=state,freq_mask=np.load(source/'mask.npy'))
  with validated_radial_splines():
   axes=list(range(a.N));r=GammaResponse(out,np.load(orb/'orbitals.npy'),np.asarray(s['eigenvalues_Ha']),cfg['theta0'],include_coordinates=True,active_axes=axes)
   if r.zero.shape!=(a.N,) or r.n_variables!=a.N:raise ValueError('Not a genuine N input graph')
   res=float(jnp.max(jnp.linalg.norm(r.ham(r.q)-r.q*r.values[None,:],axis=0)))
   if res>1e-7:raise ValueError('Formal primal does not match certified state')
   bfun=jax.jit(lambda:r.project(r.apply_local(r.q,r.kernel(r.edge_density))))
   directional=jax.jit(lambda d:jax.jvp(r.explicit_gap,(r.zero,),(d,))[1])
   stage('warmup',input_shape=list(r.zero.shape),axes=axes)
   b=bfun();jax.block_until_ready(b);warm_lam,warm_rec=r.solve(b,None)
   grad=np.asarray(r.reverse_gradient(warm_lam))*HA_EV
   ref=np.asarray(json.loads((a.root/'mg_mode_acceptance.json').read_text())['reverse_gradient_eV'])[axes]
   if np.max(abs(grad-ref))>5e-6:raise ValueError('N embedding gradient differs from certified full54')
   unit=jnp.eye(a.N)[0];jax.block_until_ready(directional(unit));warm_u,warm_fw=r.solve(-r.project(r.external(unit)),None)
   warm_memory_sampler.finish();warm_memory_sampler=None
   warm_seconds=time.monotonic()-start;rows=[];fw_rows=[]
   def budget():
    if time.monotonic()-start>a.budget_hours*3600:raise RuntimeError('Formal active-GPU budget exceeded; incomplete repeats retained')
   for repeat in range(3):
    budget();t=time.monotonic();b=bfun();jax.block_until_ready(b);lam,record=r.solve(b,None);g=r.reverse_gradient(lam)*HA_EV;jax.block_until_ready(g);elapsed=time.monotonic()-t
    error=float(np.max(abs(np.asarray(g)-ref)))
    if error>5e-6:raise ValueError('Measured reverse gradient mismatch')
    rows.append(dict(N=a.N,repeat=repeat,mode='reverse',seconds=elapsed,response_solves=1,gradient_eV=np.asarray(g).tolist(),gradient_max_error=error,response_record=record))
    dump(a.output/'reverse_raw.json',rows);stage('reverse_repeat_complete',repeat=repeat,seconds=elapsed)
    budget();bstart=time.monotonic();b=bfun();jax.block_until_ready(b);rhs_seconds=time.monotonic()-bstart;sum_seconds=rhs_seconds;derivatives=[];records=[]
    for axis in axes:
     budget();direction=jnp.eye(a.N)[axis];t=time.monotonic();explicit=directional(direction);external=r.external(direction);u,rec=r.solve(-r.project(external),None)
     value=(explicit+4*real_inner(b,u))*HA_EV;jax.block_until_ready(value);elapsed=time.monotonic()-t;sum_seconds+=elapsed;value=float(value)
     if abs(value-ref[axis])>5e-6+1e-6*abs(ref[axis]):raise ValueError('Measured tangent mismatch')
     derivatives.append(value);records.append(dict(axis=axis,seconds=elapsed,response_record=rec))
     dump(a.output/f'forward_repeat_{repeat}_partial.json',dict(rhs_seconds=rhs_seconds,axes=records,gradient_eV=derivatives));stage('forward_axis_complete',repeat=repeat,axis=axis,seconds=elapsed)
    fw_rows.append(dict(N=a.N,repeat=repeat,mode='forward',seconds=sum_seconds,response_solves=a.N,rhs_seconds=rhs_seconds,axes=records,gradient_eV=derivatives))
    dump(a.output/'forward_raw.json',fw_rows)
   device=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout
   total=time.monotonic()-start;dump(a.output/'result.json',dict(passed=True,N=a.N,axes=axes,input_shape=list(r.zero.shape),repeats=3,
    reverse=rows,forward=fw_rows,warmup_and_compilation_seconds=warm_seconds,total_allocated_GPU_step_seconds=total,
    active_GPU_hours=total/3600,whole_four_GPU_node_reservation_hours=4*total/3600,node='hn48',device_inventory=device,
    primal_residual_Ha=res,variable_names=r.layout.names,gradient_units=r.layout.gradient_units,warmup_memory=__import__('json').loads((a.output/'memory_peak.json').read_text()),software_sha256={n:sha(Path(__file__).with_name(n)) for n in ['gamma_response.py','geometry_layout.py','projected_solver.py',Path(__file__).name]},
    scope='One state and one physical GPU; raw repetitions, not general statistical evidence. N=54 contains three translation-redundant coordinates. Forward timings summed outside file I/O.'))
   stage('completed',passed=True)
except Exception as exc:stage('failed',error=str(exc));raise
finally:
 if warm_memory_sampler is not None:warm_memory_sampler.finish()
