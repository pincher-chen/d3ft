"""One warmed serial reverse/forward comparison on the same physical GPU.

Queued only after the four correctness workers pass. Memory is reported as the
combined-process high-water mark; it is not a separately reset per-mode peak.
"""
import argparse,json,datetime,time,os
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from gamma_response import GammaResponse,validated_radial_splines,HA_EV,dump
from bounded_xc import memory_bounded_xc
from projected_solver import real_inner

ap=argparse.ArgumentParser();ap.add_argument('protocol',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
cfg=json.loads(a.protocol.read_text());a.output.mkdir(exist_ok=False)
def stage(label,**kw):
 d=dict(stage=label,time=datetime.datetime.now().isoformat(),pid=os.getpid(),**kw)
 dump(a.output/'status.json',d);print('COST_STAGE',json.dumps(d),flush=True)
try:
 import sys
 sys.path.insert(0,str(Path(cfg['runtime'])/'validation/workflows/response_design'))
 from radial_quadrature import local_simpson_patch
 from d3ft.calc.ground_state import calc
 from d3ft.parallel import plain_fft_patch
 source=Path(cfg['source']);orb=Path(cfg['orbitals'])
 c=ConfigDict(json.loads((source/'metadata.json').read_text())['config'])
 c.epoch=3;c.crystal_file_path_path=str(source/'input.xyz');c.d3ft.parallel.k_pool=False
 with local_simpson_patch(),plain_fft_patch(k_pool=False):
  stage('reconstruct_geometry');out=calc(c,freq_mask=np.load(source/'mask.npy'))
  with validated_radial_splines(),memory_bounded_xc():
   summary=json.loads((orb/'summary.json').read_text());r=GammaResponse(out,np.load(orb/'orbitals.npy'),summary['eigenvalues_Ha'],cfg['theta0'])
   stage('warmup_compile');warm=time.monotonic();zero=jnp.zeros(6)
   b=r.project(r.apply_local(r.q,r.kernel(r.edge_density)))
   jax.block_until_ready(r.hessian(b));jax.block_until_ready(r.precondition(b));jax.block_until_ready(r.external(jnp.eye(6)[0]))
   bare=r.bare(zero);jax.block_until_ready(bare)
   jax.block_until_ready(r.reverse_gradient(b));compile_seconds=time.monotonic()-warm
   stage('warmed_reverse');start=time.monotonic()
   lam,adj=r.solve(b,lambda row:stage('reverse_iteration',**row))
   reverse=r.reverse_gradient(lam)*HA_EV;jax.block_until_ready(reverse);rt=time.monotonic()-start
   stage('warmed_forward');start=time.monotonic();forward=[];records=[]
   for axis in range(6):
    external=r.external(jnp.eye(6)[axis]);u,record=r.solve(-r.project(external),lambda row:stage('forward_iteration',axis=axis,**row))
    forward.append(float((bare[axis]+4*real_inner(b,u))*HA_EV));records.append(record)
   ft=time.monotonic()-start
   reference=np.asarray(json.loads(Path(cfg['native_gradient']).read_text())['gap_gradient_eV'])
   reverse=np.asarray(reverse);forward=np.asarray(forward)
   passed=bool(np.all(abs(reverse-reference)<=.01*abs(reference)+1e-4) and np.all(abs(reverse-forward)<=1e-5+1e-4*abs(reverse)))
   memory=jax.devices()[0].memory_stats()
   dump(a.output/'result.json',dict(passed=passed,reverse_seconds=rt,full_six_direction_forward_seconds=ft,
      first_use_compile_seconds=compile_seconds,reverse_gradient_eV=reverse.tolist(),forward_gradient_eV=forward.tolist(),
      response_records={'adjoint':adj,'tangents':records},hardware=str(jax.devices()[0]),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
      device_memory_stats=memory,memory_scope='Combined-process high-water, includes prior compilation; not per-mode reset peak',
      scope='One warmed, synchronized, same-program/same-physical-GPU complete-gradient instance; no statistical or universal speedup claim'))
   stage('completed',passed=passed)
   if not passed:raise SystemExit(1)
except Exception as exc:stage('failed',error=str(exc));raise
