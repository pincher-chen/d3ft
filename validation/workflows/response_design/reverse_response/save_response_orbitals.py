"""Save missing band-edge orbitals for the one registered Gamma response point/FD pair."""
import argparse,json,hashlib,time,sys,warnings
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from scipy.sparse.linalg import LinearOperator,lobpcg
ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
runtime=Path('${D3FT_ROOT}/validation/results/response_design_20261002_resource_fill/comparison_runtime');sys.path.insert(0,str(runtime/'validation/workflows/phase3'));sys.path.insert(0,str(runtime/'validation/workflows/response_design'))
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from radial_quadrature import local_simpson_patch
from bounded_xc import memory_bounded_xc
from variational_ks import operator_factory
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((a.source/'point.json').read_text());m=json.loads((a.source/'metadata.json').read_text())
if not p['stationarity_pass'] or p['residual']>1e-6 or p['acoustic_sum_rule_Ha_Bohr']>5e-8:raise ValueError('Parent stationary/ASR gates failed')
a.output.mkdir(exist_ok=False)
meta={'source':str(a.source),'source_hashes':{n:sha(a.source/n) for n in ['point.json','metadata.json','density.npy','mask.npy','input.xyz']},'scope':'Occupied and two empty Gamma orbitals for later projected response; not a completed adjoint or new SCF point','physical_gpu':__import__('os').environ['CUDA_VISIBLE_DEVICES']}
(a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
cfg=ConfigDict(m['config']);cfg.epoch=3;cfg.crystal_file_path_path=str(a.source/'input.xyz');cfg.d3ft.parallel.k_pool=False
print('ORBITAL_STAGE reconstruct_geometry',flush=True)
with local_simpson_patch(),plain_fft_patch(k_pool=False),memory_bounded_xc():
 out=calc(cfg,freq_mask=np.load(a.source/'mask.npy'));out.density_grid=jnp.asarray(np.load(a.source/'density.npy'))
 if len(out.k_frac)!=1:raise ValueError('Only Gamma source allowed')
 charge=float(np.asarray(out.density_grid).sum(axis=0).mean()*abs(np.linalg.det(out.crystal.cell_vectors)))
 if abs(charge-96)>0.00096:raise ValueError('Density charge failure')
 op,dim=operator_factory(out)(0);action=jax.jit(op);batch=jax.jit(jax.vmap(op,in_axes=1,out_axes=1))
 matrix=LinearOperator((dim,dim),dtype=np.complex128,matvec=lambda x:np.asarray(action(jnp.asarray(x.reshape(-1)))),matmat=lambda x:np.asarray(batch(jnp.asarray(x))))
 diagonal=1/(1+np.asarray(op.kinetic_diagonal));pre=LinearOperator((dim,dim),dtype=np.complex128,matvec=lambda x:diagonal*x.reshape(-1),matmat=lambda x:diagonal[:,None]*x)
 rng=np.random.default_rng(0);start=np.linalg.qr(rng.normal(size=(dim,54))+1j*rng.normal(size=(dim,54)))[0];t=time.monotonic();print('ORBITAL_STAGE block_eigensolve',flush=True)
 with warnings.catch_warnings(record=True):values,vectors=lobpcg(matrix,start,M=pre,largest=False,tol=1e-10,maxiter=1000)
 order=np.argsort(values)[:50];values=values[order];vectors=vectors[:,order];res=np.linalg.norm(matrix.matmat(vectors)-vectors*values[None,:],axis=0);orth=float(np.max(abs(vectors.conj().T@vectors-np.eye(50))))
 if not np.isfinite(values).all() or not np.isfinite(res).all() or max(res)>1e-7 or orth>1e-8:raise ValueError('Eigenpair/orthogonality gate failed')
 delta=None  # Fresh checkpoint: eigenpairs independently certified by true residual and orthogonality above.
 np.save(a.output/'orbitals.npy',vectors)
 result={'eigenvalues_Ha':values.tolist(),'residuals_Ha':res.tolist(),'orthogonality_error':orth,'saved_orbitals_sha256':sha(a.output/'orbitals.npy'),'existing_spectrum_max_difference_Ha':delta,'seconds':time.monotonic()-t,'plane_waves':dim,'vbm_adjacent_separation_Ha':float(values[47]-values[46]),'cbm_adjacent_separation_Ha':float(values[49]-values[48]),'gap_eV':float((values[48]-values[47])*27.211386245988),'scope':meta['scope']}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print('ORBITAL_COMPLETED',json.dumps(result),flush=True)
