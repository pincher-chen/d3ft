"""CPU integration check: identical saved Si seed, full DFT two-update k2 runs."""
import argparse,json,pathlib,pickle,sys,time,gc
import numpy as np,jax
jax.config.update('jax_enable_x64',True)
from contextlib import nullcontext
from ml_collections import ConfigDict
from compact_k_stream import compact_k_stream_patch
P=pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0,str(P/'validation/workflows/phase3'));sys.path.insert(0,str(P/'validation/workflows/response_design'))
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from residual_report import fused_report
from d3ft.calc.forces import force_report
from radial_quadrature import local_simpson_patch
from gamma_response import dump
ap=argparse.ArgumentParser();ap.add_argument('--source',type=pathlib.Path,required=True);ap.add_argument('--read-only',action='store_true');ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
m=json.loads((a.source/'metadata.json').read_text());mask=np.load(a.source/'mask.npy')
with (a.source/'state.pkl').open('rb') as f:seed=pickle.load(f)
assert seed['pw'].shape[1]==1
q=np.repeat(seed['pw'],8,axis=1);answers=[]
for compact in [False,True]:
 c=ConfigDict(m['config']);c.k_grid_sizes=2;c.crystal_file_path_path=str(a.source/'input.xyz');c.d3ft.parallel.k_pool=False
 c.epoch=1 if a.read_only else 2;c.optimizer_args.learning_rate=0. if a.read_only else .001;c.d3ft.optim.lr_schedule='constant';c.d3ft.charge_gate=dict(enabled=True,absolute_tolerance=1e-6,report_path=str(a.output/('compact_charge.jsonl' if compact else 'dense_charge.jsonl')))
 with plain_fft_patch(),local_simpson_patch(),(compact_k_stream_patch(mask) if compact else nullcontext()):
  out=calc(c,init_params={'pw':q},freq_mask=mask);rep=fused_report(out);f=force_report(out)
  from d3ft.calc.implicit import _kwargs_from
  from d3ft.calc.stress import free_energy_at_strain,ewald_translation_indices
  from gamma_response import validated_radial_splines
  kw=_kwargs_from(out);params=kw.pop('params_pw');occ=kw.pop('params_occ');kw['ewald_indices']=ewald_translation_indices(out.crystal.cell_vectors,out.config.ewald_args['ewald_cutoff'])
  with validated_radial_splines():
   full_gradient=jax.jit(jax.grad(lambda p:free_energy_at_strain(__import__('jax.numpy',fromlist=['zeros']).zeros((3,3)),p,occ,**kw)))(params)
  row=dict(electronic_gradient=np.asarray(full_gradient),energy=float(out.total_energy),history=np.asarray(out.energy_history),pw=np.asarray(out.params_pw),density=np.asarray(out.density_grid),stress=np.asarray(rep['stress']),residual=rep['variational_residual'],forces=np.asarray(f['forces_Ha_Bohr']))
  answers.append(row);dump(a.output/('compact.json' if compact else 'dense.json'),{k:float(v) if np.ndim(v)==0 else np.asarray(v).tolist() for k,v in row.items() if k not in ['pw','density','electronic_gradient']})
 del out;jax.clear_caches();gc.collect()
checks={}
for name in answers[0]:
 delta=float(np.max(np.abs(answers[1][name]-answers[0][name])));checks[name]=dict(max_abs_difference=delta,passed=bool(np.allclose(answers[1][name],answers[0][name],atol=2e-11,rtol=2e-10)))
result=dict(passed=all(v['passed'] for v in checks.values()),checks=checks,source=str(a.source),read_only=a.read_only,scope='Same saved Si k2 orbitals; full DFT energy/electronic gradient/force/stress/density/charge equivalence at fixed orbitals' if a.read_only else 'Two true Si k2 optimizer updates; raw orbitals can amplify reduction roundoff through Adam normalization')
dump(a.output/'acceptance.json',result)
if not result['passed']:raise ValueError('Compact DFT integration mismatch')
print('COMPACT_CALC_PASS '+json.dumps(checks),flush=True)
