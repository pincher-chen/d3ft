#!/usr/bin/env python3
"""CPU tests from a clean checkout; original module tests are retained."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--suite',choices=['smoke','cpu'],default='smoke');p.add_argument('--report',type=Path,default=ROOT/'runs/test_report.json');a=p.parse_args()
smoke=['test_00_import.py','test_01_stiefel.py','test_02_sinkhorn.py','test_12_occupation_charge.py','test_13_kmesh_symmetry.py','test_18_riemannian_gradient.py','test_26_masked_projectors.py','test_paper_coordinates.py']
extra=['test_03_lanczos.py','test_05_lanczos_grad.py','test_11_band_gap_differentiable.py','test_14_elastic.py','test_15_fit.py','test_16_implicit.py','test_17_uncertainty.py','test_19_channel_scaling.py','test_20_xc_tunable.py','test_21_density_grid.py','test_22_relaxation_guard.py','test_23_acoustic_sum_guard.py','test_24_force_amplification.py','test_25_tetragonal_strain.py','test_27_phase3_guards.py','test_28_equilibrium_volume_response.py','test_29_phase3_trial_basis.py','test_29_kpool_projectors.py','test_30_xc_density_refinement.py','test_31_phase3_force_budget.py','test_31_kpool_density.py','test_32_phase3_continuation.py']
env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'third_party/jrystal_dependency',ROOT/'reproducibility/implementation'])))
records=[];a.report.parent.mkdir(parents=True,exist_ok=True)
for name in smoke+(extra if a.suite=='cpu' else []):
 test_env=env.copy();source_scope='current NLCC-repaired paper solver'
 if name=='test_12_occupation_charge.py':
  source_scope='archived development Sinkhorn implementation (not the FD paper solver)'
  test_env['PYTHONPATH']=os.pathsep.join(map(str,[ROOT/'archives/development_20261013',ROOT/'third_party/jrystal_dependency',ROOT/'reproducibility/implementation']))
 if name in ['test_31_kpool_density.py','test_29_kpool_projectors.py']:
  test_env['XLA_FLAGS']='--xla_force_host_platform_device_count=2'
 start=time.monotonic();r=subprocess.run([sys.executable,str(ROOT/'tests'/name)],cwd=ROOT,env=test_env,capture_output=True,text=True)
 log=a.report.parent/(name+'.txt');log.write_text(r.stdout+r.stderr)
 records.append(dict(test=name,source_scope=source_scope,passed=r.returncode==0,returncode=r.returncode,seconds=round(time.monotonic()-start,3),log=str(log)))
 print(('PASS ' if r.returncode==0 else 'FAIL ')+name,flush=True)
 a.report.write_text(json.dumps(dict(suite=a.suite,python=sys.version,tests=records,passed=all(x['passed'] for x in records)),indent=2)+'\n')
raise SystemExit(0 if all(x['passed'] for x in records) else 1)
