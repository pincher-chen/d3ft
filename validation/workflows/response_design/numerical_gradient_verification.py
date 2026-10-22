"""One preregistered MgSiN2 XC-quadrature gradient refinement campaign.

Uses immutable production runtime/checkpoints; terminal failure is evidence,
never silently admitted. GPU jobs survive the interactive session.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

P = Path('${D3FT_ROOT}')
R = P / 'validation/results/response_design_20261005_numerical_verification/gradient_v3'
OLD = P / 'validation/results/response_design_20261004_reverse_response_v5'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def matrix(v):
    return [[v[0], v[5]/2, v[4]/2], [v[5]/2, v[1], v[3]/2],
            [v[4]/2, v[3]/2, v[2]]]


def prepare():
    R.mkdir(exist_ok=False)
    for name in ['runs', 'orbitals', 'response', 'logs', 'commands', 'drivers']:
        (R/name).mkdir()
    shutil.copy2(__file__, R/'gradient_controller.py')
    for f in OLD.glob('*.py'):
        shutil.copy2(f, R/'drivers'/f.name)
    save = P/'validation/workflows/response_design/reverse_response/save_response_orbitals.py'
    shutil.copy2(save, R/'drivers'/save.name)
    shutil.copy2(P/'validation/workflows/response_design/reverse_response/bounded_sigma_fft.py',R/'drivers/bounded_sigma_fft.py')
    (R/'drivers/sigma_runner.py').write_text('''import runpy,sys
from pathlib import Path
from bounded_sigma_fft import bounded_sigma_fft
driver=sys.argv.pop(1)
sys.path.insert(0,str(Path(driver).parent))
sys.argv[0]=driver
with bounded_sigma_fft():
    runpy.run_path(driver,run_name='__main__')
''')
    (R/'drivers/bounded_generic.py').write_text('''import runpy,sys
import gc,jax
from pathlib import Path
from bounded_xc import memory_bounded_xc
driver=sys.argv.pop(1)
sys.path.insert(0,str(Path(driver).parent))
sys.path.insert(0,str(Path(driver).parents[1]/'phase3'))
# Ground-state construction has already validated these radial spline inputs.
# Avoid the eager duplicate XC validation and release the update executable
# before compiling the identical stress/stationarity derivatives.
import residual_report
source=Path(residual_report.__file__).read_text()
old="    if not sharded_report:\\n        jax.block_until_ready(free_energy_at_strain(\\n            jnp.zeros((3,3)),params_pw=pw,params_occ=occ,**kw))\\n"
if source.count(old)!=1:raise RuntimeError('Frozen report anchor changed')
source=source.replace(old,"    import gc\\n    jax.clear_caches()\\n    gc.collect()\\n",1)
exec(compile(source,residual_report.__file__,'exec'),residual_report.__dict__)
from d3ft.calc import forces
original_force_report=forces.force_report
def force_report(*args,**kwargs):
    jax.clear_caches();gc.collect()
    return original_force_report(*args,**kwargs)
forces.force_report=force_report
sys.argv[0]=driver
with memory_bounded_xc():
    runpy.run_path(driver,run_name='__main__')
''')
    (R/'private_tmp.sh').write_text('''#!/bin/bash
set -eu
case "${TMPDIR:-}" in /dev/shm/d3ft_response_design_numcheck_*) ;; *) exit 90;; esac
[ ! -e "$TMPDIR" ] || exit 91
mkdir -m 700 -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
ulimit -c 0
"$@"
''')
    base = read(P/'validation/results/response_design_20261004_reverse_endpoint_k4/ground_command.json')
    b, e = base.index('env')+1, base.index('/bin/bash')
    settings = dict(x.split('=', 1) for x in base[b:e])
    cfg = read(OLD/'protocol.json')
    query = ['srun', '--jobid=250212', '--overlap', '--cpu-bind=none',
             '-N1', '-n1', '--cpus-per-task=32']
    allowed = json.loads(subprocess.check_output(query + ['/usr/bin/python3', '-c',
        'import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'], text=True))
    physical = [x for x in allowed if x < 64]
    if len(physical) < 8:
        raise RuntimeError('Insufficient CPUs for four independent workers')
    recovery=P/'validation/results/response_design_20261005_numerical_verification/gradient_v2/runs/fine_reference'
    cfg.update(root=str(R), jobid=250212, settings=settings, query=query,
               allowed_cpus=allowed, physical_cpus=physical,
               python=base[e+2], fine_xc_factor=5, production_xc_factor=4,
               epochs=3000, fd_steps=[0.001, 0.0005],
               gradient_absolute_floor_eV=0.0001, gradient_relative_tolerance=0.01,
               gap_setting_budget_eV=0.005,
               recovery_checkpoint=str(recovery),
               recovery_source_hashes={str(recovery/n):sha(recovery/n) for n in
                   ['metadata.json','state.pkl','mask.npy','energy_history.npy','input.xyz']},
               source_hashes={str(f): sha(f) for f in (R/'drivers').glob('*.py')},
               scope='One Gamma input direction, identical Miller basis and FFT144; XC quadrature576->720 only. Not a multi-k adjoint or cutoff-convergence certificate.')
    cfg['parent_hashes'] = {str(Path(cfg['source'])/n): sha(Path(cfg['source'])/n)
                            for n in ['metadata.json', 'point.json', 'mask.npy',
                                      'density.npy', 'state.pkl', 'input.xyz']}
    dump(R/'gradient_protocol.json', cfg)
    dump(R/'gradient_status.json', dict(stage='prepared', time=datetime.datetime.now().isoformat()))
    dump(R.parent/'gradient_active.json',dict(root=str(R),prior_failure='XC720 electronic3000 complete; eager stress prevalidation requested16.68GiB cuFFT batch3 scratch. v3 reuses saved checkpoint; exact scalar-component FFTs and cache release; no quadrature/tolerance changes.'))
    print(R)


def execute():
    cfg = read(R/'gradient_protocol.json')
    def stage(label, **extra):
        row = dict(stage=label, time=datetime.datetime.now().isoformat(), pid=os.getpid(), **extra)
        dump(R/'gradient_status.json', row)
        print('NUMERICAL_GRADIENT_STAGE', json.dumps(row), flush=True)
    def prefix(gpu, tag):
        tmp = '/dev/shm/d3ft_response_design_numcheck_'+tag
        env = dict(cfg['settings'])
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=tmp,
                   PYTHONPYCACHEPREFIX=tmp+'/python_cache',
                   JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache',
                   JAX_ENABLE_COMPILATION_CACHE='false', OMP_NUM_THREADS='2')
        cores = cfg['physical_cpus'][2*gpu:2*gpu+2]
        affinity = ','.join(str(x) for c in cores for x in [c, c+64] if x in cfg['allowed_cpus'])
        return cfg['query'] + ['--gres=gpu:4', '--gpu-bind=none', 'taskset', '-c', affinity,
          'timeout', '--signal=TERM', '--kill-after=30s', '28800s', 'env'] + [
          k+'='+v for k,v in env.items()] + ['/bin/bash', str(R/'private_tmp.sh'), cfg['python'], '-u']
    def run(cmd, tag):
        dump(R/'commands'/(tag+'.json'), cmd)
        with (R/'logs'/(tag+'.log')).open('xb') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        dump(R/'commands'/(tag+'_exit.json'), dict(returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f'{tag}: return code {result.returncode}; see retained log')
    def point(tag, theta, gpu, parent, quadrature=False,report_only=False):
        runtime = Path(cfg['runtime'])
        driver = runtime/'validation/workflows/response_design/generic_point.py'
        pm = read(Path(parent)/'metadata.json')
        out = R/'runs'/tag
        args = ['--structure', pm['arguments']['structure'], '--pp-dir', pm['arguments']['pp_dir'],
          '--cutoff', '100', '--grid', '144', '--k', '1', '--epochs', str(cfg['epochs']),
          '--xc-grid-factor', '5', '--local-radial-quadrature', 'simpson',
          '--output', str(out), '--strain', json.dumps(matrix(theta)),
          '--resume-from', str(parent), '--require-stationary', '--fused-report-all-devices',
          '--learning-rate', '0.001', '--final-learning-rate', '0.000001',
          '--resume-quadrature-refinement' if quadrature else '--resume-geometry-continuation']
        if report_only:
            args[args.index('--epochs')+1]='1'
            args.append('--checkpoint-report-only')
        run(prefix(gpu, tag)+[str(R/'drivers/sigma_runner.py'),str(R/'drivers/bounded_generic.py'), str(driver)]+args, tag)
        result = read(out/'point.json')
        if not result['stationarity_pass'] or result['residual'] > 1e-6:
            raise RuntimeError('Stationarity failed: '+tag)
        if result['acoustic_sum_rule_Ha_Bohr'] > 5e-8:
            raise RuntimeError('Raw ASR failed: '+tag)
        if abs(result['integrated_valence_density']-96) > .00096:
            raise RuntimeError('Charge failed: '+tag)
        if sha(out/'mask.npy') != sha(Path(parent)/'mask.npy'):
            raise RuntimeError('Fixed Miller basis changed: '+tag)
        return out
    def orbitals(src, tag, gpu):
        output = R/'orbitals'/tag
        run(prefix(gpu, tag+'_orbitals')+[str(R/'drivers/sigma_runner.py'),str(R/'drivers/save_response_orbitals.py'),
             str(src), str(output)], tag+'_orbitals')
        return output
    try:
        stage('validating_sources')
        for f,h in {**cfg['parent_hashes'], **cfg['source_hashes'],**cfg['recovery_source_hashes']}.items():
            if sha(f) != h:
                raise RuntimeError('Immutable input changed: '+f)
        inv = subprocess.check_output(cfg['query']+['nvidia-smi', '--query-gpu=index,memory.used',
          '--format=csv,noheader,nounits'], text=True)
        if any(int(row.split(',')[1]) > 100 for row in inv.strip().splitlines()):
            raise RuntimeError('Four-GPU allocation is occupied')
        theta = np.asarray(cfg['theta0']); direction = np.asarray(cfg['direction'])
        stage('fine_reference_checkpoint_recovery', jobid=250212, physical_gpu=0,recovered_electronic_steps=3000)
        fine = point('fine_reference', theta, 0, cfg['recovery_checkpoint'],report_only=True)
        stage('fine_reference_orbitals')
        orb = orbitals(fine, 'fine_reference', 0)
        response_cfg = dict(cfg, source=str(fine), orbitals=str(orb))
        dump(R/'fine_response_protocol.json', response_cfg)
        tasks = [('plus_h', .001, 0), ('minus_h', -.001, 1),
                 ('plus_half', .0005, 2), ('minus_half', -.0005, 3)]
        def fd_task(task):
            tag, t, gpu = task
            src = point(tag, theta+t*direction, gpu, fine)
            return tag, orbitals(src, tag, gpu)
        stage('fine_reoptimized_fd_pairs', tasks=[t[0] for t in tasks])
        with ThreadPoolExecutor(max_workers=4) as pool:
            fd_orbitals = dict(pool.map(fd_task, tasks))
        stage('fine_reverse_response', physical_gpu=0)
        run(prefix(0, 'fine_reverse')+[str(R/'drivers/sigma_runner.py'),str(R/'drivers/gamma_response.py'),
            str(R/'fine_response_protocol.json'), '--mode', 'reverse', '--output', str(R/'response/fine_reverse')],
            'fine_reverse')
        fine_result = read(R/'response/fine_reverse/result.json')
        fine_gradient = np.asarray(fine_result['results']['gap_gradient_eV'])
        production_gradient = np.asarray(read(OLD/'reverse/result.json')['results']['gap_gradient_eV'])
        fine_ad = float(fine_gradient@direction); coarse_ad = float(production_gradient@direction)
        gaps = {tag: read(path/'summary.json')['gap_eV'] for tag,path in fd_orbitals.items()}
        fd_h = (gaps['plus_h']-gaps['minus_h'])/.002
        fd_half = (gaps['plus_half']-gaps['minus_half'])/.001
        richardson = (4*fd_half-fd_h)/3
        budget = cfg['gradient_absolute_floor_eV'] + cfg['gradient_relative_tolerance']*abs(fine_ad)
        coarse_gap = read(Path(cfg['orbitals'])/'summary.json')['gap_eV']
        fine_gap = read(orb/'summary.json')['gap_eV']
        checks = dict(adjoint_converged=fine_result['results']['response_record']['converged'],
           fine_ad_vs_half_fd=abs(fine_ad-fd_half)<=budget,
           fine_ad_vs_richardson=abs(fine_ad-richardson)<=budget,
           fd_step_stability=abs(fd_h-fd_half)<=budget,
           production_vs_fine_gradient=abs(coarse_ad-fine_ad)<=budget,
           production_vs_fine_gap=abs(coarse_gap-fine_gap)<=cfg['gap_setting_budget_eV'])
        report = dict(completed=True, passed=all(checks.values()), checks=checks,
           production_directional_gradient_eV=coarse_ad, fine_directional_gradient_eV=fine_ad,
           fine_fd_h_eV=fd_h, fine_fd_half_eV=fd_half, fine_fd_richardson_eV=richardson,
           gradient_setting_difference_eV=fine_ad-coarse_ad, gradient_budget_eV=budget,
           production_gamma_gap_eV=coarse_gap, fine_gamma_gap_eV=fine_gap,
           gap_setting_difference_eV=fine_gap-coarse_gap, fine_gap_gradient_eV=fine_gradient.tolist(),
           sources={tag:str(path) for tag,path in fd_orbitals.items()},
           immutable_sources_rechecked=all(sha(f)==h for f,h in cfg['parent_hashes'].items()),
           scope=cfg['scope'], time=datetime.datetime.now().isoformat())
        if not report['immutable_sources_rechecked']:
            raise RuntimeError('Production source changed during campaign')
        dump(R/'gradient_result.json', report)
        stage('completed', passed=report['passed'])
    except Exception as exc:
        stage('failed', error=str(exc))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prepare', action='store_true')
    a = ap.parse_args()
    prepare() if a.prepare else execute()
