"""One bounded six-strain Gamma design, with serial same-GPU AD-mode controls.

Fresh complete gradients are computed in both modes. Only the qualified initial
electronic state/orbitals are shared. Every proposed geometry is independently
reoptimized and read out; no linear prediction is accepted as a result.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

P = Path(__file__).resolve()
while not (P / 'validation').is_dir():
    P = P.parent
DEFAULT_ROOT = P / 'validation/results/response_design_20261006_six_strain_design'


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def matrix(theta):
    a, b, c, d, e, f = theta
    return np.array([[a, f / 2, e / 2], [f / 2, b, d / 2], [e / 2, d / 2, c]])


def proposal(theta, origin, gradient, gap, target, step_radius, total_radius):
    """Minimum-norm linearized target step in six free engineering strains."""
    theta, origin, gradient = map(lambda x: np.asarray(x, dtype=float),
                                  [theta, origin, gradient])
    if any(x.shape != (6,) for x in [theta, origin, gradient]):
        raise ValueError('Expected six strain components')
    if not np.isfinite(np.r_[theta, origin, gradient, gap, target]).all():
        raise ValueError('Nonfinite optimizer input')
    if not 0 < step_radius <= total_radius:
        raise ValueError('Invalid strain radii')
    squared = float(gradient @ gradient)
    if squared < 1e-12:
        raise RuntimeError('Gap gradient is too small')
    step = (target - gap) * gradient / squared
    step *= min(1., step_radius / max(np.linalg.norm(step), 1e-30))
    delta = theta + step - origin
    delta *= min(1., total_radius / max(np.linalg.norm(delta), 1e-30))
    trial = origin + delta
    if np.linalg.norm(trial - theta) < 1e-10:
        raise RuntimeError('Bounded optimizer stalled')
    if np.linalg.det(np.eye(3) + matrix(trial)) <= 0:
        raise RuntimeError('Nonpositive proposed cell volume')
    return trial


def source_gate(cfg, source):
    source = Path(source)
    point, meta = read(source / 'point.json'), read(source / 'metadata.json')
    parent = read(Path(cfg['base_source']) / 'metadata.json')
    if not point['stationarity_pass'] or point['residual'] > 1e-6:
        raise ValueError('Electronic stationarity gate failed')
    if point['acoustic_sum_rule_Ha_Bohr'] > 5e-8:
        raise ValueError('Raw acoustic sum gate failed')
    if abs(point['integrated_valence_density'] - 96.) > .00096:
        raise ValueError('96-electron charge gate failed')
    if sha(source / 'mask.npy') != cfg['parent_mask_sha256']:
        raise ValueError('Fixed Miller mask changed')
    if meta['code_sha256'] != parent['code_sha256'] or meta['physical']['pp'] != parent['physical']['pp']:
        raise ValueError('Frozen physical solver/pseudopotentials changed')
    if len(point['k_fractional']) != 1:
        raise ValueError('This protocol is Gamma-only')
    cell = np.array(meta['physical']['cell_angstrom'])
    positions = np.array(meta['physical']['positions_angstrom'])
    pcell = np.array(parent['physical']['cell_angstrom'])
    ppositions = np.array(parent['physical']['positions_angstrom'])
    fractional_difference = positions @ np.linalg.inv(cell) - ppositions @ np.linalg.inv(pcell)
    if np.max(abs(fractional_difference)) > 1e-9:
        raise ValueError('Clamped fractional coordinates changed')
    return point


def orbital_gate(source, orbital):
    summary, meta = read(orbital / 'summary.json'), read(orbital / 'metadata.json')
    for name, digest in meta['source_hashes'].items():
        if sha(source / name) != digest:
            raise ValueError('Orbital parent binding changed: ' + name)
    if sha(orbital / 'orbitals.npy') != summary['saved_orbitals_sha256']:
        raise ValueError('Orbital coefficients changed')
    if max(summary['residuals_Ha']) > 1e-7 or summary['orthogonality_error'] > 1e-8:
        raise ValueError('Eigenpair gate failed')
    if min(summary['vbm_adjacent_separation_Ha'], summary['cbm_adjacent_separation_Ha']) <= 1e-5:
        raise ValueError('Band edge requires degenerate-subspace treatment')
    return summary


def prepare(root, jobid, gpu):
    check_script = P / 'validation/workflows/response_design/check_six_strain_optimizer.py'
    optimizer_check = json.loads(subprocess.check_output([sys.executable, str(check_script)], text=True))
    if not optimizer_check['passed']:
        raise ValueError('Independent six-strain optimizer checks failed')
    old = P / 'validation/results/response_design_20261004_response_design'
    accepted_root = P / 'validation/results/response_design_20261004_reverse_response_v5'
    accepted = read(accepted_root / 'acceptance.json')
    if not accepted['passed']:
        raise ValueError('Base six-component physical response not accepted')
    cfg = read(old / 'protocol.json')
    query = ['srun', f'--jobid={jobid}', '--overlap', '--cpu-bind=none', '-N1', '-n1', '--cpus-per-task=32']
    information = subprocess.check_output(query + ['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = [row.split(',') for row in information.strip().splitlines()]
    selected = [row for row in rows if int(row[0]) == gpu]
    if len(selected) != 1 or int(selected[0][2]) > 100 or int(selected[0][3]) > 0:
        raise RuntimeError('Selected GPU is not idle: ' + information)
    allowed = json.loads(subprocess.check_output(query + ['python3', '-c',
        'import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))'], text=True))
    physical = [x for x in allowed if x < 64][:2]
    cpu_mask = ','.join(str(x) for core in physical for x in [core, core + 64] if x in allowed)
    root.mkdir(exist_ok=False)
    for name in ['drivers', 'execution', 'runs', 'orbitals', 'responses']:
        (root / name).mkdir()
    for name in ['gamma_response.py', 'projected_solver.py', 'bounded_xc.py']:
        shutil.copy2(accepted_root / name, root / 'drivers' / name)
    orbital_helper = old / 'save_response_orbitals.py'
    if sha(orbital_helper) != cfg['source_hashes']['save_response_orbitals.py']:
        raise ValueError('Qualified orbital helper changed')
    shutil.copy2(orbital_helper, root / 'drivers/save_response_orbitals.py')
    shutil.copy2(Path(__file__), root / 'controller.py')
    wrapper = root / 'private_tmp.sh'
    wrapper.write_text('''#!/bin/bash
set -eu
case "${TMPDIR:-}" in /dev/shm/d3ft_response_design_20261006_sixstrain_*) ;; *) exit 90;; esac
[ ! -e "$TMPDIR" ] || exit 91
mkdir -m 700 -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
ulimit -c 0
"$@"
''')
    previous_prefix = cfg['prefixes']['0']
    begin = previous_prefix.index('env') + 1
    end = previous_prefix.index('/bin/bash')
    settings = dict(item.split('=', 1) for item in previous_prefix[begin:end])
    settings.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    cfg.update(jobid=jobid, physical_gpu=gpu, gpu_uuid=selected[0][1].strip(),
        cpu_mask=cpu_mask, query_prefix=query, environment=settings,
        private_wrapper=str(wrapper), python=previous_prefix[-1],
        initial_orbitals=read(accepted_root / 'protocol.json')['orbitals'],
        root=str(root), design_max_updates=3, max_backtracks=2,
        step_radius=.01, total_radius=.03, design_tolerance_eV=.001,
        base_response_protocol=read(accepted_root / 'protocol.json'),
        accepted_initial_gradient=accepted['reverse_gradient_eV'],
        mode_order=['reverse', 'forward'],
        optimizer='Six-free-component minimum-norm linearized target step; L2 step/total bounds; up to two halvings accepted only by actual reoptimized loss.',
        timing_scope='Cold subprocess wall time includes srun launch, compilation, physical response/ground/readout, hash checks and I/O. Same physical GPU/CPU affinity serial modes. Shared qualified initial state excluded; initial complete gradient recomputed for each mode. Existing separate warmed six-gradient benchmark retained, no universal end-to-end speedup claim.',
        frozen_at=datetime.datetime.now().isoformat(),
        scope='One start, one scalar Gamma transition target, six engineering-strain inputs, fixed fractions/occupations/Miller/PBE. Not full-BZ fundamental-gap design, relaxed-ion optimization, or high-dimensional ML generalization.')
    cfg.pop('prefixes', None)
    parent_hashes = {str(Path(cfg['base_source']) / n): sha(Path(cfg['base_source']) / n)
        for n in ['point.json', 'metadata.json', 'input.xyz', 'density.npy', 'mask.npy']}
    orb = Path(cfg['initial_orbitals'])
    parent_hashes.update({str(orb / n): sha(orb / n) for n in ['summary.json', 'metadata.json', 'orbitals.npy']})
    files = [root / 'controller.py', wrapper, *sorted((root / 'drivers').glob('*.py')), check_script,
             Path(cfg['point_tail'][1]), Path(__file__).resolve()]
    cfg['frozen_hashes'] = {str(f): sha(f) for f in files}
    cfg['parent_hashes'] = parent_hashes
    source_gate(cfg, Path(cfg['base_source']))
    summary = orbital_gate(Path(cfg['base_source']), orb)
    if abs(summary['gap_eV'] - cfg['reference_gamma_eV']) > 3e-6:
        raise ValueError('Initial gap differs from qualified reference')
    dump(root / 'protocol.json', cfg)
    dump(root / 'optimizer_check.json', dict(optimizer_check, check_script_sha256=sha(check_script)))
    dump(root / 'resource_preflight.json', dict(time=cfg['frozen_at'], gpu_information=information,
        allowed_cpus=allowed, selected_gpu=cfg['gpu_uuid'], cpu_mask=cpu_mask))
    dump(root / 'status.json', dict(stage='prepared', time=cfg['frozen_at'], mode_order=cfg['mode_order']))
    return cfg


def trajectory(root, cfg, mode):
    history = []
    timings = []
    start = time.monotonic()

    def stage(label, **extra):
        record = dict(stage=label, mode=mode, time=datetime.datetime.now().isoformat(), **extra)
        dump(root / (mode + '_status.json'), record)
        dump(root / 'status.json', record)
        print('SIX_STRAIN_STAGE', json.dumps(record), flush=True)

    def run(tail, tag, category):
        temporary = '/dev/shm/d3ft_response_design_20261006_sixstrain_' + tag
        env = dict(cfg['environment'], TMPDIR=temporary,
            JAX_COMPILATION_CACHE_DIR=temporary + '/jax_cache', PYTHONPYCACHEPREFIX=temporary + '/python_cache')
        command = cfg['query_prefix'] + ['--gres=gpu:4', '--gpu-bind=none', 'taskset', '-c', cfg['cpu_mask'],
            'timeout', '--signal=TERM', '--kill-after=30s', '21600s', 'env']
        command += [f'{k}={v}' for k, v in env.items()]
        command += ['/bin/bash', cfg['private_wrapper'], cfg['python'], *tail]
        dump(root / 'execution' / (tag + '_command.json'), command)
        t0 = time.monotonic()
        with (root / 'execution' / (tag + '.log')).open('xb') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = dict(tag=tag, category=category, wall_seconds=time.monotonic() - t0,
            returncode=completed.returncode, mode=mode, includes_srun_and_first_use_compile=True)
        timings.append(record)
        dump(root / (mode + '_timings.json'), timings)
        if completed.returncode:
            raise RuntimeError(f'{tag}: RC {completed.returncode}; original log retained')

    source = Path(cfg['base_source'])
    orbital = Path(cfg['initial_orbitals'])
    theta = np.array(cfg['theta0'])
    origin = theta.copy()
    target = cfg['target_gamma_eV']
    gap = orbital_gate(source, orbital)['gap_eV']
    source_gate(cfg, source)
    history.append(dict(iteration=0, theta=theta.tolist(), gap_eV=gap, error_eV=gap-target,
        source=str(source), orbitals=str(orbital), shared_qualified_initial_primal=True))
    dump(root / (mode + '_history.json'), history)
    for iteration in range(cfg['design_max_updates']):
        tag = f'{mode}_iter{iteration}'
        stage('electronic_response', iteration=iteration)
        response_cfg = dict(cfg['base_response_protocol'], source=str(source), orbitals=str(orbital),
            theta0=theta.tolist(), source_hashes=cfg['frozen_hashes'])
        protocol = root / 'execution' / (tag + '_response_protocol.json')
        dump(protocol, response_cfg)
        output = root / 'responses' / tag
        tail = ['-u', str(root / 'drivers/gamma_response.py'), str(protocol), '--mode', mode, '--output', str(output)]
        if mode == 'forward':
            tail += ['--axes', '0', '1', '2', '3', '4', '5']
        run(tail, tag + '_response', 'response')
        result = read(output / 'result.json')
        if result['kernel_symmetry_relative_error'] > 1e-8:
            raise ValueError('Response symmetry gate failed')
        records = [result['results']['response_record']] if mode == 'reverse' else [result['results'][str(i)]['response_record'] for i in range(6)]
        if not all(row['converged'] and row['residual'] <= max(row.get('threshold', 1e-11), 1e-11) for row in records):
            raise ValueError('True electronic response residual gate failed')
        gradient = np.array(result['results']['gap_gradient_eV'] if mode == 'reverse' else
            [result['results'][str(i)]['gap_derivative_eV'] for i in range(6)])
        if iteration == 0 and np.max(abs(gradient - cfg['accepted_initial_gradient'])) > 1e-5:
            raise ValueError('Fresh initial gradient differs from qualified six-component gradient')
        history[-1].update(gradient_eV=gradient.tolist(), response=str(output),
            response_right_hand_sides=len(records), response_records=[{k:v for k,v in row.items() if k!='history'} for row in records],
            solve_and_contract_seconds=result['solve_and_contract_seconds'],
            response_kernel_compile_seconds=result['kernel_compile_seconds'])
        dump(root / (mode + '_history.json'), history)
        candidate = proposal(theta, origin, gradient, gap, target, cfg['step_radius'], cfg['total_radius'])
        accepted = False
        trials = []
        for backtrack in range(cfg['max_backtracks'] + 1):
            trial = theta + (candidate - theta) * (.5 ** backtrack)
            trial_tag = f'{mode}_iter{iteration+1}_trial{backtrack}'
            stage('ground_state', iteration=iteration+1, backtrack=backtrack, theta=trial.tolist())
            new_source = root / 'runs' / trial_tag
            tail = list(cfg['point_tail'])
            for flag, value in [('--output', str(new_source)), ('--resume-from', str(source)),
                    ('--strain', json.dumps(matrix(trial).tolist())), ('--epochs', str(cfg['design_epochs']))]:
                tail[tail.index(flag) + 1] = value
            run(tail, trial_tag + '_ground', 'ground_state')
            source_gate(cfg, new_source)
            new_orbital = root / 'orbitals' / trial_tag
            stage('readout', iteration=iteration+1, backtrack=backtrack)
            run(['-u', str(root / 'drivers/save_response_orbitals.py'), str(new_source), str(new_orbital)],
                trial_tag + '_orbitals', 'readout')
            summary = orbital_gate(new_source, new_orbital)
            new_gap = summary['gap_eV']
            better = abs(new_gap-target) < abs(gap-target)
            trials.append(dict(backtrack=backtrack, theta=trial.tolist(), gap_eV=new_gap,
                error_eV=new_gap-target, actual_loss_accepted=better, source=str(new_source), orbitals=str(new_orbital)))
            dump(root / 'execution' / (tag + '_trials.json'), trials)
            if better:
                accepted = True
                history.append(dict(iteration=iteration+1, theta=trial.tolist(), gap_eV=new_gap,
                    error_eV=new_gap-target, actual_loss_accepted=True, step_norm=float(np.linalg.norm(trial-theta)),
                    total_displacement_norm=float(np.linalg.norm(trial-origin)), source=str(new_source), orbitals=str(new_orbital),
                    backtracks=backtrack, actual_improvement_eV=abs(gap-target)-abs(new_gap-target)))
                theta, gap, source, orbital = trial, new_gap, new_source, new_orbital
                dump(root / (mode + '_history.json'), history)
                break
        if not accepted:
            stage('completed', converged=False, reason='Actual objective failed all registered backtracking trials')
            break
        if abs(gap-target) <= cfg['design_tolerance_eV']:
            stage('completed', converged=True, updates=iteration+1, gap_eV=gap, error_eV=gap-target)
            break
    else:
        stage('completed', converged=False, reason='Registered accepted-update limit')
    dump(root / (mode + '_summary.json'), dict(mode=mode, status=read(root/(mode+'_status.json')),
        elapsed_seconds=time.monotonic()-start, subprocess_seconds=sum(row['wall_seconds'] for row in timings),
        response_seconds=sum(row['wall_seconds'] for row in timings if row['category']=='response'),
        ground_seconds=sum(row['wall_seconds'] for row in timings if row['category']=='ground_state'),
        readout_seconds=sum(row['wall_seconds'] for row in timings if row['category']=='readout'),
        gpu_uuid=cfg['gpu_uuid'], scope=cfg['timing_scope']))


def execute(root):
    cfg = read(root / 'protocol.json')
    for path, digest in {**cfg['frozen_hashes'], **cfg['parent_hashes']}.items():
        if sha(path) != digest:
            raise RuntimeError('Frozen input changed: ' + path)
    lock = root / 'running.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()) + '\n')
    try:
        for mode in cfg['mode_order']:
            uuid = subprocess.check_output(cfg['query_prefix'] + ['nvidia-smi', f'--id={cfg["physical_gpu"]}',
                '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
            if uuid != cfg['gpu_uuid']:
                raise ValueError('Physical GPU changed between AD modes')
            trajectory(root, cfg, mode)
        audit(root)
    except Exception as exc:
        failure = dict(stage='failed', error=str(exc), mode=read(root/'status.json').get('mode'),
                       time=datetime.datetime.now().isoformat())
        dump(root / 'status.json', failure)
        if failure['mode']:
            dump(root / (failure['mode']+'_status.json'), failure)
        raise
    finally:
        lock.unlink()


def audit(root):
    cfg = read(root / 'protocol.json')
    for path, digest in {**cfg['frozen_hashes'], **cfg['parent_hashes']}.items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed at final audit: ' + path)
    histories = {}
    for mode in cfg['mode_order']:
        history = read(root / (mode + '_history.json'))
        for index, row in enumerate(history):
            source_gate(cfg, Path(row['source']))
            summary = orbital_gate(Path(row['source']), Path(row['orbitals']))
            if abs(summary['gap_eV'] - row['gap_eV']) > 1e-12:
                raise ValueError('Recorded objective differs from eigenpairs')
            if abs(row['error_eV'] - (row['gap_eV']-cfg['target_gamma_eV'])) > 1e-12:
                raise ValueError('Target arithmetic changed')
            strain = np.array(json.loads(read(Path(row['source'])/'metadata.json')['arguments']['strain']))
            if np.max(abs(strain-matrix(row['theta']))) > 1e-12:
                raise ValueError('Recorded six strain coordinates differ from physical geometry')
            if 'response' in row:
                raw = read(Path(row['response'])/'result.json')
                values = raw['results']['gap_gradient_eV'] if mode=='reverse' else [raw['results'][str(i)]['gap_derivative_eV'] for i in range(6)]
                if np.max(abs(np.array(values)-row['gradient_eV'])) > 1e-12:
                    raise ValueError('Optimizer gradient differs from actual response')
                if raw['primal_residual_Ha'] > 1e-7 or raw['kernel_symmetry_relative_error'] > 1e-8:
                    raise ValueError('Response primal/kernel gates failed')
            if index:
                if abs(row['error_eV']) >= abs(history[index-1]['error_eV']):
                    raise ValueError('Non-improving accepted update')
                if np.linalg.norm(np.array(row['theta'])-cfg['theta0']) > cfg['total_radius']+1e-12:
                    raise ValueError('Total strain bound exceeded')
                if np.linalg.norm(np.array(row['theta'])-history[index-1]['theta']) > cfg['step_radius']+1e-12:
                    raise ValueError('Strain step bound exceeded')
        histories[mode] = history
    reverse, forward = histories['reverse'], histories['forward']
    matched_updates = len(reverse) == len(forward)
    theta_difference = max(float(np.max(abs(np.array(a['theta'])-b['theta']))) for a,b in zip(reverse,forward))
    gap_difference = max(abs(a['gap_eV']-b['gap_eV']) for a,b in zip(reverse,forward))
    converged = all(read(root/(m+'_status.json')).get('converged',False) for m in cfg['mode_order'])
    passed = converged and matched_updates and theta_difference <= 1e-6 and gap_difference <= 1e-5
    result = dict(completed=True, passed=passed, both_targets_reached=converged,
        matched_update_counts=matched_updates, maximum_component_trajectory_difference=theta_difference,
        maximum_gap_trajectory_difference_eV=gap_difference,
        summaries={m:read(root/(m+'_summary.json')) for m in cfg['mode_order']},
        time=datetime.datetime.now().isoformat(), protocol_sha256=sha(root/'protocol.json'),
        scope=cfg['scope'], timing_scope=cfg['timing_scope'])
    dump(root / 'result.json', result)
    dump(root / 'status.json', dict(stage='completed', passed=passed, time=result['time']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--jobid', type=int, default=250212)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.jobid, args.gpu)
        print(args.root)
    if args.run:
        execute(args.root)
    if args.audit:
        print(json.dumps(audit(args.root), indent=2))
