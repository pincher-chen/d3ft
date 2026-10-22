"""Persist campaign progress and audit terminal scientific outcomes independently."""
import argparse, datetime, hashlib, json, os, time
from pathlib import Path
import numpy as np

P = Path('${D3FT_ROOT}')
R = P / 'validation/results/response_design_20261004_response_design'

def read(path):
    return json.loads(Path(path).read_text())

def optional(path):
    return read(path) if Path(path).exists() else {}

def dump(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def audit_elastic():
    cfg = read(R / 'execution/elastic_response_protocol.json')
    response = read(R / 'elastic_response/result.json')
    sources = [Path(cfg['source']), Path(cfg['fd_plus']), Path(cfg['fd_minus']),
               R / 'runs/elastic_plus_half', R / 'runs/elastic_minus_half']
    points = [read(source / 'point.json') for source in sources]
    base = points[0]
    base_meta = read(sources[0] / 'metadata.json')
    for source, point in zip(sources, points):
        assert point['stationarity_pass'] and point['residual'] <= 1e-6
        assert point['acoustic_sum_rule_Ha_Bohr'] <= 5e-8
        assert abs(point['integrated_valence_density'] - 96) <= .00096
        assert sha(source / 'mask.npy') == sha(sources[0] / 'mask.npy')
        meta = read(source / 'metadata.json')
        assert meta['code_sha256'] == base_meta['code_sha256']
        assert meta['physical']['pp'] == base_meta['physical']['pp']
    # Validate geometries rather than identifying a point by its filename alone.
    theta = np.asarray(cfg['theta0']); direction = np.asarray(cfg['direction'])
    def strain(v):
        return np.array([[v[0],v[5]/2,v[4]/2], [v[5]/2,v[1],v[3]/2], [v[4]/2,v[3]/2,v[2]]])
    base_cell = np.asarray(base_meta['physical']['cell_angstrom'])
    base_inverse = np.linalg.inv(np.eye(3) + strain(theta))
    for source, t in zip(sources, [0., .001, -.001, .0005, -.0005]):
        meta = read(source / 'metadata.json')
        expected = base_cell @ (np.eye(3) + t * strain(direction) @ base_inverse).T
        assert np.allclose(meta['physical']['cell_angstrom'], expected, atol=1e-10, rtol=0)
    scale = 29421.015697 / base['volume_bohr3']
    energy = [point['energy_Ha'] for point in points]
    coarse = (energy[1] - 2*energy[0] + energy[2]) / .001**2 * scale
    fine = (energy[3] - 2*energy[0] + energy[4]) / .0005**2 * scale
    ad = response['selfconsistent_modulus_GPa']
    tolerance = lambda value: cfg['elastic_absolute_tolerance_GPa'] + cfg['elastic_relative_tolerance'] * abs(value)
    passed = (response['response_record']['converged'] and response['passed']
              and abs(ad-fine) <= tolerance(fine) and abs(coarse-fine) <= tolerance(fine))
    return dict(passed=bool(passed), analytic_GPa=ad, FD_h_GPa=coarse, FD_half_h_GPa=fine,
                analytic_vs_half_h_GPa=abs(ad-fine), FD_step_difference_GPa=abs(coarse-fine),
                registered_absolute_tolerance_GPa=cfg['elastic_absolute_tolerance_GPa'],
                registered_relative_tolerance=cfg['elastic_relative_tolerance'],
                sources=[str(s) for s in sources],
                scope='One Gamma clamped-ion affine energy-curvature mode including prestress; not a relaxed-ion/full elastic tensor')

def snapshot():
    native_active = optional(R/'native_active.json')
    native_root = Path(native_active.get('output',str(R/'native_path')))
    statuses = {name: optional(R / (name+'_status.json')) for name in ['reverse','forward']}
    statuses['native'] = optional(native_root / 'status.json')
    launch = read(R / 'launch.json')
    processes = {row['task']: alive(row['pid']) for row in launch['tasks']}
    errors = []
    for task, expected in [('half_plus', 'runs/elastic_plus_half/point.json'),
                           ('elastic', 'runs/elastic_minus_half/point.json')]:
        if not processes[task] and not (R/expected).exists():
            failure = dict(task=task,error='Controller exited without its required FD point')
            errors.append(failure)
            if not (R/'elastic_task_failure.json').exists(): dump(R/'elastic_task_failure.json',failure)
    elastic_ready = all((R / f).exists() for f in ['elastic_response/result.json',
                        'runs/elastic_plus_half/point.json','runs/elastic_minus_half/point.json'])
    if elastic_ready and not (R/'elastic_acceptance.json').exists():
        try:
            dump(R/'elastic_acceptance.json', audit_elastic())
        except Exception as exc:
            dump(R/'elastic_acceptance.json', dict(passed=False, audit_error=str(exc)))
    elastic = optional(R/'elastic_acceptance.json')
    native_launcher = optional(native_active.get('launch',str(R/'native_launch_v3.json')))
    native_pid = native_launcher.get('pid')
    native_alive = alive(native_pid) if native_pid else None
    for name in ['reverse','forward']:
        s = statuses[name]
        if s.get('stage') == 'failed' or s.get('stage') == 'completed' and not s.get('converged'):
            errors.append(dict(task=name, status=s))
        elif not processes[name] and s.get('stage') != 'completed':
            errors.append(dict(task=name, error='Controller exited before terminal status'))
    if statuses['native'].get('stage') == 'failed':
        errors.append(dict(task='native',status=statuses['native']))
    elif native_alive is False and statuses['native'].get('stage') != 'completed':
        errors.append(dict(task='native',error='Native launcher exited before terminal status'))
    if elastic and not elastic.get('passed'):
        errors.append(dict(task='elastic',acceptance=elastic))
    if not elastic_ready and not processes['elastic'] and not processes['half_plus']:
        errors.append(dict(task='elastic',error='Controllers ended without both FD points and response'))
    terminal = (all(s.get('stage') in ['completed','failed'] for s in statuses.values())
                and (elastic_ready or not processes['elastic'] and not processes['half_plus']))
    data = dict(time=datetime.datetime.now().isoformat(),stage='completed' if terminal else 'running',
                trajectory_statuses=statuses,controller_alive=processes,native_launcher_alive=native_alive,
                elastic_acceptance=elastic,errors=errors,
                endpoint_queues={f'k{k}':optional(P/f'validation/results/response_design_20261004_reverse_endpoint_k{k}/status.json') for k in [2,4]})
    if terminal:
        cfg = read(R/'protocol.json')
        histories = {name: optional(R/(name+'_history.json')) for name in ['reverse','forward']}
        histories['native'] = optional(native_root/'history.json')
        data['histories'] = histories
        if all(statuses[n].get('converged') for n in ['reverse','forward']):
            rev, fwd = histories['reverse'], histories['forward']
            path_difference = max(abs(a['t']-b['t']) for a,b in zip(rev,fwd))
            gap_difference = max(abs(a['gap_eV']-b['gap_eV']) for a,b in zip(rev,fwd))
            matched = len(rev)==len(fwd) and path_difference <= 1e-6 and gap_difference <= cfg['design_tolerance_eV']
            data['same_program_comparison'] = dict(passed=matched,max_path_difference=path_difference,max_gap_difference_eV=gap_difference)
            if not matched: data['errors'].append(dict(task='same_program_comparison',error='Matched trajectory disagreement'))
        native_converged = statuses['native'].get('converged',False)
        data['native_trajectory_converged'] = native_converged
        if not native_converged: data['errors'].append(dict(task='native',error='Native target not reached'))
        data['passed'] = not data['errors'] and bool(elastic.get('passed'))
    dump(R/'campaign_progress.json',data)
    if terminal: dump(R/'campaign_summary.json',data)
    return terminal

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--once', action='store_true');args=parser.parse_args()
    for _ in range(2160):
        if snapshot() or args.once: break
        time.sleep(20)
    else: dump(R/'monitor_timeout.json',dict(error='Registered 12h monitoring limit; computations not cancelled'))
