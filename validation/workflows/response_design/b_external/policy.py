"""CPU-only gates for the approved, frozen AlP/GaAs external evaluations."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def now():
    return dt.datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def note(project, text):
    for name in ['paper/response_design_EXPERIMENTS_20261014.md', 'response_design_to_do_list.md']:
        with (Path(project) / name).open('a') as stream:
            stream.write('\n- ' + now() + ' ' + text + '\n')


def dependency_ready(root):
    """A live or partially accepted E10 never releases B."""
    root = Path(root)
    try:
        accepted = read(root / 'E10_acceptance.json')
        state = read(root / 'path_status.json')
        protocol = read(root / 'protocol.json')
        count = len(read(protocol['E10']['path_source'])['k_fractional'])
        expected = {'E4_initial', 'E4_joint54_endpoint'}
        if not (accepted.get('completed') and accepted.get('passed') and
                state.get('stage') == 'complete' and not state.get('active') and
                state.get('completed_batches') == 16 and set(accepted['states']) == expected):
            return False
        for name in expected:
            item = accepted['states'][name]
            scf = read(root / 'SCF' / name / 'accepted.json')
            if not (item['passed'] and item['points'] == count and item['bands'] == 56
                    and item['maximum_ritz_Ha'] <= 1e-7 and scf['passed']):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def material_spent(budget, material):
    return sum(float(a['GPU_hours']) for a in budget['attempts']
               if a['material'] == material and a.get('GPU_hours') is not None)


def forecast(protocol, budget, material, live_GPU_hours=0., label=None, current_output=None):
    """Count completed points once; include future models and reconstruction."""
    setting = protocol['settings'][material]
    rate = setting['point_rate_GPU_hours']
    progress = {'fitted': 0, 'PBE': 0}
    for a in budget['attempts']:
        if a['material'] != material:
            continue
        folder = Path(a['output'])
        try:
            points = read(folder / 'points.json')
        except (OSError, ValueError):
            points = []
        valid = [p for p in points if p['label'].startswith('eq_') and p['residual'] <= 1e-6]
        progress[a['model']] = max(progress[a['model']], len({p['label'] for p in valid}))
        for point in valid:
            rate = max(rate, float(point['seconds']) * len(a.get('gpus', range(8))) / 3600)
    remaining = 0
    for model in ['fitted', 'PBE']:
        key = material + '_' + model
        if key in budget.get('logical_results', {}):
            continue
        requested = int(label.split('_')[1]) + 1 if label and model == budget.get('active_model') else 0
        remaining += max(0, max(3, requested) - progress[model])
    # Current POINT may be partially done. Keeping it as a full future POINT
    # deliberately overestimates, rather than grants extra budget.
    used = material_spent(budget, material) + live_GPU_hours
    predicted = used + rate * remaining + (0.5 if remaining else 0.)
    return dict(time=now(), material=material, used_GPU_hours=used,
                forecast_GPU_hours=predicted, cap_GPU_hours=setting['upper_GPU_hours'],
                rate_GPU_hours_per_point=rate, complete_points=progress,
                remaining_planning_points=remaining,
                allowed=predicted <= setting['upper_GPU_hours'],
                conservative_no_partial_credit=True)


def pause_after_first_gaas_point(work, label, mask):
    """Called by the native worker only after its durable point/native gate."""
    import numpy as np
    work = Path(work)
    point = read(work / (label + '.json'))
    root = os.environ.get('D3FT_B_ROOT')
    if root and read(Path(root)/'GaAs_first_point_policy.json').get('automatic_continue_within_cap'):
        root=Path(root)
        if not (root/'GaAs_first_point_measurement.json').exists():
            raise ValueError('First-point measurement must be reported before automatic continuation')
        if not label.startswith('eq_') or point['residual']>1e-6:
            raise ValueError('Cannot continue a nonstationary GaAs pressure point')
        np.save(work/'mask.npy',np.asarray(mask))
        protocol=read(root/'protocol.json');budget=read(root/'budget.json')
        live=max(0.,__import__('time').time()-float(os.environ['D3FT_B_START_EPOCH']))*int(os.environ['D3FT_B_GPU_COUNT'])/3600
        decision=forecast(protocol,budget,'GaAs',live,label,work)
        dump(work/(label+'_continuation_budget.json'),decision)
        dump(root/'GaAs_continuation_budget.json',decision)
        if not decision['allowed']:
            dump(root/'NEEDS_USER_APPROVAL.json',dict(message='需用户批准：GaAs总预测超过200GPUh',**decision))
            raise SystemExit(75)
        return
    if label != 'eq_0' or point['residual'] > 1e-6:
        raise ValueError('Unexpected/unqualified GaAs measurement point')
    np.save(work / 'mask.npy', np.asarray(mask))
    dump(work / 'first_point_pause.json', dict(time=now(),
        first_point_complete=True, next_point_not_authorized=True,
        classification='user requested cost measurement pause, not physical failure'))
    print('GaAs first point complete; user budget confirmation required', flush=True)
    raise SystemExit(76)
