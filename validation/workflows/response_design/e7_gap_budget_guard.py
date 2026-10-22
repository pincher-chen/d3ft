"""CPU forecasting and opt-in GaP point gates. Never selects physical models.

Three pressure-root points/model is the disclosed planning forecast, not a
convergence guarantee. Increase it to include every newly requested point;
completed models use their actual cost. Both live elapsed time and recovery
overhead are counted, without adding reservations a second time.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil

KEYS = ('external_GaP_fitted', 'external_GaP_PBE')


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def now():
    return dt.datetime.now().astimezone().isoformat()


def allowed_recovery(record):
    return (record.get('returncode') == 124 and record.get('node') == 'hn35'
            and record.get('jobid') == 250285 and len(record.get('gpus', [])) == 8)


def progress(source):
    """Only durable, native-gate-passing POINTs count as reusable work."""
    if not source or not Path(source).exists():
        return dict(complete=0, partial=0., rate_hours=3.8, finished=False)
    source = Path(source)
    points = read(source/'points.json') if (source/'points.json').exists() else []
    points = [p for p in points if p['label'].startswith('eq_') and p['residual'] <= 1e-6]
    labels = {p['label'] for p in points}
    rate = max([3.8] + [p['seconds']/3600 for p in points])
    partial = 0.
    for latest in source.glob('checkpoints/*/latest.json'):
        state, model = read(latest), read(latest.parent/'model.json')
        label = Path(model['config']['crystal_file_path_path']).stem
        if label in labels or not state.get('complete'):
            continue
        prior = model['config'].get('d3ft', {}).get('checkpoint_resume', {})
        total = prior.get('original_total_steps', model['config']['epoch'])
        partial = max(partial, min(1., (prior.get('completed_steps', 0)+state['iteration'])/total))
        # Use actual checkpoint cadence too, when it is slower than POINT rate.
        states = sorted(latest.parent.glob('state_*.json'))
        # Production checkpoint files are pickle+latest; no reliance on this
        # optional history for correctness or on unvalidated density contents.
        if len(states) >= 2:
            x, y = read(states[-2]), read(states[-1])
            delta = y['iteration']-x['iteration']
            if delta > 0:
                elapsed = states[-1].stat().st_mtime-states[-2].stat().st_mtime
                rate = max(rate, elapsed/delta*total/3600)
    return dict(complete=len(labels), partial=partial, rate_hours=rate,
                finished=(source/'summary.json').exists())


def forecast(policy, context=None, current_time=None):
    """Forecast includes all ledger spending and unfinished logical GaP work."""
    context = context or {}
    evaluation = Path(policy['evaluation'])
    ledger = read(policy['budget_file'])
    instant = current_time or dt.datetime.now().astimezone()
    used = float(ledger['initial_GPU_hours'])
    for tag, row in ledger['jobs'].items():
        if 'GPU_hours' in row:
            used += row['GPU_hours']
        else:
            launch = evaluation/(tag+'_launch.json')
            if not launch.exists():
                # Reservation without durable start evidence: fail conservatively.
                used += row['reserved_GPU_hours']
            else:
                rec = read(launch)
                elapsed = max(0., (instant-dt.datetime.fromisoformat(rec['time'])).total_seconds())
                used += elapsed*len(rec['gpus'])/3600
    rows = {}
    for key in KEYS:
        launches = []
        for path in evaluation.glob(key+'__*_launch.json'):
            rec = read(path)
            launches.append((rec['time'], rec))
        launches.sort(key=lambda x: x[0])
        source = launches[-1][1]['output'] if launches else None
        if context.get('key') == key:
            source = context.get('source') or source
        p = progress(source)
        # A physical failure is terminal; only 124 and explicit budget pause
        # are infrastructure continuations. Neither may masquerade as PASS.
        terminal_failure = bool(launches and launches[-1][1].get('returncode') not in (None,0,75,124))
        requested = (int(context['label'].split('_')[1])+1
                     if context.get('key') == key and context.get('label') else 0)
        target = max(policy['planning_points_per_model'], p['complete']+int(p['partial']>0), requested)
        if p['finished'] or terminal_failure:
            remaining = 0.
        else:
            remaining = max(0., target-p['complete']-p['partial'])*p['rate_hours']*8
        rows[key] = dict(source=source, **p, forecast_total_points=target,
                         remaining_GPU_hours=remaining, terminal_failure=terminal_failure)
    # Include reconstruction/JIT overhead once for a newly launched continuation
    # or future PBE worker; this allowance is counted before launch, not hidden.
    overhead = policy['launch_overhead_GPU_hours'] if context.get('before_launch') else 0.
    predicted = used + sum(row['remaining_GPU_hours'] for row in rows.values()) + overhead
    record_only = (ledger.get('budget_enforcement') == 'record_only'
                   and policy.get('budget_enforcement') == 'record_only')
    cap = ledger['cap_GPU_hours']
    if cap is None and not record_only:
        raise RuntimeError('Unbounded budget requires matching record-only ledger and GaP policy authorization')
    reference = ledger.get('reference_cap_GPU_hours',cap)
    return dict(time=instant.isoformat(), used_GPU_hours=used,
                forecast_total_GPU_hours=predicted, cap_GPU_hours=cap,
                budget_enforcement='record_only' if record_only else 'enforce',
                reference_cap_GPU_hours=reference,
                reference_cap_exceeded=reference is not None and predicted>reference,
                allowed=record_only or predicted <= cap, models=rows,
                launch_overhead_GPU_hours=overhead,
                forecast_is_not_a_convergence_guarantee=True)


class BudgetApprovalRequired(RuntimeError):
    pass


def enforce(policy_file, context, report):
    result = forecast(read(policy_file), context)
    result['context'] = context
    dump(report, result)
    if result['budget_enforcement']=='record_only':
        print('E7_BUDGET_RECORD_ONLY '+json.dumps(dict(
            forecast_total_GPU_hours=result['forecast_total_GPU_hours'],
            used_GPU_hours=result['used_GPU_hours'],reference_cap_GPU_hours=result['reference_cap_GPU_hours'],
            reference_cap_exceeded=result['reference_cap_exceeded'],blocking=False)),flush=True)
    if not result['allowed']:
        message = ('需用户批准：E7预计总量 %.3f > %.3f GPU·h；保持现场，不启动新GaP计算'
                   % (result['forecast_total_GPU_hours'], result['cap_GPU_hours']))
        result.update(message=message, no_GPU_work_started=True)
        dump(Path(report).parent/'NEEDS_USER_APPROVAL.json', result)
        print(message, flush=True)
        raise BudgetApprovalRequired(message)
    return result


def prepare_reuse(source, destination):
    """New staging directory only; never edit old results/checkpoints.

    The original root search saves mask.npy only after convergence. Recover a
    missing reuse mask from the last completed POINT's own validated checkpoint.
    Unfinished points still restore their own exact checkpoint Miller mask.
    """
    from scf_checkpoint_resume import discover, history_prefix, sha
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.name == 'mask.npy':
            continue
        (destination/item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    points = read(source/'points.json') if (source/'points.json').exists() else []
    if any(p['residual'] > 1e-6 for p in points):
        raise ValueError('Cannot reuse a physical stationarity failure')
    if (source/'mask.npy').exists():
        shutil.copyfile(source/'mask.npy', destination/'mask.npy')
    elif points:
        label = points[-1]['label']
        candidates = []
        for model_file in source.glob('checkpoints/*/model.json'):
            model = read(model_file)
            if Path(model['config']['crystal_file_path_path']).stem == label:
                mask = model_file.parent/'mask.npy'
                if sha(mask) != model['mask_sha256']:
                    raise ValueError('Completed POINT checkpoint mask SHA mismatch')
                candidates.append(mask)
        if not candidates:
            raise ValueError('No validated mask for reuse of completed POINTs')
        shutil.copyfile(candidates[-1], destination/'mask.npy')
    plans = discover(source)
    for plan in plans.values():
        checkpoint = Path(plan['checkpoint'])
        if sha(checkpoint/plan['state']['state_file']) != plan['state']['state_sha256']:
            raise ValueError('Resume state SHA mismatch before GPU launch')
        history_prefix(plan)
        if plan['original_total_steps'] != 9000:
            raise ValueError('GaP checkpoint must retain 9000 original steps')
    dump(destination/'staging_provenance.json', dict(time=now(),source=str(source),
         reused_points=[p['label'] for p in points],resume_labels=list(plans),
         old_results_unchanged=True))
    return destination


def install(worker, policy_file, key, source=None):
    """Before every future, non-reused POINT, test the shared forecast."""
    original = worker.calc
    if source:
        import scf_checkpoint_resume
        scf_checkpoint_resume.install(worker, source, os.environ['D3FT_RESUME_REPORT'])
        original = worker.calc

    def guarded(config, init_params=None, freq_mask=None):
        xyz = Path(config.crystal_file_path_path)
        if source and (Path(source)/'mask.npy').exists() and not (xyz.parent/'mask.npy').exists():
            shutil.copyfile(Path(source)/'mask.npy',xyz.parent/'mask.npy')
        # Completed earlier POINTs have already been copied by worker before
        # calc is called. Use the source progress if it contains a partial SCF.
        context_source = str(xyz.parent)
        if source and progress(source)['complete'] >= progress(context_source)['complete']:
            context_source = str(source)
        try:
            enforce(policy_file, dict(key=key,source=context_source,label=xyz.stem),
                    xyz.parent/(xyz.stem+'_budget_forecast.json'))
        except BudgetApprovalRequired:
            raise SystemExit(75)
        out = original(config, init_params=init_params, freq_mask=freq_mask)
        import numpy as np
        np.save(xyz.parent/'mask.npy', np.asarray(out.freq_mask))
        return out
    worker.calc = guarded
