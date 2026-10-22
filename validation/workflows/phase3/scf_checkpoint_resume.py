"""Physical checkpoint continuation for an infrastructure-paused Phase 3 run.

Opt-in entry point only. Live workers and numerical kernels are unchanged.
Adam moments are unavailable: preserve the original schedule tail, disclose the
reset, and still require the original independent physical/numerical gates.
"""
import copy
import hashlib
import importlib
import json
import os
import pickle
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physical_config(config):
    """Ignore only execution paths, update count and restart bookkeeping."""
    c = copy.deepcopy(config)
    for key in ['crystal_file_path_path', 'epoch']:
        c.pop(key, None)
    for key in ['checkpoint', 'checkpoint_resume']:
        c.get('d3ft', {}).pop(key, None)
    return c


def discover(source):
    """Latest atomic state for each unfinished POINT; never an intent/tmp file."""
    source = Path(source)
    completed = {p['label'] for p in json.loads((source / 'points.json').read_text())} if (source / 'points.json').exists() else set()
    plans = {}
    for latest in source.glob('checkpoints/*/latest.json'):
        s = json.loads(latest.read_text())
        m = json.loads((latest.parent / 'model.json').read_text())
        label = Path(m['config']['crystal_file_path_path']).stem
        if label in completed or not s.get('complete'):
            continue
        prior = m['config'].get('d3ft', {}).get('checkpoint_resume', {})
        offset = int(prior.get('completed_steps', 0))
        total = int(prior.get('original_total_steps', m['config']['epoch']))
        done = min(total, offset + int(s['iteration']))
        if not 0 < done <= total:
            raise ValueError('Invalid cumulative checkpoint update count')
        plan = dict(label=label, checkpoint=str(latest.parent), state=s, model=m,
                    completed_steps=done, original_total_steps=total,
                    prefix_file=prior.get('history_prefix_file'))
        if label not in plans or (done, latest.stat().st_mtime) > plans[label]['order']:
            plan['order'] = (done, latest.stat().st_mtime)
            plans[label] = plan
    return plans


def history_prefix(plan):
    values = []
    if plan['prefix_file']:
        values = np.load(plan['prefix_file']).tolist()
    values += plan['state']['energy_history_Ha']
    values = values[:plan['completed_steps']]
    if len(values) != plan['completed_steps']:
        raise ValueError('Checkpoint energy history does not match update count')
    return values


def tail_schedule(original, config, offset, total, base=None):
    full = copy.deepcopy(config)
    full.epoch = total
    schedule = original(full, base=base)
    return (lambda count: schedule(count + offset)) if callable(schedule) else schedule


def install(worker, source, report_dir):
    """Wrap this future worker's calc; all completed POINTs use --reuse-from."""
    source, report_dir = Path(source), Path(report_dir)
    plans = discover(source)
    original_calc = worker.calc
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'resume_plans.json').write_text(json.dumps(plans, indent=2) + '\n')

    def resumed_calc(config, init_params=None, freq_mask=None):
        label = Path(config.crystal_file_path_path).stem
        if label not in plans:
            return original_calc(config, init_params=init_params, freq_mask=freq_mask)
        plan = plans.pop(label)
        checkpoint = Path(plan['checkpoint'])
        model, state = plan['model'], plan['state']
        old_config = copy.deepcopy(model['config'])
        # A previous read-only reconstruction used LR=0 solely for readout.
        prior = old_config.get('d3ft', {}).get('checkpoint_resume', {})
        if prior.get('read_only_reconstruction'):
            old_config['optimizer_args']['learning_rate'] = prior['original_learning_rate']
            old_config['d3ft']['optim'] = prior['original_optim']
        if physical_config(config.to_dict()) != physical_config(old_config):
            raise ValueError('Resume would change the frozen physical/numerical model')
        if Path(config.crystal_file_path_path).read_bytes() != Path(model['config']['crystal_file_path_path']).read_bytes():
            raise ValueError('Checkpoint POINT geometry differs')
        if config.d3ft.occupation.enabled or (config.d3ft.get('free_energy', {}).get('enabled', False)
                                              and float(config.smearing) > 0):
            raise ValueError('This registered recovery supports fixed occupations and no thermal schedule only')
        if sha(checkpoint / state['state_file']) != state['state_sha256']:
            raise ValueError('Atomic checkpoint state SHA mismatch')
        if sha(checkpoint / 'mask.npy') != model['mask_sha256']:
            raise ValueError('Atomic checkpoint Miller mask SHA mismatch')
        mask = np.load(checkpoint / 'mask.npy')
        if freq_mask is not None and not np.array_equal(mask, np.asarray(freq_mask)):
            raise ValueError('Resume cannot change Miller basis')
        with (checkpoint / state['state_file']).open('rb') as f:
            params = pickle.load(f)
        history = history_prefix(plan)
        offset, total = plan['completed_steps'], plan['original_total_steps']
        read_only = offset == total
        c = copy.deepcopy(config)
        original_rate, original_optim = c.optimizer_args.learning_rate, c.d3ft.optim.to_dict()
        c.epoch = 1 if read_only else total - offset
        prefix = report_dir / (label + '_history_prefix.npy')
        np.save(prefix, np.asarray(history))
        c.d3ft.checkpoint_resume = dict(completed_steps=offset, original_total_steps=total,
            history_prefix_file=str(prefix.resolve()), optimizer_moments_restored=False,
            read_only_reconstruction=read_only, original_learning_rate=original_rate,
            original_optim=original_optim, source=str(checkpoint))
        ground = importlib.import_module('d3ft.calc.ground_state')
        original_schedule = ground._learning_rate_schedule
        if read_only:
            c.optimizer_args.learning_rate = 0.
            c.d3ft.optim.lr_schedule = 'constant'
        else:
            ground._learning_rate_schedule = lambda cfg, base=None: tail_schedule(
                original_schedule, cfg, offset, total, base)
        event = dict(label=label, source=str(checkpoint), source_state_sha256=state['state_sha256'],
            completed_steps=offset, new_updates=0 if read_only else total-offset,
            read_only_reconstruction_updates=1 if read_only else 0,
            optimizer_moments_restored=False, learning_rate=f'original {total}-step schedule tail',
            no_completed_updates_recomputed=True)
        (report_dir / (label + '_resume.json')).write_text(json.dumps(event, indent=2) + '\n')
        print('SCF_PHYSICAL_RESUME ' + json.dumps(event), flush=True)
        try:
            out = original_calc(c, init_params=params, freq_mask=mask)
        finally:
            ground._learning_rate_schedule = original_schedule
        if not np.allclose(np.asarray(out.k_frac), model['k_fractional'], atol=2e-12, rtol=0):
            raise ValueError('Resume k points changed')
        if not np.allclose(np.asarray(out.crystal.cell_vectors), model['cell_bohr'], atol=2e-11, rtol=0):
            raise ValueError('Resume cell changed')
        if not np.allclose(np.asarray(out.crystal.positions), model['positions_bohr'], atol=2e-11, rtol=0):
            raise ValueError('Resume atomic coordinates changed')
        if read_only and (checkpoint / 'density.npy').exists():
            delta = float(np.max(abs(np.asarray(out.density_grid) - np.load(checkpoint / 'density.npy'))))
            event['read_only_density_max_abs_error'] = delta
            if delta > 1e-9:
                raise ValueError('Read-only reconstruction changed saved density')
        out.energy_history = history if read_only else history + list(out.energy_history)
        if len(out.energy_history) != total:
            raise ValueError(f'Resumed total update history differs from registered {total}')
        # Report derivatives at the original registered config, not a short schedule.
        out.config.epoch = total
        out.config.optimizer_args.learning_rate = original_rate
        out.config.d3ft.optim.update(original_optim)
        event['completed'] = True
        (report_dir / (label + '_resume.json')).write_text(json.dumps(event, indent=2) + '\n')
        return out

    worker.calc = resumed_calc


if __name__ == '__main__':
    import worker
    install(worker, os.environ['D3FT_RESUME_SOURCE'], os.environ['D3FT_RESUME_REPORT'])
    worker.main()
