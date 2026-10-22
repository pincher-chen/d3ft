"""Post-freeze numerical controls and external evaluation, never fitting.

Parameters and settings are fixed before the corresponding result is read.
Failures are recorded, never used to choose a new model or relax a gate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from gpu_probe_retry import install_gpu_probe_retry
from e1_three_parameter_queue import read, dump
from e7_shared_budget import SharedBudget
from run_identity import reserve_name
install_gpu_probe_retry()


def execution_target(job):
    target = job.get('execution_target', dict(jobid=250285, node='hn35'))
    pair = (target['jobid'], target['node'])
    if pair != (250285, 'hn35') and not (
        job['key'] == 'LOO_diamond_fine_validation' and pair == (250212, 'hn48')):
        raise ValueError('Only the explicitly authorized LOO-diamond may use hn48/250212')
    return target


def reservation_slot(gpu, node):
    return str(gpu) if node == 'hn35' else node + ':' + str(gpu)


def observed_exit_code(log, output, tag):
    text = Path(log).read_text(errors='replace')
    if (Path(output)/'summary.json').exists() and ('DONE '+tag) in text:
        return 0, 'worker DONE plus summary; inherited waitpid unavailable'
    codes = re.findall(r'exited with exit code\s+(\d+)', text, flags=re.I)
    if codes:
        return int(codes[-1]), 'explicit srun stderr; inherited waitpid unavailable'
    return 125, 'unknown infrastructure exit; no physical failure inference'


class DetachedWorker:
    """Observe an unchanged srun after CPU-only coordinator handover."""
    def __init__(self, record, log, tag):
        self.pid, self.output, self.log, self.tag = record['pid'], record['output'], log, tag
        self.returncode = None
        self.exit_evidence = None

    def poll(self):
        stat = Path('/proc')/str(self.pid)/'stat'
        try:
            if stat.read_text().rsplit(')',1)[1].split()[0] != 'Z':
                return None
        except (FileNotFoundError, ProcessLookupError):
            pass
        self.returncode, self.exit_evidence = observed_exit_code(self.log,self.output,self.tag)
        return self.returncode


def main(root, handover=None):
    c = read(root / 'protocol.json')
    if handover:
        c.update(read(handover))
    project, training = Path(c['project']), Path(c['training_root'])
    runtime = Path(c['runtime'])
    sys.path.insert(0, str(runtime / 'validation/workflows/phase3'))
    from frozen_parameters import freeze_result, canonical
    from equilibrium_jacobian import derive
    budget = SharedBudget(c['shared_budget_file'])
    registry = project / 'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
    jobs, active, completed = {}, {}, {}
    gap_guard = c.get('gap_guard')
    blocked = {}
    if gap_guard:
        from e7_gap_budget_guard import enforce, prepare_reuse, allowed_recovery, forecast, BudgetApprovalRequired
        gap_policy = Path(gap_guard['policy'])
        for key in ['external_GaP_fitted','external_GaP_PBE']:
            paused = root/(key+'_budget_pause.json')
            if paused.exists() and read(gap_policy).get('budget_enforcement') != 'record_only':
                blocked[key] = read(paused)

    def now():
        return dt.datetime.now().astimezone().isoformat()

    def note(message):
        for file in [project / 'response_design_to_do_list.md', project / 'paper/response_design_EXPERIMENTS_20261014.md']:
            with file.open('a') as f:
                f.write('\n- ' + now() + ' ' + message + '\n')

    migrations = c.get('migrations', {})

    def release(gpus, node='hn35'):
        with registry.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owners = read(registry)
            for gpu in gpus:
                slot = reservation_slot(gpu,node)
                if owners.get(slot, {}).get('controller_pid') == os.getpid():
                    owners.pop(slot)
            dump(registry, owners)

    def claim(job, memory):
        target = execution_target(job)
        node = target['node']
        if node != 'hn35':
            probe = subprocess.run(['srun','--jobid='+str(target['jobid']), '--overlap',
                '--cpu-bind=none','--nodelist='+node,'-N1','-n1','-c1','nvidia-smi',
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=45)
            if probe.returncode:
                return None
            memory = {int(a):int(b) for a,b in (line.split(',') for line in probe.stdout.splitlines())}
        if job.get('preferred_GPU') is not None:
            candidates = [job['preferred_GPU']]
        else:
            candidates = list(range(4 if node=='hn48' else 8))
        with registry.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owners = read(registry)
            free = [g for g in candidates if memory.get(g, 10**9) <= 3 and reservation_slot(g,node) not in owners]
            if len(free) < job['GPU_count']:
                return None
            gpus = free[:job['GPU_count']]
            for gpu in gpus:
                owners[reservation_slot(gpu,node)] = dict(owner='E7 frozen evaluation only',
                    controller_pid=os.getpid(),status_file=str(root/'status.json'),
                    node=node,jobid=target['jobid'],logical_job=job['key'])
            dump(registry, owners)
            return gpus

    def add(key, fit, material, kind, settings, GPU_count, preferred=None, after=None, vector=None):
        if key not in jobs:
            jobs[key] = dict(key=key, fit=fit, material=material, kind=kind, settings=settings,
                             GPU_count=GPU_count, preferred_GPU=preferred, after=after, vector=vector)
            if key in migrations:
                jobs[key]['execution_target'] = migrations[key]['to']
                jobs[key]['queue_entry_status'] = 'migrated_not_deleted'
                execution_target(jobs[key])

    def refresh():
        for fit in ['full', 'LOO_Si', 'LOO_BN', 'LOO_diamond']:
            folder = training / fit
            result = folder / (fit + '_result.json')
            if not result.exists():
                continue
            cert = folder / (fit + '_parameter_freeze.json')
            frozen = freeze_result(result, cert)
            result_data = read(result)
            if fit == 'full':
                for material in ['Si', 'BN', 'diamond']:
                    add('full_fine_' + material, fit, material, 'fine', c['fine_settings'][material],
                        4 if material == 'diamond' else 1, 6 if material in ['Si','BN'] else None,
                        'full_fine_Si' if material == 'BN' else None)
                for material in ['SiC', 'GaP']:
                    add('external_' + material + '_fitted', fit, material, 'external_a0', c['external_settings'][material],
                        1 if material == 'SiC' else 8, 7 if material == 'SiC' else None)
                    add('external_' + material + '_PBE', fit, material, 'external_a0', c['external_settings'][material],
                        1 if material == 'SiC' else 8, 7 if material == 'SiC' else None,
                        'external_' + material + '_fitted', vector=[0.] * 7)
            elif 'validation' in result_data:
                held = fit[4:]
                add(fit + '_fine_validation', fit, held, 'fine', c['fine_settings'][held],
                    4 if held == 'diamond' else 1)
            for job in jobs.values():
                if job['fit'] == fit:
                    job['certificate'] = str(cert)
                    if job['vector'] is None:
                        job['vector'] = frozen['vector']

    def launch(job, gpus):
        target = execution_target(job)
        tag = reserve_name(root, job['key'], output_parent=root / 'runs', log_parent=root)
        reuse = None
        if gap_guard and job['material']=='GaP':
            enforce(gap_policy,dict(key=job['key'],source=job.get('_resume_source'),before_launch=True),
                    Path(gap_guard['root'])/(tag+'_launch_forecast.json'))
            if job.get('_resume_source'):
                reuse = prepare_reuse(job['_resume_source'],Path(gap_guard['root'])/(tag+'_reuse'))
        budget.reserve(tag, 12 * len(gpus))
        settings, material = job['settings'], job['material']
        fit_result = read(training / job['fit'] / (job['fit'] + '_result.json'))
        if job['kind'] == 'fine':
            coarse = Path(fit_result['states'][material] if job['fit'] == 'full' else fit_result['validation']['state'])
            initial = read(coarse / 'summary.json')['a0_A']
        else:
            coarse = None
            initial = c['external_initial_a0_A'][material]
        certificate = job['certificate'] if job['vector'] != [0.] * 7 else c['PBE_certificate']
        if hashlib.sha256(canonical(job['vector'])).hexdigest() != read(certificate)['vector_sha256']:
            raise ValueError('Evaluation vector does not match immutable certificate')
        tmp = '/dev/shm/' + tag
        env = dict(os.environ)
        env.update(c['environment'])
        env.update(PYTHONPATH=str(runtime) + ':' + c['jrystal'],
                   D3FT_PHASE3_RESULTS_ROOT=str(root / 'runs'),
                   D3FT_EVALUATION_FREEZE_CERT=certificate,
                   D3FT_CHECKPOINT_ROOT=str(root / 'runs' / tag / 'checkpoints'),
                   D3FT_CHECKPOINT_INTERVAL='500', TMPDIR=tmp,
                   JAX_COMPILATION_CACHE_DIR=tmp + '/jax', PYTHONPYCACHEPREFIX=tmp + '/pycache')
        entry = runtime / 'validation/workflows/phase3/worker.py'
        if gap_guard and material=='GaP':
            entry = Path(gap_guard['worker_entry'])
            env.update(D3FT_ORIGINAL_PHASE3=str(runtime/'validation/workflows/phase3'),
                D3FT_GAP_GUARD_POLICY=str(gap_policy),D3FT_GAP_LOGICAL_KEY=job['key'])
            if reuse:
                env.update(D3FT_RESUME_SOURCE=str(reuse),
                    D3FT_RESUME_REPORT=str(Path(gap_guard['root'])/'resume_reports'/tag))
        command = ['srun', '--jobid='+str(target['jobid']), '--overlap', '--cpu-bind=none', '--nodelist='+target['node'],
                   '-N1', '-n1', '-c' + str(4 * len(gpus)), '--gpu-bind=none', 'bash', '-c',
                   'mkdir -p "$TMPDIR"; exec "$@"', 'E7-frozen-eval', 'env',
                   'CUDA_VISIBLE_DEVICES=' + ','.join(map(str,gpus)), 'timeout', '43200',
                   c['python'], '-u', str(entry),
                   '--material', material, '--vector', json.dumps(job['vector']), '--tag', tag,
                   '--mode', 'evaluate', '--bulk-only', '--grid', str(settings['grid']),
                   '--cutoff', str(settings['cutoff_Ha']), '--k', str(settings['k']),
                   '--epochs', str(settings.get('epochs',3000)), '--empty', '0',
                   '--xc-grid-factor', str(settings['xc_grid_factor']),
                   '--strain-step', str(settings['strain_step']), '--a0', str(initial), '--fused-report']
        if len(gpus) > 1:
            command += ['--kpool']
        if job['kind'] == 'external_a0':
            command += ['--equilibrium-only']
        elif material != 'diamond':
            command += ['--fixed-mask', str(coarse / 'mask.npy')]
        if reuse:
            command += ['--reuse-from',str(reuse)]
        stream = (root / (tag + '.log')).open('xb')
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                cwd=project, start_new_session=True)
        record = dict(time=now(), pid=proc.pid, jobid=target['jobid'], node=target['node'], gpus=gpus,
                      output=str(root / 'runs' / tag), command=command, status='running',
                      timeout_seconds=43200, checkpoint_interval=500, vector_sha256=read(certificate)['vector_sha256'],
                      frozen_certificate=certificate, evaluation_only=True, settings=settings)
        if reuse:
            record.update(checkpoint_resume_source=job['_resume_source'],reuse_staging=str(reuse),
                original_total_updates_per_point=9000,optimizer_moments_restored=False)
        dump(root / (tag + '_launch.json'), record)
        active[job['key']] = dict(job=job, proc=proc, stream=stream, record=record,
                                  tag=tag, started=time.monotonic(), coarse=coarse)
        if job['key'] in migrations:
            note('E7 LOO-diamond注册细验已唯一迁移并启动于hn48/250212，hn35条目标记迁移保留；12h/500步/原门/共享400GPUh不变，节点不同不可作hn35计时比较。')

    def finish(key, run):
        proc, record = run['proc'], run['record']
        run['stream'].close()
        hours = (time.monotonic() - run['started']) * len(record['gpus']) / 3600
        if isinstance(proc, DetachedWorker):
            record.update(inherited_waitpid_exit_code_unavailable=True,
                exit_code_evidence=proc.exit_evidence,
                GPU_hours_scope='original launch to observed finish, conservative CPU-handover upper bound')
        record.update(end=now(), returncode=proc.returncode, GPU_hours=hours,
                      status='completed' if proc.returncode == 0 else 'failed')
        dump(root / (run['tag'] + '_launch.json'), record)
        if 'GPU_hours' not in read(c['shared_budget_file'])['jobs'][run['tag']]:
            budget.finish(run['tag'], hours, record['status'])
        release(record['gpus'],record.get('node','hn35'))
        result = dict(time=now(), job=run['job'], record=record, passed=False,
                      parameters_changed_after_freeze=False, used_for_training=False)
        output = Path(record['output'])
        if gap_guard and run['job']['material']=='GaP' and proc.returncode in (124,75):
            if proc.returncode==124 and not allowed_recovery(record):
                raise RuntimeError('GaP timeout recovery requires unchanged hn35 eight-card authorization')
            result.update(classification='infrastructure_pause_exit124' if proc.returncode==124
                else 'budget_pause_needs_user_approval',physical_gate_failed=False,
                effective_acceptance_pending=True)
            dump(root/(run['tag']+'_attempt_acceptance.json'),result)
            if not (root/(key+'_acceptance.json')).exists():
                dump(root/(key+'_acceptance.json'),result)
            run['job']['_resume_source']=str(output)
            dump(root/(key+'_recovery_pending.json'),dict(time=now(),source=str(output),
                returncode=proc.returncode,only_124_authorizes_automatic_recovery=True))
            if proc.returncode==75:
                blocked[key]=result
                dump(root/(key+'_budget_pause.json'),result)
                print('需用户批准：'+key+'预算闸已暂停，原结果和检查点保留',flush=True)
            note('E7 '+key+' '+result['classification']+'；原attempt/POINT/500步检查点保留，'
                 '不是物理门失败；124仅在预算闸与原门不变条件下继续，75等待用户批准。')
            return
        if proc.returncode == 0:
            try:
                summary = read(output / 'summary.json')
                if run['job']['kind'] == 'fine':
                    derive(output)
                    coarse = read(run['coarse'] / 'summary.json')
                    ar = abs(coarse['a0_A'] - summary['a0_A']) / abs(summary['a0_A'])
                    bd = abs(coarse['B_GPa'] - summary['B_GPa'])
                    result.update(passed=ar < 1e-4 and bd < .1, a0_relative_error=ar,
                                  B_difference_GPa=bd, a0_gate=1e-4, B_gate_GPa=.1,
                                  coarse=str(run['coarse']), fine=str(output), native_gate_passed=True)
                else:
                    passed = summary['equilibrium_residual'] <= 1e-6 and summary['root_relative'] < 1e-5
                    target = c['external_reference_a0_A'][run['job']['material']]
                    result.update(passed=passed, a0_A=summary['a0_A'], reference_a0_A=target,
                                  relative_error=(summary['a0_A']-target)/target,
                                  reference_scope='local reference transcription, finite-temperature proxy; no new original-source certification',
                                  bulk_modulus_certified=False, native_equilibrium_gate_passed=passed)
                result['summary'] = summary
            except Exception as exc:
                result['error'] = repr(exc)
        else:
            result['error'] = 'Subprocess failed; inspect retained stderr; no model change or gate relaxation'
            stderr = (root / (run['tag'] + '.log')).read_text()
            oom = any(token in stderr.upper() for token in
                      ['RESOURCE_EXHAUSTED', 'OUT OF MEMORY', 'CUDA_ERROR_OUT_OF_MEMORY'])
            result['infrastructure_OOM'] = oom
            if oom and run['job']['material']=='SiC' and run['job']['GPU_count']==1:
                recovery = key + '_4gpu_recovery'
                jobs[recovery] = dict(run['job'], key=recovery, GPU_count=4, preferred_GPU=None, after=None)
                # Predeclared resource-only fallback: same frozen vector,
                # cutoff/grid/k/XC/tolerances, never a response to its error.
                if key == 'external_SiC_fitted':
                    jobs['external_SiC_PBE'].update(GPU_count=4, preferred_GPU=None, after=recovery)
        acceptance = key + ('_resolved_acceptance.json' if run['job'].get('_resume_source') else '_acceptance.json')
        dump(root / acceptance, result)
        completed[key] = result
        note('E7冻结后评估 ' + key + ' 完成，验收=' + str(result['passed']) +
             '，实测' + str(round(hours,3)) + 'GPU·h；不回传训练、不按结果选模型，失败保留。')

    if handover:
        dump(Path(handover).parent/'new_coordinator_ready.json',dict(time=now(),pid=os.getpid(),
            waiting_for_original_controller_lock=True,no_GPU_signals=True))
    with (root / 'controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX if handover else fcntl.LOCK_EX | fcntl.LOCK_NB)
        if c.get('adopt_existing'):
            refresh()
            previous = read(root/'status.json')
            for key in jobs:
                path = root/(key+'_acceptance.json')
                resolved = root/(key+'_resolved_acceptance.json')
                if resolved.exists():
                    path=resolved
                if path.exists():
                    accepted=read(path)
                    if gap_guard and key.startswith('external_GaP_') and accepted.get('effective_acceptance_pending'):
                        pending=read(root/(key+'_recovery_pending.json'))
                        jobs[key]['_resume_source']=pending['source']
                    else:
                        completed[key] = accepted
            for key, old in previous.get('active',{}).items():
                if key in completed:
                    continue
                if key in migrations:
                    raise RuntimeError('Cannot migrate a job already launched on hn35')
                tag = Path(old['output']).name
                record = read(root/(tag+'_launch.json'))
                job = jobs[key]
                fit_result = read(training/job['fit']/(job['fit']+'_result.json'))
                coarse = (Path(fit_result['states'][job['material']] if job['fit']=='full'
                    else fit_result['validation']['state']) if job['kind']=='fine' else None)
                elapsed = max(0,(dt.datetime.now().astimezone()-dt.datetime.fromisoformat(record['time'])).total_seconds())
                active[key] = dict(job=job,proc=DetachedWorker(record,root/(tag+'.log'),tag),
                    stream=(root/(tag+'.log')).open('ab'),record=record,tag=tag,
                    started=time.monotonic()-elapsed,coarse=coarse)
            with registry.with_name('gpu_reservations.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                owners = read(registry)
                for owner in owners.values():
                    if owner.get('controller_pid') == c['previous_controller_pid'] and owner.get('status_file') == str(root/'status.json'):
                        owner['controller_pid'] = os.getpid()
                dump(registry,owners)
            dump(Path(handover).parent/'adoption_acknowledged.json',dict(time=now(),pid=os.getpid(),
                adopted_active={k:dict(pid=r['record']['pid'],gpus=r['record']['gpus']) for k,r in active.items()},
                migrated=migrations,no_GPU_workers_restarted=True))
        while True:
            refresh()
            for key, run in list(active.items()):
                if run['proc'].poll() is not None:
                    finish(key, run)
                    active.pop(key)
            probe = subprocess.run(['srun', '--jobid=250285', '--overlap', '--cpu-bind=none',
                '--nodelist=hn35', '-N1', '-n1', '-c1', 'nvidia-smi',
                '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=45)
            memory = {int(v[0]):int(v[1]) for line in probe.stdout.splitlines()
                      if len(v:=line.split(','))==2} if probe.returncode==0 else {}
            for key, job in jobs.items():
                if key in active or key in completed or key in blocked:
                    continue
                if job['after'] and job['after'] not in completed:
                    continue
                gpus = claim(job, memory)
                if gpus is not None:
                    try:
                        launch(job,gpus)
                    except RuntimeError as exc:
                        release(gpus,execution_target(job)['node'])
                        if 'budget' in str(exc).lower() or (gap_guard and isinstance(exc,BudgetApprovalRequired)):
                            dump(root / (key + '_budget_wait.json'), dict(time=now(), error=str(exc),
                                no_GPU_launched=True, no_cap_relaxation=True))
                            if gap_guard and job['material']=='GaP':
                                blocked[key]=dict(time=now(),error=str(exc),message='需用户批准',
                                    no_GPU_launched=True,original_results_preserved=True)
                                dump(root/(key+'_budget_pause.json'),blocked[key])
                                print('需用户批准：'+key+' '+str(exc),flush=True)
                            continue
                        raise
            fits_done = all((training / f / (f+'_result.json')).exists() and
                            'validation' in read(training / f / (f+'_result.json'))
                            for f in ['LOO_Si','LOO_BN','LOO_diamond'])
            done = fits_done and len(completed)==len(jobs) and not active
            dump(root / 'status.json', dict(time=now(), controller_pid=os.getpid(),
                stage='completed_evaluation_only' if done else 'frozen_evaluation_running',
                jobid=250285, node='hn35', shared_budget=budget.summary(),
                active={k:r['record'] for k,r in active.items()}, pending=[k for k in jobs if k not in active and k not in completed and k not in migrations],
                completed={k:dict(passed=v['passed'],output=v['record']['output']) for k,v in completed.items()},
                migrated={k:dict(v,queue_entry_status='migrated_not_deleted',
                    execution_status='completed' if k in completed else 'running' if k in active else 'pending')
                    for k,v in migrations.items()},
                blocked_needs_user_approval=blocked,no_fit_or_tuning=True))
            if gap_guard:
                report=forecast(read(gap_policy))
                dump(Path(gap_guard['root'])/'status.json',dict(time=now(),controller_pid=os.getpid(),
                    stage='needs_user_approval' if blocked else 'observing_or_executing_exit124_recovery',
                    active_keys=list(active),blocked=blocked,forecast=report,
                    CPU_guard_only=True,original_GPU_worker_unchanged=True,
                    pending=[k for k in jobs if k not in completed and k not in active]))
            if done:
                dump(root / 'result.json', dict(completed=True, evaluations=completed,
                    shared_budget=budget.summary(), no_parameter_selection_or_refitting=True))
                return
            time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--handover',type=Path)
    args = parser.parse_args()
    main(args.root,args.handover)
