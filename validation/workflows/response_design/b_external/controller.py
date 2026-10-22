"""Detached CPU queue: accepted E10 -> eight-card AlP -> accepted AlP -> GaAs."""
import argparse
import datetime as dt
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

from policy import dependency_ready, dump, forecast, material_spent, note, now, read, sha


def prepare_reuse(source, destination, expected_steps, phase3):
    """Stage immutable old results; restore only incomplete checkpoint POINTs."""
    sys.path.insert(0, str(phase3))
    from scf_checkpoint_resume import discover, history_prefix
    source, destination = Path(source), Path(destination)
    plans = discover(source)
    points = read(source / 'points.json') if (source / 'points.json').exists() else []
    if any(p['residual'] > 1e-6 for p in points):
        raise ValueError('Cannot recover a physical stationarity failure')
    for plan in plans.values():
        checkpoint = Path(plan['checkpoint'])
        if plan['original_total_steps'] != expected_steps:
            raise ValueError('Recovery would change registered update count')
        if sha(checkpoint / plan['state']['state_file']) != plan['state']['state_sha256']:
            raise ValueError('Checkpoint SHA mismatch')
        history_prefix(plan)
    if not plans and not points and not (source / 'summary.json').exists():
        raise ValueError('Timeout has no complete recoverable checkpoint')
    destination.mkdir(exist_ok=False)
    for p in source.iterdir():
        if p.name != 'mask.npy':
            (destination / p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
    if (source / 'mask.npy').exists():
        shutil.copy2(source / 'mask.npy', destination / 'mask.npy')
    elif points:
        last = points[-1]['label']
        found = []
        for m in source.glob('checkpoints/*/model.json'):
            model = read(m)
            if Path(model['config']['crystal_file_path_path']).stem == last:
                if sha(m.parent / 'mask.npy') != model['mask_sha256']:
                    raise ValueError('Completed POINT Miller-mask SHA mismatch')
                found.append(m.parent / 'mask.npy')
        if not found:
            raise ValueError('No validated mask for completed POINT reuse')
        shutil.copy2(found[-1], destination / 'mask.npy')
    dump(destination / 'staging_provenance.json', dict(
        time=now(), source=str(source), resume_labels=list(plans),
        reused_points=[p['label'] for p in points], original_total_steps=expected_steps,
        optimizer_moments_restored=False, old_results_unchanged=True))
    return destination


def idle_eight_cards(gpus=tuple(range(8))):
    """Probe only after E10 acceptance; unknown/timeout is never idle."""
    command = ['srun', '--jobid=250285', '--overlap', '--nodelist=hn35',
               '--cpu-bind=none', '-N1', '-n1', '-c1', 'nvidia-smi',
               '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']
    try:
        q = subprocess.run(command, capture_output=True, text=True, timeout=45)
        if q.returncode:
            return False
        memory = {int(a):int(b) for a,b in (line.split(',') for line in q.stdout.splitlines())}
        return all(i in memory and memory[i] <= 64 for i in gpus)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False


def validate_result(output, protocol, material, model):
    summary = read(output / 'summary.json')
    meta = read(output / 'metadata.json')
    setting = protocol['settings'][material]
    vector = protocol['fitted_vector'] if model == 'fitted' else [0.] * 7
    if meta['parameter_vector'] != vector or summary['parameter_vector'] != vector:
        raise ValueError('Frozen evaluation vector mismatch')
    if meta['pseudopotentials'] != protocol['pseudopotentials'][material]:
        raise ValueError('Frozen SG15 pseudopotential mismatch')
    passed = (summary['equilibrium_residual'] <= protocol['electronic_residual_tolerance']
              and summary['root_relative'] < protocol['root_relative_tolerance'])
    target = setting['reference_a0_A']
    return dict(time=now(), material=material, model=model, passed=bool(passed),
                a0_A=summary['a0_A'], reference_a0_A=target,
                relative_error_percent=100 * (summary['a0_A'] / target - 1),
                reference_scope=setting['reference_scope'], summary=summary,
                used_for_fitting=False, parameters_changed=False, bulk_modulus_certified=False)


class AdoptedChild:
    """Read-only observation of the same srun after CPU-coordinator handover."""
    def __init__(self, record):
        self.record = record
        self.pid = record['pid']
    def poll(self):
        try:
            if Path('/proc',str(self.pid),'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z':
                return None
        except (OSError,IndexError):
            pass
        log = Path(self.record['run'],'stdout.log').read_text(errors='replace')
        if (Path(self.record['output'])/'summary.json').exists() and 'DONE ' in log:
            return 0
        codes = re.findall(r'Exited with exit code\s+(\d+)',log,flags=re.I)
        return int(codes[-1]) if codes else 125


def _run(root):
    root = root.resolve()
    protocol = read(root / 'protocol.json')
    if protocol['node'] != 'hn35' or protocol['jobid'] != 250285:
        raise ValueError('Unauthorized resource')
    if protocol['material_order'] != ['AlP', 'GaAs']:
        raise ValueError('Unapproved material scope')
    budget = read(root / 'budget.json')
    budget.setdefault('logical_results', {})
    # Never duplicate a possibly live child after an unexpected CPU restart.
    unsettled = [a for a in budget['attempts'] if a.get('GPU_hours') is None]
    active = None
    if unsettled:
        handover = read(root / 'CPU_handover_authorization.json')
        if len(unsettled) != 1 or unsettled[0]['pid'] != handover['child_pid']:
            raise RuntimeError('Unsettled child does not match authorized CPU handover')
        active = dict(proc=AdoptedChild(unsettled[0]),stream=open(os.devnull,'w'),record=unsettled[0])
    dump(root / 'budget.json', budget)
    resume_source = None
    authorization = root / 'resume_authorization.json'
    if authorization.exists() and not (root / 'resume_authorization_consumed.json').exists():
        instruction = read(authorization)
        prior = budget['attempts'][-1]
        if not (instruction['user_approved'] and instruction['source'] == prior['output']
                and prior['returncode'] == 75 and prior['material'] == 'AlP'):
            raise ValueError('Invalid explicit budget-pause recovery authorization')
        resume_source = instruction['source']
    expected_code = read(root / 'execution_SHA.json')

    def status(stage, **extra):
        spent = {m:material_spent(budget, m) for m in protocol['material_order']}
        if active:
            spent[active['record']['material']] += (time.time()-active['record']['start_epoch']) * len(active['record']['gpus'])/3600
        dump(root / 'status.json', dict(time=now(), stage=stage, controller_pid=os.getpid(),
            jobid=250285, node='hn35', used_GPU_hours=spent,
            active=None if not active else active['record'],
            logical_results=budget['logical_results'], **extra))

    def pause(reason, **extra):
        dump(root / 'NEEDS_USER_APPROVAL.json', dict(time=now(), message=reason, **extra))
        status('paused_needs_user_report', reason=reason)
        note(protocol['project'], 'B暂停：' + reason + '；未续提、不改门，现场保留。')

    status('waiting_E10_complete_and_accepted_no_GPU_submitted')
    while True:
        if active:
            rc = active['proc'].poll()
            if rc is None:
                status('B_running')
                time.sleep(20)
                continue
            active['stream'].close()
            record = active['record']
            record.update(end=now(), returncode=rc,
                          GPU_hours=(time.time()-record['start_epoch'])*len(record['gpus'])/3600)
            dump(Path(record['run']) / 'launch.json', record)
            key = record['material'] + '_' + record['model']
            material, model = record['material'], record['model']
            active = None
            dump(root / 'budget.json', budget)
            if material == 'AlP' and material_spent(budget, material) >= protocol['settings'][material]['upper_GPU_hours']:
                pause('需用户批准：B-' + material + '累计机时已达草案区间上限')
                return
            if rc == 76 and material == 'GaAs':
                points = read(Path(record['output']) / 'points.json')
                point = next(p for p in points if p['label'] == 'eq_0')
                if point['residual'] > 1e-6:
                    raise ValueError('GaAs first point is not stationary')
                measured = material_spent(budget, 'GaAs')
                report = dict(time=now(), material='GaAs', model=model,
                    first_point_only=True, next_point_not_authorized=True,
                    point=point, point_solver_GPU_hours=point['seconds']*len(record['gpus'])/3600,
                    GPUs=record['gpus'], device_count=len(record['gpus']),
                    actual_allocation_GPU_hours=measured,
                    forecast_two_models_2_to_4_points_each_GPU_hours=[measured*4,measured*8],
                    forecast_three_points_per_model_GPU_hours=measured*6,
                    forecast_max_seven_points_each_GPU_hours=measured*14,
                    original_280_cap_not_used_for_release=True,
                    nonzero_parameter_result_not_used_for_refitting=True)
                cap_policy=read(root/'GaAs_first_point_policy.json')
                if cap_policy.get('automatic_continue_within_cap'):
                    decision=forecast(protocol,budget,'GaAs')
                    report.update(first_point_only=False,next_point_not_authorized=not decision['allowed'],
                        cap_GPU_hours=protocol['settings']['GaAs']['upper_GPU_hours'],
                        continuation_forecast=decision,
                        forecast_rule='Existing conservative >=3 points/model, actual rate with original30GPUh floor; reevaluated before every point',
                        continuation_authorized=decision['allowed'])
                    dump(root/'GaAs_first_point_measurement.json',report)
                    note(protocol['project'],f"GaAs首点实测报告：a={point['a0_A']:.9f}Å、"
                         f"P={point['pressure_GPa']:.6g}GPa、驻点残差={point['residual']:.3g}，"
                         f"持卡实耗{measured:.6f}GPUh，原三点/模型规划总预测"
                         f"{decision['forecast_GPU_hours']:.2f}/200GPUh；"
                         f"2–4点/模型范围{measured*4:.2f}–{measured*8:.2f}GPUh。"
                         '仅首压力点、不是平衡a0；预测≤200才自动复用首点续算，每点重新检查预算。')
                    if not decision['allowed']:
                        pause('需用户批准：GaAs首点后总预测超过200GPUh',forecast=decision)
                        return
                    resume_source=record['output']
                    status('first_point_reported_continuation_within_200',measurement=report)
                    continue
                dump(root / 'GaAs_first_point_measurement.json', report)
                status('paused_GaAs_first_point_awaiting_user_budget', measurement=report)
                note(protocol['project'], f"GaAs首个压力点完成后按用户决定暂停：a={point['a0_A']:.9f}Å，"
                     f"P={point['pressure_GPa']:.6g}GPa，驻点残差={point['residual']:.3g}；"
                     f"含初始化/恢复实测{measured:.6f}GPUh，按每模型2–4点预估两模型"
                     f"{measured*4:.2f}–{measured*8:.2f}GPUh（3+3点{measured*6:.2f}）。"
                     '仅首压力点，非已收敛a0；等待用户确认上限，280不作继续放行依据。')
                return
            if rc == 75:
                pause('需用户批准：B-' + material + '预测机时/检查点剩余额度触发停止')
                return
            if rc == 124:
                resume_source = record['output']
                note(protocol['project'], f'B {key} exit124基础设施暂停，原检查点保留，累计'
                     f'{material_spent(budget, material):.6f}GPUh；仅从最近500步完整检查点新唯一名续算。')
                status('timeout_recovery_pending')
                continue
            resume_source = None
            if rc == 0:
                try:
                    result = validate_result(Path(record['output']), protocol, material, model)
                except Exception as exc:
                    result = dict(time=now(), passed=False, error=repr(exc), material=material, model=model)
            else:
                result = dict(time=now(), passed=False, material=material, model=model,
                              returncode=rc, error='Non-timeout failure retained; no automatic retry',
                              classification=('memory short-test/infrastructure failure; not a physics result'
                                              if material=='GaAs' else 'unclassified non-timeout failure'))
            result.update(output=record['output'], actual_GPU_hours=sum(
                a['GPU_hours'] for a in budget['attempts'] if a['material']==material and a['model']==model))
            budget['logical_results'][key] = result
            dump(root / (key + '_acceptance.json'), result)
            dump(root / 'budget.json', budget)
            text = ('B ' + key + ' 完成，验收=' + str(result['passed']) +
                    '，实际GPUh=' + format(result['actual_GPU_hours'], '.6f'))
            if 'a0_A' in result:
                text += (f"，a0={result['a0_A']:.9f}Å，参考={result['reference_a0_A']}Å，"
                         f"误差={result['relative_error_percent']:+.6f}%（{result['reference_scope']}）")
            note(protocol['project'], text + '；冻结评估，不回传拟合、不改正式稿。')
            print(text, flush=True)
            continue

        jobs = [(m, v) for m in protocol['material_order'] for v in protocol['model_order']
                if m+'_'+v not in budget['logical_results']]
        if not jobs:
            status('complete', all_passed=all(r['passed'] for r in budget['logical_results'].values()))
            note(protocol['project'], 'B批准范围全部结束，结果及实际GPUh见B独立账本；AlAs/C均未执行。')
            return
        material, model = jobs[0]
        if material == 'GaAs' and not all(
                budget['logical_results'].get('AlP_'+v, {}).get('passed') for v in ['fitted', 'PBE']):
            status('stopped_AlP_not_passed_GaAs_not_submitted')
            note(protocol['project'], 'B AlP未全部通过，按依赖条件不提交GaAs；保留结果等待用户。')
            return
        priority = root / 'E9_priority_policy.json'
        resources = read(root/'resource_override.json') if (root/'resource_override.json').exists() else {}
        gpus = resources.get(material, {}).get('gpus', list(range(8)))
        concurrent = resources.get(material, {}).get('concurrent_E9_authorized', False)
        if material == 'GaAs' and concurrent and gpus != [1,2,3,4,5,6]:
            raise ValueError('Concurrent GaAs must use exactly physical GPUs1–6')
        if material == 'GaAs' and priority.exists() and not concurrent:
            release = Path(read(priority)['E9_release_file'])
            if not release.exists():
                status('waiting_E9_priority_no_GaAs_GPU_submitted')
                time.sleep(30)
                continue
        if not dependency_ready(protocol['E10_root']):
            status('waiting_E10_complete_and_accepted_no_GPU_submitted')
            time.sleep(30)
            continue
        if not idle_eight_cards(gpus):
            status('waiting_hn35_assigned_cards_free', requested_gpus=gpus)
            time.sleep(60)
            continue
        for path, expected in expected_code.items():
            if sha(path) != expected:
                raise ValueError('Frozen B execution source changed: ' + path)
        budget['active_model'] = model
        predicted = forecast(protocol, budget, material)
        dump(root / 'prelaunch_budget_forecast.json', predicted)
        if material == 'GaAs':
            if not (root / 'GaAs_first_point_policy.json').exists():
                raise ValueError('GaAs requires first-point-only user policy')
            cap_policy=read(root/'GaAs_first_point_policy.json')
            if not cap_policy.get('automatic_continue_within_cap'):
                predicted.update(allowed=True, authorization='one first pressure point only',
                                 original_280_cap_not_used_for_release=True)
            else:
                predicted.update(authorization='Frozen fitted and PBE within user-approved200GPUh; report first point before continuing',
                                 original_280_cap_not_used_for_release=True)
            dump(root / 'prelaunch_budget_forecast.json', predicted)
        if not predicted['allowed']:
            pause('需用户批准：B-' + material + '后续点预测超过草案机时上限', forecast=predicted)
            return
        tag = material+'_'+model+'__'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run = root / 'runs' / tag
        run.mkdir(parents=True, exist_ok=False)
        # Logs/launch records live in run; only worker creates its output.
        output = run / 'worker_results' / tag
        runtime = Path(protocol['runtime'])
        reuse = None
        if resume_source:
            reuse = prepare_reuse(resume_source, run / 'reuse', protocol['settings'][material]['epochs'],
                                  runtime/'validation/workflows/phase3')
        setting = protocol['settings'][material]
        env = dict(os.environ)
        env.update(protocol['environment'])
        temp = '/dev/shm/B_' + tag
        started = time.time()
        env.update(PYTHONPATH=str(runtime)+':'+protocol['jrystal'], TMPDIR=temp,
            PYTHONPYCACHEPREFIX=temp+'/pycache', JAX_COMPILATION_CACHE_DIR=temp+'/jax',
            D3FT_PHASE3_RESULTS_ROOT=str(output.parent), D3FT_CHECKPOINT_ROOT=str(output/'checkpoints'),
            D3FT_CHECKPOINT_INTERVAL='500', D3FT_B_ROOT=str(root), D3FT_B_OUTPUT=str(output),
            D3FT_B_MATERIAL=material, D3FT_B_MODEL=model, D3FT_B_START_EPOCH=str(started),
            D3FT_B_GPU_COUNT=str(len(gpus)), D3FT_B_PHYSICAL_GPUS=','.join(map(str,gpus)),
            D3FT_EVALUATION_FREEZE_CERT=str(root/('full_parameter_freeze.json' if model=='fitted' else 'PBE_parameter_freeze.json')))
        # No inherited restart/other-project yield settings may affect B.
        for key in ['D3FT_RESUME_SOURCE','D3FT_RESUME_REPORT','D3FT_YIELD_REQUEST','D3FT_GAP_GUARD_POLICY']:
            env.pop(key, None)
        if reuse:
            env.update(D3FT_RESUME_SOURCE=str(reuse), D3FT_RESUME_REPORT=str(run/'resume_reports'))
        if material == 'GaAs':
            env['D3FT_YIELD_REQUEST'] = str(root / 'GaAs_first_point_policy.json')
        vector = protocol['fitted_vector'] if model == 'fitted' else [0.]*7
        command = ['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none',
            '--gpu-bind=none','-N1','-n1','-c32','bash','-c','mkdir -p "$TMPDIR"; exec "$@"',
            'B-frozen-eval','env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)),'timeout',
            '--signal=TERM','--kill-after=30s','43200',protocol['python'],'-u',
            str(Path(__file__).resolve().parent/'worker_entry.py'),'--material',material,'--vector',json.dumps(vector),
            '--tag',tag,'--mode','evaluate','--bulk-only','--equilibrium-only','--grid',str(setting['grid']),
            '--cutoff',str(setting['cutoff_Ha']),'--k',str(setting['k']),'--epochs',str(setting['epochs']),
            '--empty','0','--xc-grid-factor',str(setting['xc_grid_factor']),
            '--strain-step',str(setting['strain_step']),'--a0',str(setting['a0_A']),
            '--fused-report','--kpool']
        if reuse:
            command += ['--reuse-from',str(reuse)]
        record = dict(time=now(), start_epoch=started, material=material, model=model,
                      jobid=250285, node='hn35', gpus=gpus, command=command,
                      run=str(run), output=str(output), GPU_hours=None, resume_source=resume_source)
        # The durable intent and both markdown timestamps precede the srun.
        dump(run/'launch_intent.json', record)
        note(protocol['project'], 'B '+material+'_'+model+' 已批准，GPU提交时间='+record['time']+
             '；E10两态路径完整验收且指定卡空闲，提交hn35/250285 GPUs='+str(gpus)+
             '，12h/500步；原物理参数/门不变，GaAs先显存门，首点报告后按最新已批准预算策略继续或暂停。')
        stream = (run/'stdout.log').open('xb')
        proc = subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                                stderr=subprocess.STDOUT,cwd=protocol['project'],start_new_session=True)
        record['pid'] = proc.pid
        if resume_source and authorization.exists() and not (root / 'resume_authorization_consumed.json').exists():
            dump(root / 'resume_authorization_consumed.json', dict(time=now(),
                source=resume_source, new_run=str(run), srun_pid=proc.pid))
        budget['attempts'].append(record)
        dump(run/'launch.json',record)
        dump(root/'budget.json',budget)
        active = dict(proc=proc, stream=stream, record=record)
        status('B_running')


def main(root):
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root)
    except Exception as exc:
        dump(args.root/'controller_error.json',dict(time=now(),error=repr(exc),automatic_relaunch=False))
        raise
