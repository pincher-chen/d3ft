"""Fresh training-only eq/iso costs; yield between stages to E3/E5/E4.

Two cheap streams use cards 0/1. Larger independent fine controls wait for
four genuinely idle cards, never displace another task. Historical results
provide only an initial lattice, never an independent acceptance certificate.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime, hashlib, json, os, subprocess, sys, time
from run_identity import reserve_name

P = Path(__file__).resolve().parents[3]
E = P / 'validation/results/response_design_20261015_e7_ct_pilot'
PROTO = json.loads((E / 'protocol.json').read_text())
R = Path(PROTO['runtime'])
PY = '/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
sys.path.insert(0, str(R / 'validation/workflows/phase3'))
from equilibrium_jacobian import derive

def now():
    return datetime.datetime.now().isoformat()

def dump(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def note(text):
    for path in [P / 'paper/response_design_EXPERIMENTS_20261014.md', P / 'response_design_to_do_list.md']:
        with path.open('a') as f:
            f.write('\n- ' + now() + ' ' + text + '\n')

def load(path):
    return json.loads(path.read_text()) if path.exists() else {}

def priority_pending():
    g = P / 'validation/results/response_design_20261015_e3_geometry'
    mode = load(g / 'mg_mode_acceptance.json')
    fd = load(g / 'mg_fd_acceptance.json')
    fs = load(g / 'mg_fd_queue_status.json')
    if mode.get('passed') and not fd.get('passed') and fs.get('stage') != 'failed':
        return 'E3 mixed FD'
    q = load(P / 'validation/results/response_design_20261015_e5_distributed/queue_status.json')
    if fd.get('passed') and q.get('stage') not in ('failed',):
        # The E5/E4 controller updates its stage through both pilots. Yield
        # until terminal status, including the post-E5 E4 stage.
        if not any(s in q.get('stage', '').lower() for s in ('complete', 'failed')):
            return 'E5/E4 priority'
        if q.get('stage') == 'Mg_k2_response_pilot_complete':
            e4 = P / 'validation/results/response_design_20261015_e4_joint_design'
            launch = load(e4 / 'production_mg_v1_launch.json')
            if not (e4 / 'production_acceptance.json').exists() and 'returncode' not in launch:
                return 'E4 priority'
    return None

def available(gpus):
    # A staged owner keeps its card during JAX imports and between spectrum
    # steps, when zero device memory alone is not evidence of availability.
    reservations = load(E / 'gpu_reservations.json')
    for key, owner in reservations.items():
        terminal = load(Path(owner['status_file'])).get('stage', '')
        if int(key) in gpus and terminal not in ('failed', 'k2_hydro_pilot_finished_k4_literature_pending'):
            return False
    cmd = ['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-N1', '-n1',
           '-c1', 'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
           '--format=csv,noheader,nounits']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError('GPU probe failed: ' + r.stderr)
    rows = {int(v[0]): (int(v[1]), int(v[2])) for line in r.stdout.splitlines()
            if len(v := line.split(',')) == 3}
    return all(rows[g][0] <= 3 and rows[g][1] == 0 for g in gpus)

def spent():
    total = 0.
    for p in E.glob('*_launch.json'):
        d = load(p)
        total += d.get('GPU_hours', 0.)
    return total

def selected_tag(material, kind):
    primary = load(E / 'primary_runs.json')
    if kind in primary.get(material, {}):
        return primary[material][kind]
    choices = load(E / 'selected_runs.json')
    return choices.get(material, {}).get(kind, material + '_' + kind)

def run(material, kind, gpus, timeout, initial, run_tag=None):
    policy=load(E/'material_policy.json')
    if material in policy.get('excluded',{}):
        raise ValueError('Material excluded from E7: '+material)
    # Longer ceilings apply only to newly launched children, with the existing
    # pilot/total budget reservation below; never resurrect excluded AlAs.
    requested_timeout = timeout
    timeout = max(timeout, 43200 if kind == 'fine' else 21600)
    logical_tag = material + '_' + kind if run_tag is None else run_tag
    tag = reserve_name(E,logical_tag,output_parent=E/'runs',log_parent=E)
    outfile = E / (tag + '_launch.json')
    cfg = dict(PROTO['cheap_settings'][material])
    if kind == 'fine':
        cfg.update(PROTO['fine_settings'][material])
    while True:
        reason = priority_pending()
        if reason is None and available(gpus):
            break
        dump(E / (tag + '_queue.json'), dict(time=now(), stage='waiting',
             reason=reason or 'cards busy', requested_gpus=gpus))
        time.sleep(20)
    if spent() + len(gpus) * timeout / 3600 > PROTO['pilot_GPU_hours_cap']:
        raise RuntimeError('Pilot budget exhausted; no new task launched')
    for path, expected in cfg['pseudopotentials'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pseudopotential changed: ' + path)
    env = dict(PROTO['environment'])
    tmp = '/dev/shm/response_design_e7_' + tag
    env.update(TMPDIR=tmp, JAX_COMPILATION_CACHE_DIR=tmp + '/jax_cache',
               PYTHONPYCACHEPREFIX=tmp + '/pycache')
    cmd = ['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-N1', '-n1',
           '--cpus-per-task=4', 'bash', '-c', 'mkdir -p "$TMPDIR"; exec "$@"',
           'e7', 'env', 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, gpus)),
           'timeout', str(timeout), PY, '-u', str(R / 'validation/workflows/phase3/worker.py'),
           '--material', material, '--vector', '[0,0,0,0,0,0,0]', '--tag', tag,
           '--mode', 'evaluate', '--bulk-only', '--grid', str(cfg['grid']),
           '--cutoff', str(cfg['cutoff_Ha']), '--k', str(cfg['k']),
           '--epochs', str(cfg['epochs']), '--empty', str(cfg['empty']),
           '--strain-step', str(cfg['h']), '--xc-grid-factor', str(cfg['XC_factor']),
           '--a0', str(initial), '--fused-report']
    if cfg.get('continuation_epochs'):
        cmd += ['--continuation-epochs', str(cfg['continuation_epochs'])]
    if len(gpus) > 1:
        cmd += ['--kpool']
    # Use exactly the same Miller mask for a k-only numerical comparison.
    # Different cutoff/grid comparisons each construct their proper basis.
    cheap = E / 'runs' / selected_tag(material, 'cheap')
    if kind == 'fine' and material in ('Si', 'BN', 'AlAs') and (cheap / 'mask.npy').exists():
        cmd += ['--fixed-mask', str(cheap / 'mask.npy')]
    rec = dict(time=now(), tag=tag, logical_tag=logical_tag, material=material, kind=kind, jobid=250285,
               gpus=gpus, command=cmd, environment=env, timeout_seconds=timeout,
               originally_requested_timeout_seconds=requested_timeout,
               timeout_policy='2026-10-16 new fine >=12h, cheap >=6h; unchanged budget gates',
               status='running', settings=cfg)
    start = time.monotonic()
    with (E / (tag + '.log')).open('xb') as f:
        proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True)
        rec['pid'] = proc.pid
        dump(outfile, rec)
        print('E7_START', tag, 'job250285 GPUs', gpus, 'PID', proc.pid, flush=True)
        code = proc.wait()
    rec.update(end=now(), returncode=code, GPU_hours=len(gpus)*(time.monotonic()-start)/3600,
               status='completed' if code == 0 else 'failed')
    dump(outfile, rec)
    if code:
        note(f'E7 Ct {tag}失败(returncode={code})，保留日志；不放宽门，不启动该材料的后续控制。')
        return None
    work = E / 'runs' / tag
    try:
        jac = derive(work)
        dump(work / 'equilibrium_jacobian.json', jac)
        gate = dict(time=now(), passed=True, scope='native equilibrium/iso stationarity and estimator gates',
                    independent_fine_certificate=False)
    except Exception as exc:
        gate = dict(time=now(), passed=False, error=str(exc), independent_fine_certificate=False)
    dump(work / 'native_acceptance.json', gate)
    if gate['passed']:
        choices=load(E/'selected_runs.json');choices.setdefault(material,{})[kind]=tag
        dump(E/'selected_runs.json',choices)
    note(f'E7 {tag} eq/iso完成：{rec["GPU_hours"]:.4f} GPU·h，原生数值门'+
         ('PASS' if gate['passed'] else 'FAIL')+'；独立加密复核另验，未做训练。')
    print(f'E7 子步骤：{tag} eq/iso完成。\n实测 {rec["GPU_hours"]:.4f} GPU·h，原生门 '+
          ('PASS' if gate['passed'] else 'FAIL')+'。\n新目录保留完整数据，独立加密门另验。', flush=True)
    return work

def compare(material):
    coarse = E / 'runs' / selected_tag(material, 'cheap')
    fine = E / 'runs' / selected_tag(material, 'fine')
    a, b = load(coarse / 'summary.json'), load(fine / 'summary.json')
    if not a or not b:
        return None
    ar = abs(a['a0_A'] - b['a0_A']) / abs(b['a0_A'])
    bd = abs(a['B_GPa'] - b['B_GPa'])
    native = all(load(w / 'native_acceptance.json').get('passed', False) for w in (coarse, fine))
    passed = native and ar < 1e-4 and bd < .1
    result = dict(time=now(), material=material, scope='a0 and B only, fresh independent numerical control',
                  passed=passed, native_gates_passed=native, a0_relative_error=ar,
                  B_absolute_difference_GPa=bd, a0_gate=1e-4, B_gate_GPa=.1,
                  coarse=str(coarse), fine=str(fine), parameter_FD_certified=False,
                  full_elastic_tensor_certified=False, heldout_accessed=False)
    dump(E / (material + '_independent_acceptance.json'), result)
    note(f'E7 {material}廉价→独立加密 a0/B门 '+('PASS' if passed else 'FAIL')+
         f'：Δa/a={ar:.3g}、ΔB={bd:.5g}GPa；未验证参数FD，未进入正式训练。')
    return result

def initial(material):
    files = {
        'Si': 'sg1_rerun_Si_20260919_233254_runtimev4',
        'BN': 'sg1_rerun_BN_20260919_233254_runtimev4',
        'diamond': 'sg2_diamond_g80_c200_h0025_k12_8gpu_233254_20260920',
        'AlAs': 'evaluate_AlAs_xc4_h005_k12_8gpu_remat_platform_233254'}
    return load(P / 'validation/results/phase3_20260908' / files[material] / 'summary.json')['a0_A']

def cheap_stream(gpu, materials):
    for material in materials:
        try:
            if (E / (material + '_cheap_launch.json')).exists():
                # Resume an idle CPU dispatcher; immutable completed/failed
                # GPU stages are never relaunched under their original tags.
                continue
            run(material, 'cheap', [gpu], 14400 if material == 'AlAs' else 10800, initial(material))
        except Exception as exc:
            dump(E / (material + '_controller_error.json'), dict(time=now(), error=str(exc)))
            note(f'E7 {material}调度失败：{exc}；未用旧结果充当复核。')

def report(final=False):
    costs, checks = {}, {}
    policy=load(E/'material_policy.json')
    for m in ('Si', 'BN', 'diamond', 'AlAs'):
        if m in policy.get('excluded',{}):continue
        d = load(E / (selected_tag(m, 'cheap') + '_launch.json'))
        if d.get('status') == 'completed':
            costs[m] = d['GPU_hours']
        c = load(E / (m + '_independent_acceptance.json'))
        if c:
            checks[m] = c
    core = ('Si', 'BN', 'diamond')
    core_cost = sum(costs[m] for m in core) if all(m in costs for m in core) else None
    four_cost = sum(costs.values()) if len(costs) == 4 else None
    # AlP's numerical/reference gate is independent and may remain unresolved.
    # Do not call a four-material Ct an actually measured five-material Ct.
    all_core = all(checks.get(m, {}).get('passed') for m in core)
    alas_pass = checks.get('AlAs', {}).get('passed')
    decision = 'pending measured costs and fresh gates'
    if core_cost is not None and all_core:
        if four_cost is not None and four_cost > 20:
            decision = 'remove AlAs; AlP still conditional, current certified plan three materials/three folds'
        elif 'AlAs' in checks and not alas_pass:
            decision = 'AlAs cheap gate failed; current certified plan three materials/three folds; AlP conditional'
        elif alas_pass:
            decision = 'four eligible materials/four folds; fifth AlP pending E1 and reference; not yet authorized as fivefold'
        else:
            decision = 'three eligible materials/three folds while AlAs/AlP pending'
        if core_cost > 20:
            decision = 'three materials/three folds, shrink to one candidate attempt and recheck <=400 GPUh'
    elif final:
        decision = 'cheap setting failed/incomplete: no fold or full-fit acceptance; retain failures and refine within budget'
    n = 4 if alas_pass and all_core and four_cost is not None and four_cost <= 20 else 3
    fit_ct = four_cost if n == 4 else core_cost
    attempts = 1 if fit_ct is not None and fit_ct > 20 else 3
    multiplier = 1 + attempts * n
    validation_spent = max(0., spent() - sum(costs.values()))
    projected = None if fit_ct is None else multiplier * fit_ct + max(120., validation_spent)
    if projected is not None and projected > 400:
        decision = 'projected budget exceeds 400 GPUh; shrink candidate attempts/materials before any full fit'
    if 'AlAs' in policy.get('excluded',{}):
        decision='AlAs excluded, no retry; Si/BN/diamond + conditional AlP fourfold after E1 QE/FD gates; otherwise core threefold; fine numerical gates still required'
    if 'AlP' in policy.get('excluded', {}):
        decision='AlAs/AlP excluded; Si/BN/diamond three materials, three LOO folds; independent fine gates still required; AlP retains K12 lattice constant only'
    dump(E / 'Ct_decision.json', dict(time=now(), measured_costs_GPU_hours=costs,
        measured_core_Ct_GPU_hours=core_cost, measured_four_Ct_GPU_hours=four_cost,
        measured_five_Ct_GPU_hours=None, independent_gates=checks, decision=decision,
        shared_baseline_cost_multiplier_by_material_count={'5':16,'4':13,'3':10},
        candidate_material_count=n if all_core else None,
        candidate_attempts_per_fit=attempts, projected_total_GPU_hours=projected,
        provisional_until_all_independent_controls_finish=not final, budget_GPU_hours=400,
        excluded_materials=list(policy.get('excluded',{})),
        allowed_conditional_material_sets=([['Si','BN','diamond']] if 'AlP' in policy.get('excluded', {}) else [['Si','BN','diamond'],['Si','BN','diamond','AlP']]) if policy else None,
        reserved_validation_GPU_hours=120, pilot_GPU_hours_spent=spent(),
        Ct_excludes_independent_validation=True, AlP=('excluded; K12 a0 only' if 'AlP' in policy.get('excluded', {}) else 'pending E1 three-parameter FD plus reference gate'),
        heldout_result_files_opened=[], no_model_fit_submitted=True, final=final))

def main():
    if '--resume-idle-handover' not in sys.argv:
        os.close(os.open(E / 'controller.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    dump(E / 'status.json', dict(time=now(), stage='cheap_eq_iso_running', jobid=250285,
         initial_gpus=[0,1], heldouts_sealed=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cheap_stream, 0, ['Si', 'diamond']),
                   pool.submit(cheap_stream, 1, ['BN', 'AlAs'])]
        for f in futures:
            f.result()
    report()
    dump(E / 'status.json', dict(time=now(), stage='fresh_independent_controls_queued', jobid=250285))
    for material in ('Si', 'BN', 'diamond', 'AlAs'):
        if (E / (material + '_fine_launch.json')).exists():
            compare(material)
            continue
        cheap = E / 'runs' / selected_tag(material, 'cheap')
        if not load(cheap / 'native_acceptance.json').get('passed'):
            continue
        gpus = [0] if material in ('Si', 'BN') else [0,1,5,7]
        try:
            # Historical fresh eq/iso cost was ~21.5 / ~56.5 GPUh for
            # diamond / AlAs. A 4h four-card ceiling would force a needless
            # timeout. These remain inside the shared 120 GPUh pilot cap.
            timeout = {'Si':14400, 'BN':14400, 'diamond':28800, 'AlAs':54000}[material]
            run(material, 'fine', gpus, timeout, load(cheap / 'summary.json')['a0_A'])
            compare(material)
        except Exception as exc:
            dump(E / (material + '_fine_controller_error.json'), dict(time=now(), error=str(exc)))
            note(f'E7 {material}独立复核未完成：{exc}；不认定廉价设置通过。')
        report()
    report(final=True)
    decision = load(E / 'Ct_decision.json')['decision']
    dump(E / 'status.json', dict(time=now(), stage='pilot_finished_review_certificates',
         decision=decision, spent_GPU_hours=spent(), no_model_fit_submitted=True))
    note('E7 Ct pilot终态：'+decision+'；详见response_design_20261015_e7_ct_pilot/Ct_decision.json，SiC/GaP继续封存。')
    print('E7 Ct计量及独立门流程结束。\n'+decision+'。\n未提交正式校准，完整证据已落盘，留出材料封存。',flush=True)

if __name__ == '__main__':
    main()
