"""E1: refine K10 -> K12 -> optional K14 without changing acceptance gates.

Fresh results, immutable physical runtime and Miller mask. GPU tasks run under
the existing allocation; no cancellation and no replacement of historical rows.
This controller certifies k/numerical gates, not an unfinished QE/parameter FD.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[3]
E = P / 'validation/results/response_design_20261014_e1_alp'
A = E / 'extension_20261015'
PY = '/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
GPUS = [2, 3, 4, 6]
sys.path.insert(0, str(P / 'validation/workflows/phase3'))
from check_elastic_convergence import compare
from equilibrium_jacobian import derive
import numpy as np


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def now():
    return datetime.datetime.now().isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(stage, **values):
    dump(A / 'status.json', dict(time=now(), stage=stage, jobid=250285,
                               formal_E1_complete=False, **values))


def note(message):
    for path in [P / 'paper/response_design_EXPERIMENTS_20261014.md', P / 'response_design_to_do_list.md']:
        with path.open('a') as f:
            f.write('\n- ' + now() + ' ' + message + '\n')


def idle_probe():
    cmd = ['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-N1', '-n1',
           '--cpus-per-task=1', 'nvidia-smi', '--query-gpu=index,uuid,memory.used',
           '--format=csv,noheader,nounits']
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('GPU probe failed: ' + result.stderr)
    rows = {int(r.split(',')[0]): r.strip().split(',') for r in result.stdout.splitlines()}
    # No task is killed to clear a card. Retry while another task owns a card.
    return result.stdout, all(int(rows[g][2]) == 0 for g in GPUS)


def run_stage(tag, k, a0, mode='evaluate', extra_args=()):
    remaining = 250 - spent_before - sum(r['GPU_hours'] for r in launches)
    # Four active cards; timeout never permits this controller to cross E1 cap.
    timeout = min(14400, int(max(0, remaining - 2) * 900))
    if timeout < 900:
        raise RuntimeError('E1 budget exhausted; no further GPU task submitted')
    while True:
        probe, idle = idle_probe()
        dump(A / 'gpu_probe.json', dict(time=now(), output=probe, gpus=GPUS, idle=idle))
        if idle:
            break
        status('waiting_selected_cards', tag=tag)
        time.sleep(20)
    tmp = '/dev/shm/response_design_e1_ext_' + tag
    env.update(TMPDIR=tmp, JAX_COMPILATION_CACHE_DIR=tmp + '/jax_cache',
               PYTHONPYCACHEPREFIX=tmp + '/pycache')
    cmd = ['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-N1', '-n1',
           '--cpus-per-task=16', 'bash', '-c', 'mkdir -p "$TMPDIR"; exec "$@"',
           'e1-extension', 'env', 'CUDA_VISIBLE_DEVICES=2,3,4,6', 'timeout',
           str(timeout), PY, '-u', str(runtime / 'validation/workflows/phase3/worker.py'),
           '--material', 'AlP', '--parameters', 'base', '--tag', tag, '--mode', mode,
           '--grid', '64', '--cutoff', '50', '--k', str(k), '--epochs', '3000',
           '--empty', '0', '--strain-step', '0.005', '--a0', str(a0), '--kpool',
           '--fused-report', '--fixed-mask', str(mask)]
    if mode == 'sample':
        cmd += ['--sample-observables', 'full']
    cmd += list(extra_args)
    rec = dict(start=now(), jobid=250285, gpus=GPUS, tag=tag, k=k, mode=mode,
               command=cmd, environment=dict(env), status='running', timeout_seconds=timeout)
    dump(A / (tag + '_launch.json'), rec)
    status('GPU_stage_running', tag=tag, k=k, GPU_hours_completed=sum(r['GPU_hours'] for r in launches))
    print('E1_START', tag, 'job250285 GPUs2,3,4,6', flush=True)
    start = time.monotonic()
    with (A / (tag + '.log')).open('xb') as f:
        proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True)
        rec['pid'] = proc.pid
        dump(A / (tag + '_launch.json'), rec)
        code = proc.wait()
    rec.update(end=now(), returncode=code, GPU_hours=4 * (time.monotonic() - start) / 3600,
               status='completed' if code == 0 else 'failed')
    launches.append(rec)
    dump(A / (tag + '_launch.json'), rec)
    if code:
        raise RuntimeError(f'{tag}: returncode {code}; failure retained, no physics gate bypass')
    return A / 'runs' / tag


def main():
    global env, runtime, mask, launches, spent_before
    A.mkdir(exist_ok=True)
    fd = os.open(A / 'controller.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    original = json.loads((E / 'protocol.json').read_text())
    gate = json.loads(Path(original['formal_production_gate']).read_text())
    if not gate['passed']:
        raise RuntimeError('E0 not passed')
    runtime = Path(original['runtime'])
    previous = E / 'runs/AlP_NLCC_k10'
    mask = previous / 'mask.npy'
    spent_before = sum(json.loads(p.read_text())['GPU_hours'] for p in E.glob('*_launch.json')
                       if 'GPU_hours' in json.loads(p.read_text()))
    # Additional K10 force-control launch has no GPU-hour field yet; reserve its
    # full one-hour timeout (4 GPU h), rather than undercount its actual use.
    spent_before += 4
    base = json.loads((P / 'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())
    env = base['environment'].copy()
    env['PYTHONPATH'] = env['PYTHONPATH'].replace(
        str(P / 'validation/results/response_design_20261014_e0_migration/runtime_tail'), str(runtime))
    env['D3FT_PHASE3_RESULTS_ROOT'] = str(A / 'runs')
    (A / 'runs').mkdir(exist_ok=True)
    launches = []
    dump(A / 'protocol.json', dict(time=now(), decision='User: K4 insufficient, refine K10→K12→optional K14',
        k_queue=[12, 14], jobid=250285, gpus=GPUS, runtime=str(runtime), cutoff_Ha=50,
        grid=64, epochs=3000, strain_step=.005, empty_bands=0, Miller_mask=str(mask),
        Miller_sha256=sha(mask), reference=str(previous), lattice_relative_gate=1e-4,
        bulk_absolute_gate_GPa=.1, E1_budget_GPU_hours=250, conservative_spent_before_GPU_hours=spent_before,
        parameter_order=original['parameter_order'], adopt='fine mesh of first passing adjacent pair',
        old_SG15_rows_replaced=False, parameter_FD_and_QE_are_separate_gates=True))
    accepted = None
    for k in [12, 14]:
        s0 = json.loads((previous / 'summary.json').read_text())
        fine = run_stage(f'AlP_NLCC_k{k}', k, s0['a0_A'])
        result = compare(previous, fine, 'k', lattice_budget=1e-4, elastic_budget=.1)
        coarse_k = json.loads((previous / 'metadata.json').read_text())['arguments']['k']
        dump(A / f'convergence_k{coarse_k}_k{k}.json', result)
        note(f'E1 K{coarse_k}→K{k}既定k门 {"PASS" if result["passed"] else "FAIL"}；'
             f'证据 extension_20261015/convergence_k{coarse_k}_k{k}.json，门不放宽，旧SG15行不替换。')
        print(f'E1 子步骤：K{coarse_k}→K{k} '+('PASS' if result['passed'] else 'FAIL')+
              '\n晶格相对门1e-4、B差门0.1GPa保持。\n原始结果及旧SG15行保留。', flush=True)
        if result['passed']:
            accepted = (k, fine, result)
            break
        previous = fine
    if accepted is None:
        status('K14_still_not_converged', k_convergence_passed=False,
               E7_AlP_eligible=False, total_GPU_hours_upper_bound=spent_before+sum(r['GPU_hours'] for r in launches))
        note('E1 K12→K14仍未过原门；AlP从E7候选训练集剔除，替换材料须为未作为留出的材料；不改变SiC/GaP留出身份，不删除失败记录。')
        return
    k, fine, convergence = accepted
    summary = json.loads((fine / 'summary.json').read_text())
    derived = derive(fine)
    da = np.asarray(derived['dlogV_equilibrium_dp']) * summary['a0_A'] / 3
    db = np.asarray(derived['dB_equilibrium_dp'])
    checks = dict(finite_seven=bool(da.shape == (7,) and db.shape == (7,) and np.isfinite(np.r_[da, db]).all()),
                  bulk_step_stability=abs(summary['B_h_GPa']-summary['B_2h_GPa']) < .1,
                  da0_method_agreement=bool(np.all(abs(da-np.asarray(summary['da0_dp'])) <= .02*np.maximum(abs(da),abs(np.asarray(summary['da0_dp'])))+1e-7)),
                  dB_method_agreement=bool(np.all(abs(db-np.asarray(summary['dB_equilibrium_dp'])) <= .02*np.maximum(abs(db),abs(np.asarray(summary['dB_equilibrium_dp'])))+1e-3)))
    row = dict(k=k, a0_A=summary['a0_A'], B_GPa=summary['B_GPa'], da0_dp=da.tolist(),
               dB_equilibrium_dp=db.tolist(), parameter_order=original['parameter_order'],
               source=str(fine), source_hashes={name:sha(fine/name) for name in ['metadata.json','points.json','summary.json','mask.npy']},
               method='equilibrium_jacobian method_version=2; fresh adopted-mesh evaluations',
               numerical_checks=checks, k_convergence_passed=True, numerical_passed=all(checks.values()),
               accepted_for_calibration=False, independently_reoptimized_parameter_FD=False,
               old_SG15_rows_replaced=False)
    dump(A / f'response_row_k{k}.json', row)
    if not all(checks.values()):
        raise RuntimeError('Adopted-mesh response numerical gate failed; row not certified')
    force = run_stage(f'AlP_NLCC_k{k}_force', k, summary['a0_A'], mode='sample')
    point = json.loads((force/'points.json').read_text())[0]
    forces = np.asarray(point['forces'])
    force_checks = dict(stationarity=point['residual'] <= 1e-6,
                        pressure_root=abs(point['pressure_GPa'])/(3*summary['B_GPa']) < 1e-5,
                        force_max=float(abs(forces).max()) < 1e-6,
                        force_sum=float(abs(forces.sum(axis=0)).max()) < 5e-8)
    dump(A / 'adopted_force_acceptance.json', dict(k=k, checks=force_checks,
        passed=all(force_checks.values()), source=str(force), forces_Ha_Bohr=forces.tolist()))
    status('adopted_mesh_native_complete_pending_QE_parameter_FD', adopted_k=k,
           k_convergence_passed=True, response_numerical_passed=True,
           force_passed=all(force_checks.values()), independently_reoptimized_parameter_FD=False,
           QE_at_adopted_mesh_passed=False, response_row=str(A/f'response_row_k{k}.json'),
           total_GPU_hours_upper_bound=spent_before+sum(r['GPU_hours'] for r in launches))
    note(f'E1采用候选K{k}：k/原生响应数值检查完成；七参数行已在新网格重新计算。正式完成仍需匹配QE及独立参数FD，不替换旧SG15。')
    print(f'E1 采用候选K{k}，新七参数行已保存。\n匹配QE/独立参数FD尚待验收，未冒称E1完成。\n旧SG15与粗网格失败记录保留。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if A.exists():
            status('failed', error=str(exc))
            note('E1扩展阶段失败：'+str(exc)+'；保留现场，不取消其他作业。')
        raise
