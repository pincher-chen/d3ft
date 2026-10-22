"""Repair only four failed k6 spectra; release FD data after E6 acceptance.

Predecessor terminal records are archived byte-for-byte before its mutable
consumer index is redirected. No failed spectrum/SCF directory is modified.
The already waiting E5 process may still hold the predecessor path in memory.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from run_identity import reserve_name


def read(path): return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def validate_local(folder, source_hash, k):
    folder = Path(folder)
    row = read(folder/'summary.json')
    values = np.asarray(row['eigenvalues_Ha'])
    residuals = np.asarray(row['residuals_Ha'])
    if row['k_index'] != k or row['source_state_sha256'] != source_hash:
        raise ValueError('Wrong k/source for reusable spectrum')
    if row['saved_orbitals_sha256'] != sha(folder/'orbitals.npy'):
        raise ValueError('Saved Ritz vectors changed')
    if values.shape != (52,) or residuals.shape != (52,) or not np.isfinite(np.r_[values, residuals, row['orthogonality_max']]).all():
        raise ValueError('Nonfinite/incomplete local spectrum')
    if np.min(np.diff(values)) < -1e-10 or np.max(residuals) > 1e-9 or row['orthogonality_max'] > 1e-10:
        raise ValueError('Original local eigenpair gate failed')
    return row


def main(T):
    os.close(os.open(T/'controller.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    cfg = read(T/'protocol.json')
    P = Path(cfg['project'])
    R = Path(cfg['runtime'])
    E = P/'validation/results/response_design_20261015_e6_k2_hydro'
    old = Path(cfg['previous_root'])
    Q = P/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
    sys.path.insert(0, str(R/'validation/workflows/response_design/reverse_response'))
    from mesh_target import select_mesh_target
    spent = cfg['prior_GPU_hours']
    rows = []
    current = None

    def now(): return dt.datetime.now().astimezone().isoformat()

    def status(stage, **kw):
        result = dict(time=now(), stage=stage, controller_pid=os.getpid(), jobid=250285,
            gpus=[7], spent_GPU_hours=spent, required_k6_repairs=4, completed_points=len(rows), **kw)
        dump(T/'status.json', result)
        dump(E/'status.json', result)
        # Consumer compatibility for E5's already-resolved local H variable.
        # Keep it waiting (rather than trigger its old local-SCF fallback) until
        # all four values AND the E6 h/h2 adjoint gate have passed.
        if not (T/'E5_release.json').exists():
            dump(old/'status.json', dict(result, stage='waiting_fresh_E6_repair_acceptance',
                recovery_stage=stage, recovery_root=str(T), predecessor_run_failed=True,
                immutable_failure_record=str(T/'predecessor_failure/status.json'),
                E5_FD_released=False))

    def note(s):
        for path in [P/'paper/response_design_EXPERIMENTS_20261014.md', P/'response_design_to_do_list.md']:
            with path.open('a') as f: f.write('\n- '+now()+' '+s+'\n')

    def idle():
        probe = subprocess.run(['srun', '--jobid=250285', '--overlap', '--cpu-bind=none',
            '-N1', '-n1', '-c1', 'nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=45)
        if probe.returncode: raise RuntimeError(probe.stderr)
        memory = {int(v[0]):int(v[1]) for l in probe.stdout.splitlines() if len(v:=l.split(',')) == 2}
        dump(T/'GPU_probe.json', dict(time=now(), memory_MiB=memory))
        return memory.get(7, 10**9) <= 3

    try:
        status('waiting_actual_idle_GPU7')
        while not idle(): time.sleep(15)
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reservations = read(Q)
            if reservations.get('7', {}).get('status_file') not in [None, str(old/'status.json')]:
                raise RuntimeError('GPU7 has another reservation')
            reservations['7'] = dict(owner='E6 four k6 Ritz restarts', status_file=str(T/'status.json'),
                controller_pid=os.getpid())
            dump(Q, reservations)
        for task in cfg['tasks']:
            h, sign = task['h'], task['sign']
            point = Path(task['source'])
            source_hash = sha(point/'state.pkl')
            if source_hash != task['source_state_sha256']: raise ValueError('SCF state changed')
            pt = read(point/'point.json')
            if not pt['stationarity_pass'] or pt['residual'] > 1e-8 or pt['acoustic_sum_rule_Ha_Bohr'] > 5e-8 or abs(pt['charge']-96) > 96e-8:
                raise ValueError('Existing SCF native gate failed')
            directories = list(task['spectrum_directories'])
            for k in range(8):
                if k != 6: validate_local(directories[k], source_hash, k)
            restart = read(Path(directories[6])/'summary.json')
            if restart['source_state_sha256'] != source_hash or restart['k_index'] != 6 or sha(Path(directories[6])/'orbitals.npy') != restart['saved_orbitals_sha256']:
                raise ValueError('Changed/wrong old k6 Ritz restart')
            if spent+cfg['spectrum_timeout_seconds']/3600+1/60 > 60:
                raise RuntimeError('E6 total 60 GPUh reservation cap; no task launched')
            tag = reserve_name(T, 'nu'+str(sign*h)+'_k6_Ritz', output_parent=T, log_parent=T)
            output = T/tag
            protocol = read(task['spectrum_protocol'])
            protocol.update(runtime=str(R), spectrum_maxiter=1200,
                spectrum_restart={'6':dict(orbitals=str(Path(directories[6])/'orbitals.npy'),
                    sha256=restart['saved_orbitals_sha256'], source_state_sha256=source_hash)})
            proto = T/(tag+'_protocol.json')
            dump(proto, protocol)
            tmp = '/dev/shm/'+tag
            env = {**os.environ, **cfg['environment'], 'PYTHONPATH':str(R)+':'+cfg['jrystal'],
                'TMPDIR':tmp, 'PYTHONPYCACHEPREFIX':tmp+'/pycache', 'JAX_COMPILATION_CACHE_DIR':tmp+'/jax'}
            command = ['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-N1', '-n1', '-c4',
                'bash', '-c', 'mkdir -p "$TMPDIR"; exec "$@"', 'E6-k6-repair', 'env',
                'CUDA_VISIBLE_DEVICES=7', 'timeout', str(cfg['spectrum_timeout_seconds']), cfg['python'],
                '-u', str(R/'validation/workflows/response_design/reverse_response/e5_mg_sharded_driver.py'),
                '--protocol', str(proto), '--stage', 'spectrum', '--k-index', '6', '--output', str(output)]
            f = (T/(tag+'.log')).open('xb')
            start = time.monotonic()
            current = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(time=now(), pid=current.pid, jobid=250285, gpus=[7], h=h, sign=sign,
                command=command, environment=env, output=str(output), status='running', timeout_seconds=cfg['spectrum_timeout_seconds'])
            dump(T/(tag+'_launch.json'), record)
            while current.poll() is None:
                status('restarting_local_k6_from_old_Ritz', h=h, sign=sign, pid=current.pid,
                       elapsed_seconds=time.monotonic()-start)
                time.sleep(15)
            f.close()
            hours = (time.monotonic()-start)/3600
            spent += hours
            record.update(end=now(), returncode=current.returncode, GPU_hours=hours,
                status='completed' if not current.returncode else 'failed')
            dump(T/(tag+'_launch.json'), record)
            if current.returncode: raise RuntimeError('Restarted k6 failed; immutable record '+tag)
            validate_local(output, source_hash, 6)
            directories[6] = str(output)
            values = np.asarray([validate_local(d, source_hash, k)['eigenvalues_Ha'] for k, d in enumerate(directories)])
            _, target = select_mesh_target(values, 48)
            fractional = np.asarray(pt['k_fractional'])
            gamma = int(np.argmin(np.linalg.norm(fractional-np.rint(fractional), axis=1)))
            row = dict(h=h, sign=sign, nu=sign*h, passed=True, source=str(point),
                source_state_sha256=source_hash, fundamental_eV=target['hard_gap_Ha']*27.211386245988,
                Gamma_eV=float((values[gamma,48]-values[gamma,47])*27.211386245988),
                vbm_k=target['vbm_k_index'], cbm_k=target['cbm_k_index'], target=target,
                spectrum_directories=directories, reused_old_spectra=7, repaired_k=6)
            rows.append(row)
            dump(T/'hydro_values.json', rows)
            note('E6静水ν='+str(sign*h)+' k6旧Ritz重启PASS；其余7谱复用，SCF未重跑；四点总门仍待。')
            print('E6 ν='+str(sign*h)+' k6修复PASS。\n复用7个原合格谱，原tol/Ritz门不变。\n四点齐后再放行E5。', flush=True)
        response_file = Path(cfg['base_response_file'])
        if sha(response_file) != cfg['base_response_sha256']: raise ValueError('Base adjoint result changed')
        response = read(response_file)
        if not response['passed']: raise ValueError('Base response not accepted')
        theta = np.asarray(cfg['theta0'])
        direction = np.r_[(1+theta[:3])/3, theta[3:6]/3, theta[6:]/3]
        adjoint = float(np.asarray(response['gradient_eV'])@direction)
        derivatives, checks = [], []
        winner = response['target']
        branches = all(r['vbm_k'] == winner['vbm_k_index'] and r['cbm_k'] == winner['cbm_k_index'] for r in rows)
        for h in (.002, .001):
            pair = {r['sign']:r for r in rows if r['h'] == h}
            derivative = dict(h=h, **{k:(pair[1][k]-pair[-1][k])/(2*h) for k in ['fundamental_eV', 'Gamma_eV']})
            derivatives.append(derivative)
            error = abs(derivative['fundamental_eV']-adjoint)
            tol = 3e-5+.002*abs(adjoint)
            checks.append(dict(h=h, error_eV=error, tolerance_eV=tol, passed=error<=tol))
        result = dict(time=now(), passed=branches and all(v['passed'] for v in checks),
            k2_pilot_passed=branches and all(v['passed'] for v in checks), checks=checks,
            branch_consistent_with_adjoint=branches, adjoint_fundamental_A_eV=adjoint,
            FD=derivatives, h_h2_difference_eV=abs(derivatives[0]['fundamental_eV']-derivatives[1]['fundamental_eV']),
            points=rows, spent_GPU_hours=spent, formal_E6_complete=False,
            not_literature_PASS=True, not_k4_certificate=True,
            Gamma_FD_compared_to_fundamental_adjoint=False,
            definition='dEg/dlnV, clamped fractional positions, exact prestrain hydro direction',
            base_response_file=str(response_file), base_response_sha256=cfg['base_response_sha256'])
        dump(T/'FD_h_h2.json', dict(time=now(), derivatives=derivatives, points=rows,
             ionic_condition='fixed fractional positions', not_k4_certificate=True, not_literature_PASS=True))
        dump(T/'k2_hydro_acceptance.json', result)
        if not result['passed']: raise RuntimeError('Original E6 h/h2/branch gate failed; E5 FD not released')
        # Publish only after the independent E6 certificate exists and passes.
        dump(T/'E5_release.json', dict(time=now(), passed=True, certificate=str(T/'k2_hydro_acceptance.json'),
             predecessor_failure_backup=str(T/'predecessor_failure'), release_to_cached_E5_path=str(old)))
        dump(old/'hydro_values.json', rows)
        dump(old/'E6_recovery_release.json', dict(passed=True, recovery_root=str(T), certificate=str(T/'k2_hydro_acceptance.json'),
             original_failure_archived=str(T/'predecessor_failure'), original_spectra_unchanged=True))
        dump(old/'status.json', dict(time=now(), stage='E6_fresh_repair_accepted_for_E5_reuse',
             recovery_root=str(T), predecessor_run_failed=True, E5_FD_released=True,
             original_failure_record=str(T/'predecessor_failure/status.json')))
        status('k2_hydro_h_h2_accepted_E5_released', passed=True, formal_E6_complete=False)
        note('E6新目录四静水点/Ritz/分支及原h/h2伴随门PASS，才向E5旧缓存路径发布合格四点；原失败按字节保留，非k4/文献PASS。')
        print('E6 k2静水h/h2验收PASS。\n四点合格数据已放行E5复用，旧失败保留。\nk4与文献对照不冒称完成。', flush=True)
    except Exception as exc:
        status('failed', error=repr(exc), passed=False, E5_FD_released=False)
        note('E6 k6修复队列失败：'+repr(exc)+'；失败保留，E5保持等待，不用旧失败触发多余SCF。')
        raise
    finally:
        if current is not None and current.poll() is None:
            current.wait()  # its own bounded timeout; never signal a GPU child
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reservations = read(Q)
            if reservations.get('7', {}).get('status_file') == str(T/'status.json'):
                reservations.pop('7')
                dump(Q, reservations)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    main(ap.parse_args().root)
