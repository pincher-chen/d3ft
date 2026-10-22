"""Four independent E7 fits share eight GPUs and one400GPUh ledger.

Only CPU controllers are created here; every calculation remains explicitly
bound to allocation250285/hn35. Existing GPU children are adopted, not killed.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from e1_three_parameter_queue import read, dump, alive
from e7_shared_budget import SharedBudget


def main(root):
    cfg = read(root / 'parallel_protocol.json')
    project = Path(cfg['project'])
    ledger = SharedBudget(root / 'shared_GPU_budget.json')
    children = {}
    with (root / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ['full', 'LOO_Si', 'LOO_BN', 'LOO_diamond']:
            folder = root / name
            c = read(folder / 'protocol.json')
            env = dict(os.environ)
            env.update(PYTHONPATH=':'.join([str(root), str(project / 'validation/workflows/response_design'),
                c['runtime'], c['jrystal']]), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                response_GPU_PROBE_JOURNAL=str(folder / 'gpu_probe_retries.jsonl'))
            with (folder / 'controller.log').open('xb') as log:
                p = subprocess.Popen([c['python'], '-u', str(folder / 'controller.py'),
                    '--root', str(folder)], stdout=log, stderr=subprocess.STDOUT,
                    cwd=project, env=env, start_new_session=True)
            children[name] = dict(pid=p.pid, root=str(folder), proc=p)
            dump(folder / 'launch.json', dict(controller_pid=p.pid, selected_fit=name,
                root=str(folder), jobid=250285, node='hn35', dynamic_GPU_pool=list(range(8))))
            if name == 'full':
                q = project / 'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
                with q.with_name('gpu_reservations.lock').open('a') as qlock:
                    fcntl.flock(qlock, fcntl.LOCK_EX)
                    owners = read(q)
                    for owner in owners.values():
                        if owner.get('controller_pid') == cfg['previous_CPU_pid']:
                            owner.update(controller_pid=p.pid, status_file=str(folder / 'status.json'),
                                         owner='E7 full adopted active DFT; shared eight-card pool')
                    dump(q, owners)
        dump(root / 'children.json', {k: {n:v for n,v in d.items() if n!='proc'} for k,d in children.items()})
        reported = set()
        while True:
            states = {}
            for name, d in children.items():
                r = Path(d['root'])
                status = read(r / 'status.json') if (r / 'status.json').exists() else {}
                heartbeats = [read(p) for p in r.glob('*_worker_status.json')]
                states[name] = dict(controller_pid=d['pid'], alive=alive(d['pid']),
                    returncode=d['proc'].poll(), state=status, DFT_heartbeats=heartbeats)
                result = r / (name + '_result.json')
                complete = result.exists() and (name=='full' or 'validation' in read(result))
                if complete and name not in reported:
                    row = read(result)
                    message = (f'E7并行{name}完成：接受{row["accepted_updates"]}/{row["trials"]}，'
                               '参数冻结后验证，SiC/GaP未使用；非零参数独立加密验收仍另记。')
                    for dest in [project / 'response_design_to_do_list.md', project / 'paper/response_design_EXPERIMENTS_20261014.md']:
                        with dest.open('a') as f:
                            f.write('\n- ' + dt.datetime.now().astimezone().isoformat() + ' ' + message + '\n')
                    reported.add(name)
            done = all(d['proc'].poll() is not None for d in children.values())
            dump(root / 'status.json', dict(time=dt.datetime.now().astimezone().isoformat(),
                stage='parallel_fits_finished' if done else 'parallel_full_fit_and_three_LOO_running',
                controller_pid=os.getpid(), jobid=250285, node='hn35', configured_GPU_pool=list(range(8)),
                shared_budget=ledger.summary(), fits=states, sealed_holdouts=['SiC','GaP']))
            if done:
                all_complete = len(reported)==4 and all(d['proc'].returncode==0 for d in children.values())
                results = {name: read(Path(d['root']) / (name+'_result.json')) for name,d in children.items()
                           if (Path(d['root']) / (name+'_result.json')).exists()}
                dump(root / 'result.json', dict(completed=all_complete, full_fit_and_three_LOO=results,
                    shared_budget=ledger.summary(), sealed_external_holdouts=['SiC','GaP'],
                    external_generalization_certified=False, nonzero_parameter_fine_validation_complete=False))
                return
            time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
