"""CPU integration: preserve a complete repeat0, reject partial/incorrect data."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

source = Path(__file__).with_name('m2_planB_supervisor.py')
spec = importlib.util.spec_from_file_location('planB', source)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ref = [1.0]*54
reverse = [dict(N=54, repeat=i, mode='reverse', response_solves=1,
                seconds=1.0, gradient_eV=ref) for i in range(3)]
forward = [dict(N=54, repeat=0, mode='forward', response_solves=54,
                seconds=54.1, rhs_seconds=.1, gradient_eV=ref,
                axes=[dict(axis=i, seconds=1.0) for i in range(54)])]
assert m.validate(reverse, forward, ref)['passed']
for mutation in ['partial', 'bad_gradient', 'wrong_repeat', 'nan_reverse']:
    rr = copy.deepcopy(reverse); ff = copy.deepcopy(forward)
    if mutation == 'partial': ff[0]['axes'].pop()
    if mutation == 'bad_gradient': ff[0]['gradient_eV'][4] += .01
    if mutation == 'wrong_repeat': ff[0]['repeat'] = 1
    if mutation == 'nan_reverse': rr[0]['gradient_eV'][7] = float('nan')
    try:
        m.validate(rr, ff, ref)
    except AssertionError:
        pass
    else:
        raise AssertionError('Accepted '+mutation)

with tempfile.TemporaryDirectory(prefix='m2_planB_CPU_') as temp:
    P = Path(temp); (P/'paper').mkdir(); R = P/'validation/results/review/run'; M = R/'M2-A'
    D = M/'planB_20261021T2130'; D.mkdir(parents=True)
    O = M/'native_output'; O.mkdir(); (M/'logs').mkdir()
    e3 = P/'e3'; e3.mkdir()
    m.dump(e3/'mg_mode_acceptance.json', {'reverse_gradient_eV':ref})
    m.dump(M/'protocol.json', {'E3_root':str(e3)})
    m.dump(D/'CPU_acceptance.json', {'passed':True})
    m.dump(D/'handover_ready.json', {'CPU_test':True})
    for id,key,value in [('M3-A','total_M3_GPU_hours',9),('M5-A','total_M5_GPU_hours',5)]:
        d = R/id/'new'; d.mkdir(parents=True)
        m.dump(R/id/('current_continuation.json' if id=='M3-A' else 'current_sharded.json'), {'root':str(d)})
        m.dump(d/'GPU_ledger.json', {key:value})
    worker = P/'fake_timing.py'
    for name in ['gamma_response.py','geometry_layout.py','projected_solver.py']:
        (P/name).write_text('# CPU fixture only\n')
    worker.write_text('''import time,json,sys
from pathlib import Path
O=Path(sys.argv[sys.argv.index('--output')+1])
time.sleep(.3)
data=json.loads((O/'fixture.json').read_text())
(O/'reverse_raw.json').write_text(json.dumps(data['reverse']))
(O/'repeat0_completed_before_stop').write_text('complete54')
(O/'forward_raw.json').write_text(json.dumps(data['forward']))
time.sleep(60)
''')
    m.dump(O/'fixture.json', {'reverse':reverse,'forward':forward})
    m.dump(O/'status.json', {'stage':'forward_axis_complete','repeat':0,'axis':52})
    m.dump(O/'memory_peak.json', {'CPU_fixture':True})
    log = M/'logs/native.log'; log.write_text('CPU fixture\n')
    env = os.environ.copy(); env.update(SLURM_JOB_ID='250285', SLURM_STEP_ID='15190', CUDA_VISIBLE_DEVICES='0')
    cmd = [sys.executable, str(worker),'--root',str(e3),'--output',str(O)]
    proc = subprocess.Popen(cmd, env=env)
    record = dict(pid=proc.pid, experiment='M2-A',start_epoch=m.time.time(), output=str(O),command=cmd,log=str(log))
    m.dump(M/'logs/exclusive_N54_r3_launch.json',record)
    m.dump(D/'ledger_before_handover.json', {'jobs':[record], 'GPU_hours':{'M2-A':0}})
    os.environ.update(SLURMD_NODENAME='hn35', SLURM_JOB_ID='250285')
    m.snapshot = lambda: {'time':m.now(), 'rows':[[0,100,50]]+[[i,0,0] for i in range(1,8)]}
    try:
        m.main(R)
        code = proc.wait(timeout=5)
        assert code == -15
        assert (O/'repeat0_completed_before_stop').read_text() == 'complete54'
        assert m.read(O/'forward_raw.json') == forward
        assert m.read(D/'result.json')['forward_repeats']==1
        assert m.read(D/'acceptance.json')['passed']
        assert m.read(M/'summary.json')['passed']
        assert not (O/'result.json').exists() # no fabricated native exit0/result
        assert '3反向+1正向' in (P/'response_design_to_do_list.md').read_text()
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=5)
print(json.dumps({'passed':True,'tests':['full3plus1 accepted','partial54 rejected',
 'wrong derivative rejected','wrong repeat rejected','NaN rejected',
 'CPU live worker completed repeat0 before controlled signal',
 'native data unchanged','supervisor exits normally with result/acceptance/summary',
 'native signal exit disclosed; no fabricated native result']}))
