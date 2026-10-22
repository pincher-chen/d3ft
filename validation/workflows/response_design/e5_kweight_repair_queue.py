"""Charge-regression -> 100-step smoke -> fresh 4000-step SCF, fixed cards.

This repair queue never releases E6 on a smoke result and never cancels work.
Old failed stages count against the pilot budget and remain immutable.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse, datetime, hashlib, json, os, subprocess, time
from pathlib import Path
from run_identity import reserve_name

P = Path(__file__).resolve().parents[3]
E = P/'validation/results/response_design_20261015_e5_distributed'
A = E/'four_card_20261015'
ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
args = ap.parse_args(); T = args.root
cfg = json.loads((T/'repair_protocol.json').read_text())
R = Path(cfg['runtime']); GS = cfg['gpus']; spent = cfg['previous_GPU_hours']
PY = '/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python'
env = json.loads((P/'validation/results/response_design_20261014_e0_migration/commands/alp_new_qe_tail.json').read_text())['environment']
env['PYTHONPATH'] = str(R)+':'+env['PYTHONPATH'].split(':')[-1]

def now(): return datetime.datetime.now().isoformat()
def read(q): return json.loads(q.read_text()) if q.exists() else {}
def dump(q, row):
    tmp = q.with_suffix('.tmp'); tmp.write_text(json.dumps(row, indent=2, allow_nan=False)+'\n'); tmp.replace(q)
def status(stage, **kw):
    row = dict(time=now(), stage=stage, jobid=250285, gpus=GS,
               spent_GPU_hours_including_original_failure=spent, repair_root=str(T), **kw)
    for q in [T/'status.json', E/'queue_status.json', A/'queue_status.json']: dump(q, row)
    print(json.dumps(row), flush=True)
def note(s):
    for q in [P/'paper/response_design_EXPERIMENTS_20261014.md', P/'response_design_to_do_list.md']:
        with q.open('a') as f: f.write('\n- '+now()+' '+s+'\n')
def idle():
    r = subprocess.run(['srun', '--jobid=250285', '--overlap', '--cpu-bind=none', '-n1', '-c1',
        'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=60)
    if r.returncode: raise RuntimeError(r.stderr)
    rows = {int(x[0]): [int(v) for v in x[1:]] for line in r.stdout.splitlines() if len(x := line.split(',')) == 3}
    dump(T/'latest_GPU_probe.json', dict(time=now(), rows=rows, selected=GS))
    return all(rows[g] == [0,0] for g in GS)
def run_stage(logical, epochs, timeout_seconds, smoke):
    global spent
    if spent+len(GS)*timeout_seconds/3600 > 48: raise RuntimeError('E5 inclusive 48 GPUh pilot cap')
    tag = reserve_name(T, logical, output_parent=T, log_parent=T)
    out = T/tag; p = dict(cfg['physical_protocol'], epochs=epochs, device_count=len(GS), smoke_only=smoke,
                         fresh_scf_directory=str(out), smoke_energy_lower_Ha=cfg['energy_bounds_Ha'][0],
                         smoke_energy_upper_Ha=cfg['energy_bounds_Ha'][1])
    protocol = T/(tag+'_protocol.json'); dump(protocol,p)
    tmp = '/dev/shm/'+tag
    te = {**os.environ, **env, 'TMPDIR':tmp, 'PYTHONPYCACHEPREFIX':tmp+'/pycache',
          'JAX_COMPILATION_CACHE_DIR':tmp+'/jax', 'D3FT_TRACE_STAGES':'1'}
    cmd = ['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c8',
           'bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e5-repair','env',
           'CUDA_VISIBLE_DEVICES='+','.join(map(str,GS)), 'timeout',str(timeout_seconds),PY,'-u',
           str(R/'validation/workflows/response_design/reverse_response/e5_mg_sharded_driver.py'),
           '--protocol',str(protocol),'--stage','scf','--output',str(out)]
    rec = dict(time=now(), logical=logical, tag=tag, jobid=250285, gpus=GS, output=str(out),
               command=cmd, environment={k:te[k] for k in [*env,'TMPDIR','PYTHONPYCACHEPREFIX','JAX_COMPILATION_CACHE_DIR','D3FT_TRACE_STAGES']},
               timeout_seconds=timeout_seconds, status='running')
    with (T/(tag+'.log')).open('xb') as f:
        start=time.monotonic(); proc=subprocess.Popen(cmd,env=te,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        rec['pid']=proc.pid; dump(T/(tag+'_launch.json'),rec)
        status(logical+'_running', launch=str(T/(tag+'_launch.json')), output=str(out))
        code=proc.wait()
    used=len(GS)*(time.monotonic()-start)/3600; spent+=used
    rec.update(end=now(), returncode=code, GPU_hours=used, status='completed' if code==0 else 'failed')
    dump(T/(tag+'_launch.json'),rec)
    if code: raise RuntimeError(logical+' failed; output retained '+str(out))
    gates = [json.loads(line) for line in (out/'charge_gate.jsonl').read_text().splitlines()]
    if not {0,1,epochs}.issubset({x['step'] for x in gates}) or not all(x['passed'] and abs(x['charge']-96)<=1e-6 for x in gates):
        raise RuntimeError('Required actual-density initial/first/final charge certificates missing')
    return out, p

try:
    status('waiting_CPU_real_Si_restart_regression')
    gate = T/'cpu_regression/acceptance.json'; cpu = read(T/'cpu_regression_launch.json')
    deadline = time.monotonic()+600
    while not gate.exists():
        if time.monotonic()>deadline: raise RuntimeError('CPU regression missing; no GPU task launched')
        time.sleep(2)
    if not read(gate).get('passed'): raise RuntimeError('CPU regression failed; no GPU task launched')
    if any(not read(Path(f)).get('passed') for f in cfg['physical_protocol']['required_acceptance_files']):
        raise RuntimeError('Prior physical/response development gate missing')
    while not idle(): status('waiting_fixed_idle_cards'); time.sleep(20)
    smoke, _ = run_stage('mg_k2_charge_smoke100',100,1800,True)
    if not read(smoke/'smoke_acceptance.json').get('passed'): raise RuntimeError('Smoke not accepted')
    dump(T/'smoke_index.json',dict(time=now(), passed=True, source=str(smoke), releases_E6=False))
    note('E5修复后100步实际密度/能量短测PASS（GPU'+str(GS)+'），将以新唯一名从正确Gamma轨道/目标网格占据重跑4000步；短测不是驻点验收，E6仍等待。')
    print('E5 100步短测PASS。\n总电荷96±1e-6，有限能量下降；原八倍错误已拒绝。\n自动接4000步正式驻点验收，E6尚未释放。',flush=True)
    while not idle(): time.sleep(10)
    full,p = run_stage('mg_k2_correct_weight_scf4000',4000,21600,False)
    native=read(full/'point.json'); terminal=read(full/'status.json')
    if terminal.get('stage')!='completed' or not native.get('stationarity_pass') or native['residual']>1e-8 or native['acoustic_sum_rule_Ha_Bohr']>5e-8 or abs(native['charge']-96)>1e-6:
        raise RuntimeError('4000-step native acceptance failed; E6 not released')
    dump(T/'scf_acceptance.json',dict(time=now(), passed=True, source=str(full), point=native))
    dump(A/'mg_k2_protocol.json',p)
    dump(A/'scf_acceptance.json',dict(time=now(), passed=True, source=str(full), repair_root=str(T),scope='Fresh corrected-weight SCF native residual/ASR/charge; spectra/response pending'))
    status('corrected_k2_SCF_passed_spectrum_response_pending', passed=True, source=str(full), releases_E6=True)
    note('E5新占据4000步SCF实际电荷/驻点/ASR PASS；新源态'+str(full)+'，谱/耦合响应尚待接续，E6可以使用此新基态。')
except Exception as exc:
    status('failed', passed=False, error=str(exc), releases_E6=False)
    note('E5占据修复门控停止：'+str(exc)+'；保留全部失败，E6不释放。')
    raise
