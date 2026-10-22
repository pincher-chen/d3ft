"""Immediately use GPU1/4/5/6 for registered diamond fine control."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import e7_ct_pilot as p
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();T=a.root;GPUS=[1,4,5,6]
def state(stage,**kw):p.dump(T/'status.json',dict(time=p.now(),stage=stage,jobid=250285,gpus=GPUS,**kw))
def owned_available(gs):
 owners=p.load(p.E/'gpu_reservations.json')
 for g in gs:
  owner=owners.get(str(g),{})
  if owner and owner.get('status_file')!=str(T/'status.json') and p.load(Path(owner['status_file'])).get('stage') not in ('failed','complete'):return False
 r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr)
 rows={int(x[0]):[int(v) for v in x[1:]] for line in r.stdout.splitlines() if len(x:=line.split(','))==3}
 p.dump(T/'GPU_probe.json',dict(time=p.now(),rows=rows));return all(rows[g][0]<=3 and rows[g][1]==0 for g in gs)
p.priority_pending=lambda:None;p.available=owned_available
try:
 coarse=p.E/'runs'/p.selected_tag('diamond','cheap')
 if not p.load(coarse/'native_acceptance.json').get('passed'):raise RuntimeError('Diamond cheap native gate missing')
 state('diamond_four_card_fine_starting')
 with ThreadPoolExecutor(max_workers=1) as pool:
  future=pool.submit(p.run,'diamond','fine',GPUS,28800,p.load(coarse/'summary.json')['a0_A'],run_tag='diamond_fine_idle4')
  while not future.done():state('diamond_four_card_fine_running');time.sleep(15)
  work=future.result()
 if work is None:raise RuntimeError('Diamond fine child failed; original log retained')
 if not p.load(work/'native_acceptance.json').get('passed'):raise RuntimeError('Diamond native gate failed')
 result=p.compare('diamond');p.report();state('complete',passed=result['passed'],result=result)
 p.note('E7 diamond独立加密GPU1/4/5/6结束，原门 '+('PASS' if result['passed'] else 'FAIL')+'；训练/LOO另待。')
except Exception as exc:state('failed',passed=False,error=str(exc));p.note('E7 diamond四卡复核失败：'+str(exc)+'；记录保留。');raise
finally:
 Q=p.E/'gpu_reservations.json';rows=p.load(Q)
 for g in GPUS:
  if rows.get(str(g),{}).get('status_file')==str(T/'status.json'):rows.pop(str(g))
 p.dump(Q,rows)
