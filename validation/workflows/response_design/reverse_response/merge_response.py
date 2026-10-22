"""Gate response data against the pre-existing native baseline and FD direction."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args()
root=a.root;cfg=json.loads((root/'protocol.json').read_text());workers=['reverse','forward01','forward23','forward45']
def put(name,data):(root/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
start=time.monotonic()
while True:
 statuses={w:json.loads((root/w/'status.json').read_text()) if (root/w/'status.json').exists() else {'stage':'waiting'} for w in workers}
 put('progress.json',dict(time=datetime.datetime.now().isoformat(),workers=statuses))
 failed={w:s for w,s in statuses.items() if s['stage']=='failed'}
 if failed:put('acceptance.json',dict(passed=False,stage='worker_failed',failures=failed));raise SystemExit(1)
 if all((root/w/'result.json').exists() for w in workers):break
 if time.monotonic()-start>40000:put('acceptance.json',dict(passed=False,stage='timeout'));raise SystemExit(2)
 time.sleep(20)
r=json.loads((root/'reverse/result.json').read_text());reverse=np.asarray(r['results']['gap_gradient_eV'])
forward=np.empty(6);records=[]
for w in workers[1:]:
 d=json.loads((root/w/'result.json').read_text());records.append(d)
 for axis,row in d['results'].items():forward[int(axis)]=row['gap_derivative_eV']
native=json.loads(Path(cfg['native_gradient']).read_text());reference=np.asarray(native['gap_gradient_eV'])
direction=np.asarray(cfg['direction']);fd=native['registered_D3ft_FD_eV']
absolute=abs(reverse-reference);allowed=cfg['component_relative_tolerance']*abs(reference)+cfg['component_absolute_floor_eV']
mode_error=abs(forward-reverse);mode_allowed=1e-5+1e-4*abs(reverse)
fd_projection=float(reverse@direction);fd_relative=abs(fd_projection-fd)/abs(fd)
passed=bool(np.all(absolute<=allowed) and np.all(mode_error<=mode_allowed) and fd_relative<=.01)
put('acceptance.json',dict(time=datetime.datetime.now().isoformat(),passed=passed,
 reverse_gradient_eV=reverse.tolist(),same_program_forward_gradient_eV=forward.tolist(),
 native_gradient_eV=reference.tolist(),native_absolute_difference_eV=absolute.tolist(),
 component_error_budget_eV=allowed.tolist(),same_program_mode_difference_eV=mode_error.tolist(),
 registered_FD_direction_eV=fd,reverse_direction_eV=fd_projection,relative_FD_difference=fd_relative,
 adjoint_record=r['results']['response_record'],response_right_hand_sides={'reverse':1,'forward':6},
 timings=dict(reverse=r['solve_and_contract_seconds'],forward_worker_seconds=[d['solve_and_contract_seconds'] for d in records],
              scope='Parallel deployment and some first-use contractions; not a warmed same-device AD-mode speedup'),
 scope='Single registered Gamma point; clamped ions, fixed occupations/basis; native component and one FD projection acceptance. No full-BZ design or training result'))
print('RESPONSE_ACCEPTANCE',passed,flush=True)
raise SystemExit(0 if passed else 1)
