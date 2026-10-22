"""CPU-only durable status/accounting recorder; submits no computations."""
from pathlib import Path
import json,datetime,time,hashlib
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e3_geometry'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def dump(p,d):
 t=p.with_suffix(p.suffix+'.recorder.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
while True:
 now=datetime.datetime.now();si_coarse=read(E/'si_acceptance.json');si_fine=read(E/'si_acceptance_refined.json')
 refine=read(E/'si_refinement_status.json');mode=read(E/'mg_mode_acceptance.json');fd=read(E/'mg_fd_acceptance.json')
 active=[];spent=0.;upper=0.;records=[]
 for path in sorted((E/'commands').glob('*.json')):
  r=read(path)
  if not r or 'start' not in r:continue
  item=dict(tag=r.get('tag',path.stem),jobid=r.get('jobid'),gpu=r.get('gpu'),pid=r.get('pid'),status=r.get('status'))
  if 'GPU_hours' in r:item['GPU_hours']=r['GPU_hours'];spent+=r['GPU_hours']
  elif 'end' in r:
   # Refinement wave completion can lag a peer's actual exit. Label this as an
   # upper bound, rather than inventing precise active device measurements.
   h=max(0.,(datetime.datetime.fromisoformat(r['end'])-datetime.datetime.fromisoformat(r['start'])).total_seconds()/3600)
   item['GPU_hours_upper_bound']=h;upper+=h
  elif r.get('status')=='running':
   h=max(0.,(now-datetime.datetime.fromisoformat(r['start'])).total_seconds()/3600)
   item['inflight_GPU_hours_upper_bound']=h;upper+=h;active.append(item['tag'])
  records.append(item)
 si=bool(si_fine.get('passed') or si_coarse.get('passed'));formal=si and bool(mode.get('passed')) and bool(fd.get('passed'))
 output=dict(time=now.isoformat(),stage='derivative_gates_PASS_formal_pending' if formal else 'development_validation',
  jobid=250285,formal_timing_submitted=False,formal_timing_gate_PASS=formal,
  geometry_interface_PASS=read(E/'geometry_layout_checks.json').get('passed'),
  Si_mode_PASS=read(E/'si_mode_acceptance.json').get('passed'),Si_coarse_FD_PASS=si_coarse.get('passed'),
  Si_refined_FD_PASS=si_fine.get('passed'),Si_overall_derivative_PASS=si,
  Si_refinement=refine.get('stage'),Mg_mode_memory_translation_PASS=mode.get('passed'),
  Mg_mixed_FD_PASS=fd.get('passed'),Mg_queue=read(E/'mg_queue_status.json'),Mg_FD_queue=read(E/'mg_fd_queue_status.json'),
  measured_step_GPU_hours=spent,additional_or_inflight_GPU_hours_upper_bound=upper,
  active_tags=active,commands=records,
  E2='complete latest agreed scope: 9 MgSiN2 PASS, 6 calibration states replaced by E7',
  hn48_used=False,scope='Development records only, not scaling measurements')
 dump(E/'status.json',output)
 if formal:
  print('E3 all derivative gates PASS; formal timing remains unsubmitted',flush=True);break
 if refine.get('stage')=='failed':print('E3 refinement failed; raw data retained',flush=True);break
 if si and any(read(E/name).get('stage')=='failed' and read(E/name).get('attempt')=='refined'
               for name in ['mg_queue_status.json','mg_fd_queue_status.json']):
  print('Refined-attempt Mg queue reports failure; no new formal timing submitted',flush=True);break
 time.sleep(20)
