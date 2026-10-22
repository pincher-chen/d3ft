"""Release only this repair's idle-card reservations after its terminal stage."""
import argparse, datetime, json, time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
P=Path(__file__).resolve().parents[3]
path=P/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
terminal={'failed','corrected_k2_SCF_passed_spectrum_response_pending'}
while True:
    status=json.loads((a.root/'status.json').read_text())
    if status['stage'] in terminal:break
    time.sleep(10)
rows=json.loads(path.read_text());removed=[]
for key in ('0','5'):
    if rows.get(key,{}).get('status_file')==str(a.root/'status.json'):
        rows.pop(key);removed.append(int(key))
tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2)+'\n');tmp.replace(path)
(a.root/'reservation_release.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),
    removed=removed,after_stage=status['stage'],scope='CPU reservation release only; no signal to any GPU task'),indent=2)+'\n')
