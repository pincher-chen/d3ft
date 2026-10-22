"""Release the pre-existing E1 CPU hold only after this resource wave ends."""
from pathlib import Path
import datetime,json,os,signal,time
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e6_k2_hydro'
hold=P/'validation/results/response_design_20261015_e5_distributed/E1_CPU_observer_hold.json'
while True:
    row=json.loads((E/'status.json').read_text())
    if row['stage'] in ('failed','k2_hydro_pilot_finished_k4_literature_pending'):break
    time.sleep(30)
record=json.loads(hold.read_text());pid=record['pid'];cmd=Path(f'/proc/{pid}/cmdline')
if cmd.exists() and b'e1_finish_extension.py' in cmd.read_bytes():
    os.kill(pid,signal.SIGCONT)
    record.update(resumed=datetime.datetime.now().isoformat(),status='resumed_after_E6_resource_wave')
    hold.write_text(json.dumps(record,indent=2)+'\n')
    (E/'E1_CPU_release.json').write_text(json.dumps(record,indent=2)+'\n')
    print('E1 CPU-only observer resumed after E6 wave; GPU availability still checked by E1.',flush=True)
