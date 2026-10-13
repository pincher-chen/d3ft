"""Restore the sequence if the one-off handoff controller unexpectedly exits."""
import datetime
import json
import os
from pathlib import Path
import signal
import time
r=Path('validation/results/phase3_20260908')
owner=json.loads((r/'priority_memory_handoff_launch.json').read_text())['pid']
controller=1884532
p=Path(f'/proc/{owner}/stat')
identity=p.read_text().split()[21] if p.exists() else None
while p.exists():
    stat=p.read_text().split()
    if stat[21]!=identity or stat[2]=='Z':break
    time.sleep(30)
c=Path(f'/proc/{controller}/cmdline')
if c.exists() and b'refinement_4gpu_0908_plan.json' in c.read_bytes():
    os.kill(controller,signal.SIGCONT)
(r/'memory_handoff_watchdog_done.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),owner=owner,controller=controller)))
