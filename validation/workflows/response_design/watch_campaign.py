"""Refresh derived pilot summaries while the recorded controllers remain live."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign',type=Path)
    a=ap.parse_args();r=a.campaign.resolve()
    monitor=r/'analysis_watch'
    monitor.mkdir(exist_ok=False)
    while True:
        controllers=[]
        launches=json.loads((r/'launches.json').read_text())
        for extra in ['baseline_native_r2_launch.json','baseline_k1_launch.json',
                      'geometry_fd_xc8_launch.json','native_anchor_audit_launch.json']:
            p=r/extra
            if p.exists():launches.append(json.loads(p.read_text()))
        for launch in launches:
            pid=launch.get('controller_pid',launch.get('pid'))
            p=Path('/proc')/str(pid)/'status'
            try:
                live='State:\tZ' not in p.read_text()
            except FileNotFoundError:
                live=False
            controllers.append(dict(pid=pid,live=live,record=launch.get('record')))
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_campaign.py')),str(r)],
                              capture_output=True,text=True)
        (monitor/'latest_summary.log').write_text(result.stdout+result.stderr)
        snapshot=dict(time=datetime.datetime.now().isoformat(),controllers=controllers,
            analysis_returncode=result.returncode,
            state='watching' if any(x['live'] for x in controllers) else 'all_recorded_controllers_terminal',
            scope='Controller liveness and derived results; not a physical acceptance or proof of compute-process liveness.')
        tmp=monitor/'status.tmp';tmp.write_text(json.dumps(snapshot,indent=2)+'\n');tmp.replace(monitor/'status.json')
        if not any(x['live'] for x in controllers):break
        time.sleep(60)


if __name__=='__main__':main()
