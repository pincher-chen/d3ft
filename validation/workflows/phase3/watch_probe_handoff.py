"""Resume a paused sequence controller if a probe handoff process exits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--owner',type=int,required=True)
    ap.add_argument('--controller',type=int,required=True)
    ap.add_argument('--controller-token',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    owner=Path(f'/proc/{a.owner}/stat')
    identity=owner.read_text().split()[21] if owner.exists() else None
    while owner.exists():
        stat=owner.read_text().split()
        if stat[21]!=identity or stat[2]=='Z':break
        time.sleep(30)
    controller=Path(f'/proc/{a.controller}/cmdline')
    resumed=False
    if controller.exists() and a.controller_token.encode() in controller.read_bytes():
        os.kill(a.controller,signal.SIGCONT);resumed=True
    a.out.write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),
                                     owner=a.owner,controller=a.controller,
                                     resumed=resumed),indent=2))


if __name__=='__main__':main()
