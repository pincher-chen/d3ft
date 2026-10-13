"""Turn staged GPU diagnostic logs into a small immutable evidence record."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect(path):
    text=path.read_text(errors='replace')
    stages=[line.strip().split('D3FT_STAGE ',1)[1]
            for line in text.splitlines() if 'D3FT_STAGE ' in line]
    allocations=[int(x) for x in re.findall(
        r'allocate ([0-9.]+)GiB \(rounded to ([0-9]+)\)',text)
        for x in [x[1]]]
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                stages=stages,failed_allocation_bytes=allocations,
                oom='Out of memory while trying to allocate' in text,
                normal_point='POINT {' in text)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('logs',nargs='+',type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    records=[inspect(p) for p in a.logs]
    result=dict(records=records,
        interpretation=(
            'Stages identify the last completed boundary only. They localize '
            'the failed allocation but do not certify numerical correctness.'),
        all_complete=all(r['normal_point'] and not r['oom'] for r in records))
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
