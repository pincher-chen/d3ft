"""Compare two flat observable-probe JSON records without changing them."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference',type=Path)
    ap.add_argument('candidate',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--atol',type=float,default=1e-10)
    a=ap.parse_args()
    values=[json.loads(p.read_text()) for p in [a.reference,a.candidate]]
    keys=['energy_Ha','pressure_GPa','residual','forces','dE_dp']
    differences={k:float(np.max(np.abs(
        np.asarray(values[0][k])-np.asarray(values[1][k])))) for k in keys}
    result=dict(reference=str(a.reference),candidate=str(a.candidate),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [a.reference,a.candidate]},
        differences=differences,atol=a.atol,
        passed=max(differences.values())<=a.atol,
        scope='Small synthetic integration case; numerical parity only')
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
