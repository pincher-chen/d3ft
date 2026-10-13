"""Validate lightweight multi-process spectra against completed serial spectra."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path);ap.add_argument('reference',type=Path)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--devices',required=True)
    ap.add_argument('--compare-only',action='store_true',help='Audit existing completed shards without recomputing')
    args=ap.parse_args()
    meta=json.loads((args.reference/'metadata.json').read_text())
    if Path(meta['source']).resolve()!=args.source.resolve():raise ValueError('Reference source mismatch')
    n=len(meta['mesh']);indices=sorted(set([0,1,n//2,n-1]))
    if not args.compare_only:subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('sharded_spectrum.py')),
                    str(args.source),'--out',str(args.out),'--devices',args.devices,
                    '--indices',json.dumps(indices)],check=True)
    new_meta=json.loads((args.out/'metadata.json').read_text())
    for key in ('source_metadata_sha256','density_sha256','occupied_bands'):
        if new_meta[key]!=meta[key]:raise ValueError('Reference provenance mismatch: '+key)
    if abs(new_meta['electrons']-meta['electrons'])>1e-12:raise ValueError('Electron count changed')
    if sorted(new_meta['indices'])!=indices:raise ValueError('Validation indices changed')
    for name,digest in new_meta['workflow_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Shard source code changed')
    if new_meta['merge_script_sha256']!=hashlib.sha256(Path(__file__).with_name('sharded_spectrum.py').read_bytes()).hexdigest():
        raise ValueError('Merge source code changed')
    rows=[]
    for i in indices:
        old=json.loads((args.reference/f'k_{i:04d}.json').read_text())
        new=json.loads((args.out/f'k_{i:04d}.json').read_text())
        if max(abs(x-y) for x,y in zip(old['k_frac'],new['k_frac']))>1e-12:
            raise ValueError('k coordinates changed')
        a,b=old['eigenvalues_Ha'],new['eigenvalues_Ha']
        if len(a)!=len(b):raise ValueError('Band count changed')
        error=max(abs(x-y) for x,y in zip(a,b))
        rows.append(dict(index=i,max_eigenvalue_difference_Ha=error,passed=error<=1e-7,
                         max_residual_Ha=max(new['residuals_Ha'])))
    names=['check_occupations.py','sharded_spectrum.py','variational_ks.py','lowest_eigensystem.py']
    report=dict(passed=all(r['passed'] for r in rows),rows=rows,atol_Ha=1e-7,
                source=str(args.source.resolve()),reference=str(args.reference.resolve()),
                code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names})
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
