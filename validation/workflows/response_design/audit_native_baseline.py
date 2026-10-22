"""Wait for the real upstream run and compare exported data to archived JSON."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan',type=Path)
    ap.add_argument('--record',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text())
    a.record.mkdir(parents=True,exist_ok=False)
    (a.record/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def status(state,**kw):
        p=a.record/'status.tmp';p.write_text(json.dumps(dict(state=state,
            time=datetime.datetime.now().isoformat(),**kw),indent=2)+'\n')
        p.replace(a.record/'status.json')
    status('waiting_for_native_anchor')
    while True:
        dep=Path(plan['after_record'])/'status.json'
        if dep.exists():
            st=json.loads(dep.read_text())
            if st['state']=='completed':break
            if st['state']=='failed':status('dependency_failed',details=st);return
        try:os.kill(plan['after_pid'],0)
        except ProcessLookupError:status('dependency_controller_missing');return
        time.sleep(15)
    status('exporting_upstream_json')
    with (a.record/'export.log').open('x') as f:
        result=subprocess.run(plan['export_command'],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:status('export_failed',returncode=result.returncode);return
    name='silicon_Ecut30_kgrid4_displacefalse'
    sources={};rows=[]
    for suffix in ['_scfres.json','_derivatives_xc_kappa_plot.json',
                   '_derivatives_psp_cloc1_plot.json','_derivatives_strain_plot.json']:
        raw=Path(plan['output'])/(name+suffix)
        archived=Path(plan['reference'])/(name+suffix)
        d,b=[json.loads(p.read_text()) for p in [raw,archived]]
        for p in [raw,archived]:sources[str(p)]=digest(p)
        keys=['ρ','forces','energies','eigenvalues','eigenvalues_bs','εF','occupation'] if suffix=='_scfres.json' else [k for k in b if k!='θ']
        if suffix=='_scfres.json':d,b=d['y'],b['y']
        for key in keys:
            v,w=[np.asarray(obj[key],float).ravel() for obj in [d,b]]
            if v.shape!=w.shape or not (np.isfinite(v).all() and np.isfinite(w).all()):
                raise ValueError(f'{suffix}/{key}: incompatible dimensions or nonfinite data')
            error=float(np.abs(v-w).max()) if v.size else 0.
            scale=float(np.abs(w).max()) if w.size else 0.
            atol=plan['protocol']['absolute_tolerance'];rtol=plan['protocol']['relative_tolerance']
            rows.append(dict(file=suffix,quantity=key,entries=len(v),max_absolute_error=error,
                reference_max_abs=scale,budget=atol+rtol*scale,passed=error<=atol+rtol*scale))
    passed=all(row['passed'] for row in rows)
    (a.record/'comparison.json').write_text(json.dumps(dict(passed=passed,rows=rows,
        source_hashes=sources,protocol=plan['protocol'],script_sha256=digest(Path(__file__)),
        scope='Native Figure-1 non-displaced Si XC/projector/strain forward-AD anchor; six application benchmarks and D3FT comparison remain separate.'),indent=2)+'\n')
    status('completed_native_anchor_audit',passed=passed)


if __name__=='__main__':main()
