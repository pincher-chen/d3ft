"""CPU-only real-input, objective-definition and budget checks for E5 transfer."""
import argparse
import ast
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from e5_k4_transfer_queue import summarize_transfer


def main(B):
    c=json.loads((B/'preparation.json').read_text())
    W=Path(__file__).resolve().parent
    for f in [W/'e5_k4_transfer_queue.py', W/'reverse_response/e5_k4_transfer_point.py']:
        ast.parse(f.read_text())
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    checks={};models=[]
    for label,entry in c['states'].items():
        S=Path(entry['source']);m=json.loads((S/'metadata.json').read_text())
        pt=json.loads((S/'point.json').read_text());lines=(S/'input.xyz').read_text().splitlines()
        cell=list(map(float,re.search('Lattice="([^"]+)"',lines[1])[1].split()))
        pos=[list(map(float,line.split()[1:4])) for line in lines[2:]]
        diff=max(abs(a-b) for a,b in zip(cell,sum(m['physical']['cell_angstrom'],[])))
        diff=max(diff,max(abs(a-b) for a,b in zip(sum(pos,[]),sum(m['physical']['positions_angstrom'],[]))))
        checks[label+'_actual_geometry_matches_metadata']=diff<1e-11
        checks[label+'_stationary_and_hash']=pt['stationarity_pass'] and sha(S/'state.pkl')==m['state_sha256']
        checks[label+'_physical_program']=(m['physical']['cutoff']==100 and
            m['physical']['grid']==[144] and m['physical']['xc_grid_factor']==4)
        models.append((sha(S/'mask.npy'),m['frozen_radial_program'],m['physical']['pp']))
    checks['same_frozen_basis_radial_PP']=models[0]==models[1]
    checks['timeouts_under_budget']=(2*8*(c['SCF_timeout_seconds']+25)+2*8*(c['spectra_timeout_seconds']+25))/3600+.2<60
    s={'fundamental_gap_eV':3.9925};e={'fundamental_gap_eV':4.0125}
    r=summarize_transfer(s,e,c)
    checks['absolute_vs_increment_errors_distinct']=(abs(r['increment_error_meV'])<1e-10
        and r['target_error_meV']>1 and not r['target_transfer_passed'])
    r=summarize_transfer(s,{'fundamental_gap_eV':c['target_eV']-.0001},c)
    checks['signed_target_error']=abs(r['target_error_meV']+.1)<1e-9 and r['target_transfer_passed']
    checks['old_Gamma_observable_explicit']='DIRECT' in r['comparison_caveat'] and c['old_Gamma_only']['basis_and_target_differ']
    checks['two_states_only_no_hydro_nu']=len(c['states'])==2 and all(len(e['theta'])==54 for e in c['states'].values())
    d=dict(time=dt.datetime.now().astimezone().isoformat(),passed=all(checks.values()),checks=checks,
        scope='CPU static real-input/provenance/metric/budget checks; no GPU physics acceptance')
    (B/'CPU_preflight.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d,indent=2))
    if not d['passed']:raise ValueError('Preflight failed; no GPU queue allowed')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    main(ap.parse_args().root)
