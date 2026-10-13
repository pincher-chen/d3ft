"""Separate parameter-FD agreement from signal size at production settings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();rows=[];sources={}
    names=['theta_s','theta_p','theta_d','c:beta','c:gamma','x:kappa','x:mu']
    for p in sorted(args.results.glob('*production*fd*certificate.json')):
        d=json.loads(p.read_text())
        if not all(k in d for k in ['base','parameter_difference','rows']):continue
        base=Path(d['base'])/'summary.json';s=json.loads(base.read_text())
        sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
        sources[str(base.resolve())]=hashlib.sha256(base.read_bytes()).hexdigest()
        active=[names[i] for i,x in enumerate(d['parameter_difference']) if abs(x)>1e-15]
        direction=active[0] if len(active)==1 else 'mixed'
        for row in d['rows']:
            observable=row['observable'];budget=1e-4*abs(s['a0_A']) if observable=='a0_A' else .1
            signal=abs(row['actual_difference'])
            rows.append(dict(certificate=p.name,material=s['material'],direction=direction,
                observable=observable,fd_agreement_passed=row['passed'],signal=signal,
                numerical_resolution_budget=budget,signal_to_budget=signal/budget,
                above_budget=signal>budget,relative_error=row['relative_error'],
                joint_gate_passed=bool(row['passed'] and signal>budget)))
    groups={}
    for r in rows:groups.setdefault(r['certificate'],[]).append(r)
    report={'scope':'Observed FD response compared with the declared numerical resolution budget; neither statistical confidence nor a rigorous error bound.',
        'lattice_budget':'1e-4 times baseline a0 in Angstrom', 'elastic_budget_GPa':.1,
        'sources':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'directions':[dict(certificate=k,material=v[0]['material'],direction=v[0]['direction'],
                          all_fd_agreement_passed=all(x['fd_agreement_passed'] for x in v),
                          all_four_joint_gates_passed=len(v)==4 and all(x['joint_gate_passed'] for x in v),
                          unresolved_observables=[x['observable'] for x in v if not x['joint_gate_passed']])
                      for k,v in groups.items()]}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'audit.json').write_text(json.dumps(report,indent=2))
    with (args.out/'components.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['certificate']);w.writeheader();w.writerows(rows)
    for item in report['directions']:print(json.dumps(item))


if __name__=='__main__':main()
