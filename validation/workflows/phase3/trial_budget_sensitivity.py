"""Diagnose loss sensitivity to assumed total numerical budgets; never accept a fit."""
import argparse
import hashlib
import json
from pathlib import Path
from loss_intervals import reduction_interval
from bounded_trial import decide


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['baseline','proposal','candidate','out']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    load=lambda p:json.loads(Path(p).read_text())
    base,prop,candidate=map(load,[a.baseline,a.proposal,a.candidate])
    report={'accepted':False,'scope':'Sensitivity only: assumes total observable errors are bounded by the stated tolerances. Pairwise convergence tolerances do not establish these bounds; no certified acceptance.',
            'budget_assumption':'a0: 1e-4 times computed value; elastic observables: 0.1 GPa, independently at baseline and candidate',
            'inputs':{k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ['baseline','proposal','candidate']},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'materials':[]}
    for m in sorted(prop['input_fingerprints']):
        path=Path(candidate[m])/'summary.json'
        if not path.exists():
            report['materials'].append({'material':m,'state':'pending'});continue
        old,new=load(Path(base[m])/'summary.json'),load(path)
        if new['parameter_vector']!=prop['candidate_vector']:raise ValueError('Candidate vector mismatch')
        rows=[r for r in prop['training_rows'] if r['material']==m]
        rb,rc,eb,ec=[],[],[],[]
        for r in rows:
            key=r['observable'];denom=abs(r['reference']*r['sigma'])
            rb.append((old[key]-r['reference'])/denom);rc.append((new[key]-r['reference'])/denom)
            eb.append((abs(old[key])*1e-4 if key=='a0_A' else .1)/denom)
            ec.append((abs(new[key])*1e-4 if key=='a0_A' else .1)/denom)
        jacobian=[[g/r['sigma'] for g in r['gradient']] for r in rows]
        step=[x-y for x,y in zip(prop['candidate_vector'],prop['starting_vector'])]
        scenarios=[]
        for f in [.5,1.,2.]:
            berrors,cerrors=[f*x for x in eb],[f*x for x in ec]
            scenarios.append(dict(multiplier=f,**reduction_interval(rb,rc,berrors,cerrors),
                conditional_trust_region=decide(rb,rc,jacobian,step,berrors,cerrors,
                                               [[0.]*len(step) for _ in rows]),
                jacobian_assumption='Optimistic exact Jacobian; zero Jacobian error is a sensitivity assumption, not verified'))
        report['materials'].append({'material':m,'state':'diagnostic_only',
            'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'baseline_summary_sha256':hashlib.sha256((Path(base[m])/'summary.json').read_bytes()).hexdigest(),
            'scenarios':scenarios})
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
