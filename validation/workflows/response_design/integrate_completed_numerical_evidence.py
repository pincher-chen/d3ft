"""Publish qualified partial results without waiting for entire campaigns."""
import csv
import datetime
import json
from pathlib import Path
import shutil
from completed_numerical_evidence import P,R,collect,gradient_tex,kmesh_tex,partial_rows,sha

D=P/'paper_rewriting_output/final_paper'
M=P.parents[1]/'d3ft2-nmi/papar/sn-article-template'

def insert_block(text,start,end,anchor,body):
    block=start+'\n'+body+'\n'+end+'\n\n'
    if start in text:
        a=text.index(start);b=text.index(end,a)+len(end)
        return text[:a]+block.rstrip('\n')+text[b:]
    if text.count(anchor)!=1:raise ValueError('TeX insertion anchor changed: '+anchor)
    return text.replace(anchor,block+anchor,1)

def integrate():
    data=collect()
    data['source_hashes'][str(Path(__file__).resolve())]=sha(__file__)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dump=json.dumps(data,indent=2)+'\n'
    (R/'completed_evidence_snapshot.json').write_text(dump)
    (R/('completed_evidence_snapshot_'+stamp+'.json')).write_text(dump)
    rows=partial_rows(data)
    with (R/'numerical_verification_completed.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['group','metric','production','refined','difference','unit'])
        writer.writeheader();writer.writerows(rows)
    header='\\subsection{Completed representative numerical checks}\n\\label{sec:representative-refinement}\n'
    header+='These completed checks distinguish evidence admission from completion of the full numerical campaigns.\n'
    header+='The registered pairwise gap budget is 5 meV and the directional-gradient budget is 1\\% plus $10^{-4}$ eV per unit strain; the absolute design-target budget remains $\\pm1$ meV.\n\n'
    fragment=header
    g=data['gradient'];k=data['kmesh']
    if g:fragment+=gradient_tex(g)+'\n'
    if k:fragment+=kmesh_tex(k)
    r1='''Refining the XC quadrature from $576^3$ to $720^3$ at the same
fixed-basis response cell gave a reoptimized directional finite difference
of $-5.443429612$ eV per unit strain at $h/2=0.0005$.
It differs from the production adjoint projection by
$1.49\\times10^{-5}$ eV per unit strain; halving the finite-difference
step changes the refined slope by 0.000941\\%.
These completed checks support one directional response under XC refinement
(Supplementary Information). The analytic adjoint at the refined quadrature
remains incomplete, so this comparison does not certify its full gradient.
'''
    r3='''At the starting geometry, a separately stationary $6^3$ density
gave a $\\Gamma$ transition of 4.404709619 eV, compared with
4.404704264 eV at $4^3$, a change of 0.00535 meV.
This completed starting-point check uses the same Miller basis and
is reported in the Supplementary Information. The complete $6^3$
sampled gaps and reverse-endpoint comparison remain incomplete;
the single-transition result does not resolve endpoint target transfer.
'''
    for directory,stem in [(D,'main'),(M,'sn-article')]:
        (directory/'numerical_verification.tex').write_text(fragment)
        main=directory/(stem+'.tex');text=main.read_text()
        if g:text=insert_block(text,'% BEGIN COMPLETED XC REFINEMENT','% END COMPLETED XC REFINEMENT',
            'On the same physical H100 GPU, a warmed and synchronized full-gradient',r1)
        if k:text=insert_block(text,'% BEGIN COMPLETED K6 START','% END COMPLETED K6 START',
            'A separately executed FD baseline illustrates a limited repair.',r3)
        marker='% Representative numerical verification 20261005'
        if marker not in text:
            anchor='\\subsection{Calibration and evidence preservation}'
            if text.count(anchor)!=1:raise ValueError('Methods insertion anchor changed')
            paragraph=marker+'\nRepresentative sampling and XC-quadrature checks retain the original basis\nwithin each comparison. Their completed measurements, source qualifications\nand outstanding response or full-mesh checks are reported explicitly in the\nSupplementary Information. Pairwise numerical budgets do not replace the\noriginal design target or establish continuum convergence.\n\n'
            text=text.replace(anchor,paragraph+anchor,1)
        main.write_text(text)
        si=directory/'supplementary.tex';text=si.read_text()
        if '\\input{numerical_verification}' not in text:
            anchor='\\input{response_design-r1-gradient-table.tex}'
            if text.count(anchor)!=1:raise ValueError('SI response-table anchor changed')
            text=text.replace(anchor,anchor+'\n\n\\input{numerical_verification}',1)
        si.write_text(text)
    note='''# Completed numerical evidence integrated into the response draft

The four main Results themes are unchanged. R1 now includes the completed
XC720 reoptimized directional finite differences; R3 includes the certified
k6 starting state and its completed Gamma eigensolution. Two SI tables retain
the values and state the missing refined adjoint and full-mesh/endpoint checks.

All admitted values are regenerated from raw JSON/checkpoints by
`validation/workflows/response_design/integrate_completed_numerical_evidence.py`.
Source hashes and gate checks are in `completed_evidence_snapshot.json`;
machine-readable values are in `numerical_verification_completed.csv`.
No complete campaign or submission-readiness claim is made.

Canonical and mirror TeX preserve their respective bibliography names.
The automatic final publisher preserves admitted partial evidence when a
campaign has failed or remains incomplete.

Terminology: production adjoint = XC576 reverse response; refined FD = XC720
independently reoptimized finite difference; k6 Gamma transition = one Gamma
eigensolution at the own stationary k6 density, not a full-zone minimum gap.

Claim/evidence: FD step stability is supported by h and h/2; production-response
consistency under XC refinement is supported by production AD versus refined FD;
starting-state Gamma stability is supported by certified k4/k6 densities.
Refined analytic response and complete k6 endpoint/gain transfer need results.
'''
    (R/'completed_evidence_integration.md').write_text(note)
    (D/'response_COMPLETED_NUMERICAL_UPDATE_20261005.md').write_text(note)
    print(json.dumps(dict(integrated=True,gradient=bool(g),kmesh=bool(k),bound_files=len(data['source_hashes']),time=data['time']),indent=2))

if __name__=='__main__':integrate()
