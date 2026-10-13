"""Rebuild current QE references from full-precision raw XML, reusing input-identical points."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RESULT=ROOT/'validation/results/phase3_20260908'
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--record',type=Path,required=True)
a=ap.parse_args();a.record.mkdir(parents=True,exist_ok=False)
sources=['qe_bulk_AlAs_ecut200','qe_elastic_AlAs_ecut200','qe_bulk_AlP_ecut200',
         'qe_bulk_GaP_ecut200','qe_elastic_GaP_ecut200','qe_bulk_Si','qe_elastic_Si',
         'qe_bulk_diamond_ecut200','qe_elastic_diamond_ecut200','qe_bulk_BN_ecut200','qe_bulk_SiC_ecut200']
plan=[]
for tag in sources:
    source=RESULT/tag;target=tag+'_xml'
    if (RESULT/target).exists():raise FileExistsError(target)
    meta=json.loads((source/'metadata.json').read_text());summary=json.loads((source/'summary.json').read_text())
    arguments=meta['arguments']
    cutoffs={float(re.search(r'ecutwfc\s*=\s*([\d.]+)',p.read_text(),re.I)[1]) for p in source.glob('*/scf.in')}
    if len(cutoffs)!=1:raise ValueError('Source cutoff is missing or inconsistent')
    cutoff=cutoffs.pop()
    if 'cutoff' in arguments and cutoff!=arguments['cutoff']:raise ValueError('Metadata/input cutoff mismatch')
    mode=arguments.get('mode','bulk' if tag.startswith('qe_bulk_') else 'elastic')
    cmd=[sys.executable,'-u',str(HERE/'qe_elastic.py'),'--material',arguments['material'],
         '--mode',mode,'--cutoff',str(int(cutoff)),
         '--a0',str(summary['a0_A']),'--reuse-from',str(source),'--tag',target]
    if arguments.get('bands') is not None:cmd+=['--bands',str(arguments['bands'])]
    plan.append(dict(source=str(source),target=target,command=cmd,
        source_summary_sha256=hashlib.sha256((source/'summary.json').read_bytes()).hexdigest()))
(a.record/'plan.json').write_text(json.dumps(plan,indent=2))
(a.record/'driver_snapshot.py').write_text(Path(__file__).read_text())
for item in plan:
    print('REFRESH',item['target'],flush=True)
    with (a.record/(item['target']+'.log')).open('x') as log:
        subprocess.run(item['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    (a.record/(item['target']+'.done')).write_text('Full-precision result written; numerical convergence is a separate check.\n')
print('REFRESH_DONE',flush=True)
