"""Reuse only a registered, completed, density-bound full spectrum.

All other outputs follow the original guarded serial/shard solver unchanged.
Registration is not acceptance: full raw mesh/gap/residual checks are replayed.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
from audit_trial import audit_spectrum, read, sha

def reuse(source,output,registration):
    entry=registration['entries'].get(str(output.resolve()))
    if entry is None:raise ValueError('Existing spectrum was not registered for reuse')
    if Path(entry['source']).resolve()!=source.resolve():raise ValueError('Registered density source mismatch')
    for path,digest in entry['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Registered source changed: '+path)
    terminal=read(Path(entry['sequence_record'])/'status.json')
    if (terminal.get('state')!='sequence_exited' or
        [r.get('name') for r in terminal.get('results',[])]!=[entry['experiment']] or
        any(r.get('returncode')!=0 for r in terminal.get('results',[]))):
        raise ValueError('Precomputation did not exit successfully')
    if read(Path(entry['sequence_record'])/'plan.json')!=read(entry['sequence_plan']):
        raise ValueError('Precomputed execution differs from registered plan')
    state=read(output.with_name(output.name+'_execution')/'status.json')
    if state.get('state') not in ['completed_guarded_parallel_spectrum','completed_serial_spectrum']:
        raise ValueError('Precomputed guarded solver has no successful terminal state')
    if Path(state['source']).resolve()!=source.resolve() or Path(state['out']).resolve()!=output.resolve():
        raise ValueError('Guarded solver source/output mismatch')
    snapshot=read(output.with_name(output.name+'_execution')/'inputs.json')
    for path,digest in snapshot.items():
        if sha(path)!=digest:raise ValueError('Actual precompute input changed: '+path)
    evidence=audit_spectrum(source,output)
    metadata=read(output/'metadata.json')
    for name,digest in metadata['workflow_sha256'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Actual spectral worker changed')
    sources=dict(entry['source_sha256'])|snapshot
    for name in ['metadata.json','summary.json']:
        sources[str(output/name)]=sha(output/name)
    for index in metadata['indices']:
        sources[str(output/f'k_{index:04d}.json')]=sha(output/f'k_{index:04d}.json')
    return dict(reused=True,evidence=evidence,source_sha256=sources,
                scope='Exact density-bound full-mesh spectrum only; numerical pair and fit gates remain independent.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--solver',choices=['block'],default='block');a=ap.parse_args()
    source=a.source.resolve();output=a.out.resolve()
    registration=output.parent/'spectrum_reuse_registration.json'
    entries=read(registration) if registration.exists() else {'entries':{}}
    if str(output) in entries['entries']:
        result=reuse(source,output,entries)
        evidence=output.with_name(output.name+'_reuse_audits');evidence.mkdir(exist_ok=True)
        result['registration_sha256']=sha(registration)
        result['wrapper_sha256']=sha(__file__)
        name=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json'
        with (evidence/name).open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(dict(reused=True,**result['evidence'])),flush=True)
    else:
        if output.exists():raise FileExistsError('Unregistered existing spectrum: '+str(output))
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('guarded_sharded_spectrum.py')),
                        str(source),'--out',str(output),'--solver',a.solver],check=True)

if __name__=='__main__':main()
