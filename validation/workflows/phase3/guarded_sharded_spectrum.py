"""Validate lightweight k shards at this density, then solve the complete mesh.

Four serial/sharded points are compared before enabling independent GPU shards.
A failed probe or parallel solve falls back to the original serial block solver.
All diagnostics remain separate; only a fully audited mesh is published at out.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from audit_trial import audit_spectrum


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_indices(mesh_size):
    return sorted(set([0, 1, mesh_size//2, mesh_size-1])) if mesh_size > 1 else [0]


def publish(raw, output):
    """Keep raw shard paths intact and link only the complete merged records."""
    output.mkdir(parents=True, exist_ok=False)
    for name in ['metadata.json', 'summary.json']:
        os.link(raw/name, output/name)
    for index in read(raw/'metadata.json')['indices']:
        os.link(raw/f'k_{index:04d}.json', output/f'k_{index:04d}.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--solver', choices=['block'], default='block')
    args = ap.parse_args(); source=args.source.resolve(); output=args.out.resolve()
    record=output.with_name(output.name+'_execution'); record.mkdir(parents=True,exist_ok=False)
    if output.exists():raise FileExistsError(output)
    work=Path(__file__).resolve().parent
    checker=work/'check_occupations.py'
    merger=work/'sharded_spectrum.py'
    validator=work/'validate_sharded_spectrum.py'
    snapshot={str(p):sha(p) for p in [Path(__file__),checker,merger,validator,
        work/'variational_ks.py',work/'lowest_eigensystem.py',work/'audit_trial.py']}
    snapshot.update({str(source/name):sha(source/name) for name in ['metadata.json','summary.json','points.json','mask.npy']})
    point=[p for p in read(source/'points.json') if p['label'].startswith('eq_')][-1]
    snapshot[str(source/point['density_file'])]=sha(source/point['density_file'])
    (record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (record/'inputs.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    state=dict(source=str(source),out=str(output),accepted=False,solver='block',
        scope='Complete fixed SCF mesh at this density only; pilot equivalence is not a continuum or derivative certificate.')

    def status(label,**fields):
        state.update(state=label,time=datetime.datetime.now().isoformat(),**fields)
        tmp=record/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(record/'status.json')
        print(json.dumps(state),flush=True)

    def verify():
        for p,digest in snapshot.items():
            if sha(p)!=digest:raise ValueError('Spectrum source changed: '+p)

    def run(name,command):
        verify();status(name,command=command)
        with (record/(name+'.log')).open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        verify();return result.returncode

    def serial(target,indices=None):
        command=[sys.executable,'-u',str(checker),str(source),'--out',str(target),'--solver','block']
        if indices is not None:command+=['--indices',json.dumps(indices)]
        return command

    try:
        devices=os.environ.get('CUDA_VISIBLE_DEVICES','')
        values=devices.split(',') if devices else []
        grid=read(source/'metadata.json')['config']['k_grid_sizes']
        size=grid**3 if isinstance(grid,int) else math.prod(grid)
        parallel_ok=False;fallback_reason='single visible device or mesh point'
        if len(values)>1 and size>1:
            if len(set(values))!=len(values) or any(not d for d in values):
                raise ValueError('Invalid explicit visible devices')
            serial_probe=record/'serial_probe';parallel_probe=record/'parallel_probe'
            code=run('serial_probe',serial(serial_probe,probe_indices(size)))
            if code==0:
                command=[sys.executable,'-u',str(validator),str(source),str(serial_probe),
                    '--out',str(parallel_probe),'--devices',devices]
                code=run('parallel_probe',command)
                proof=parallel_probe/'comparison.json'
                parallel_ok=code==0 and proof.exists() and read(proof).get('passed') is True
                fallback_reason='parallel probe did not pass'
            else:fallback_reason='serial probe process failed'
        if parallel_ok:
            raw=record/'parallel_full'
            code=run('parallel_full',[sys.executable,'-u',str(merger),str(source),
                '--out',str(raw),'--devices',devices])
            if code==0:
                try:evidence=audit_spectrum(source,raw)
                except (ValueError,KeyError,OSError,AssertionError) as exc:
                    fallback_reason='parallel full audit: '+str(exc)
                else:
                    verify();publish(raw,output)
                    status('completed_guarded_parallel_spectrum',evidence=evidence,
                           probe_comparison=str(record/'parallel_probe/comparison.json'))
                    return
            else:fallback_reason='parallel full process failed'
        status('falling_back_to_serial',reason=fallback_reason)
        code=run('serial_full',serial(output))
        if code:raise RuntimeError('Serial full-mesh process failed: '+str(code))
        evidence=audit_spectrum(source,output)
        status('completed_serial_spectrum',evidence=evidence,fallback_reason=fallback_reason)
    except Exception as exc:
        status('spectrum_failed',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
