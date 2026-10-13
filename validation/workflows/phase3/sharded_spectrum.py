"""Independent single-GPU k shards, with an exact-coverage spectrum merge."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def merge(source, output, shards, indices):
    source_meta=read(source/'metadata.json')
    density=[r for r in read(source/'points.json') if r['label'].startswith('eq_')][-1]
    expected_density=sha(source/density['density_file'])
    if expected_density!=density['density_sha256']:raise ValueError('Source density hash changed')
    common=None; records={}; origins={}
    for shard in shards:
        meta=read(shard/'metadata.json')
        if Path(meta['source']).resolve()!=source.resolve() or meta['source_metadata_sha256']!=sha(source/'metadata.json'):
            raise ValueError('Shard source mismatch')
        if meta['density_sha256']!=expected_density or meta['solver']!='block':raise ValueError('Shard density/solver mismatch')
        if abs(meta['density_electrons']-meta['electrons'])>meta['charge_budget']:raise ValueError('Shard charge failure')
        if common is None:common=meta
        for key in ('mesh','occupied_bands','electrons','workflow_sha256','lightweight_reconstruction'):
            if meta[key]!=common[key]:raise ValueError('Inconsistent shard '+key)
        for i in meta['indices']:
            if i in records:raise ValueError('Duplicate k index')
            path=shard/f'k_{i:04d}.json';row=read(path)
            vals,errors=row['eigenvalues_Ha'],row['residuals_Ha'];n=meta['occupied_bands']
            if row['index']!=i or row['k_frac']!=meta['mesh'][i]:raise ValueError('k-point identity mismatch')
            if len(vals)!=n+2 or len(errors)!=len(vals):raise ValueError('Incomplete bands')
            if not all(math.isfinite(x) for x in vals+errors) or min(errors)<0 or max(errors)>1e-7:
                raise ValueError('Unconverged shard')
            if vals!=sorted(vals) or row['vbm_Ha']!=vals[n-1] or row['cbm_Ha']!=vals[n]:raise ValueError('Invalid band ordering')
            records[i]=row;origins[i]=path
    if set(records)!=set(indices) or len(indices)!=len(set(indices)):raise ValueError('Missing/extra k indices')
    grid=source_meta['config']['k_grid_sizes'];size=grid**3 if isinstance(grid,int) else math.prod(grid)
    if len(common['mesh'])!=size:raise ValueError('Mesh size changed')
    if any(i<0 or i>=size for i in indices):raise ValueError('Invalid k indices')
    full=set(indices)==set(range(size))
    vbm=max(x['vbm_Ha'] for x in records.values());cbm=min(x['cbm_Ha'] for x in records.values())
    summary=dict(vbm_Ha=vbm,cbm_Ha=cbm,gap_eV=(cbm-vbm)*27.211386245988,
                 full_mesh=full,solver='block',mesh_fixed_occupations_passed=bool(full and cbm-vbm>2e-7),
                 sampled_gap_positive=bool(cbm-vbm>2e-7),
                 max_eigen_residual_Ha=max(max(x['residuals_Ha']) for x in records.values()),
                 scope='SCF mesh only; independent GPU shards of the same saved density')
    meta=dict(common,indices=sorted(indices),shards=[str(p.resolve()) for p in shards],
              merge_script_sha256=sha(Path(__file__)),
              shard_metadata_sha256={str(p.resolve()):sha(p/'metadata.json') for p in shards})
    for i,path in origins.items():os.link(path,output/f'k_{i:04d}.json')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--devices',required=True);ap.add_argument('--indices')
    args=ap.parse_args();devices=args.devices.split(',')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(set(devices))!=len(devices) or any(d not in visible or not d for d in devices):
        raise ValueError('Devices must be unique members of the allocated visible devices')
    grid=read(args.source/'metadata.json')['config']['k_grid_sizes']
    size=grid**3 if isinstance(grid,int) else math.prod(grid)
    indices=json.loads(args.indices) if args.indices else list(range(size))
    if not indices or len(set(indices))!=len(indices) or any(not isinstance(i,int) or not 0<=i<size for i in indices):
        raise ValueError('Invalid requested indices')
    args.out.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[];shards=[];start=time.time()
    record=dict(started=datetime.datetime.now().isoformat(),devices=devices,indices=indices,workers=[])
    def save():
        temp=args.out/'sharding.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(args.out/'sharding.json')
    try:
        for slot,device in enumerate(devices):
            subset=indices[slot::len(devices)]
            if not subset:continue
            shard=args.out/f'shard_{slot}';shards.append(shard)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            env['JAX_COMPILATION_CACHE_DIR']=env.get('JAX_COMPILATION_CACHE_DIR','/tmp/d3ft_spectrum')+f'_shard{slot}'
            command=[sys.executable,'-u',str(Path(__file__).with_name('check_occupations.py')),str(args.source),
                     '--out',str(shard),'--solver','block','--indices',json.dumps(subset),'--lightweight-reconstruction']
            log=(args.out/f'shard_{slot}.log').open('x');logs.append(log)
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append(proc)
            record['workers'].append(dict(slot=slot,device=device,pid=proc.pid,command=command));save()
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError('Shard failed; see per-shard logs')
            time.sleep(2)
        if any(p.returncode!=0 for p in jobs):raise RuntimeError('Shard failed')
        summary=merge(args.source,args.out,shards,indices)
        record.update(state='complete',seconds=time.time()-start,summary=summary);save()
        print(json.dumps(summary),flush=True)
    except BaseException:
        for p in jobs:
            if p.poll() is None:p.terminate()
        for p in jobs:p.wait()
        record.update(state='failed',seconds=time.time()-start);save()
        raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':main()
