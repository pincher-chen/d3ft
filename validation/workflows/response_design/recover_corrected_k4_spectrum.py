"""Resume only missing k4 spectra across the two registered idle allocations.

Preserve failed deployment, stationary source and accepted individual rows.
Only temporary/cache layout and missing-index partitioning change.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import datetime,hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

P=Path('${D3FT_ROOT}')
OLD=P/'validation/results/response_design_20261004_corrected_k4_check'
R=P/'validation/results/response_design_20261004_corrected_k4_spectrum_recovery'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(f,d):
    f=Path(f);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(f)

def execute():
    cfg=read(R/'plan.json');src=Path(cfg['source']);original=read(OLD/'plans/corrected_endpoint_k4_spectra.json')
    g=original['gates'];density_sha=sha(src/'density.npy');point=read(src/'point.json')
    assert point['residual']<=g['electronic_residual'] and point['acoustic_sum_rule_Ha_Bohr']<=g['raw_ASR_Ha_Bohr']
    assert abs(point['integrated_valence_density']-96)<=.00096
    assert sha(src/'mask.npy')==original['parent_mask_sha256']
    rows={};bindings=[]
    def collect(entry,subset):
        output=Path(entry['output']);view=Path(entry['view']);meta=read(output/'metadata.json')
        assert meta['density_sha256']==density_sha and meta['indices']==entry['indices']
        assert meta['solver']=='block' and meta['lightweight_reconstruction'] and meta['occupied_bands']==48
        assert len(meta['mesh'])==64 and sha(view/'metadata.json')==meta['source_metadata_sha256']
        bound=read(view/'metadata.json')['generic_source_binding']
        for name in ['metadata.json','point.json','density.npy','mask.npy','input.xyz']:assert bound[str(src/name)]==sha(src/name)
        phase=P/'validation/results/response_design_20261002_resource_fill/comparison_runtime/validation/workflows/phase3'
        for name,h in meta['workflow_sha256'].items():assert sha(phase/name)==h
        for index in subset:
            f=output/f'k_{index:04d}.json';row=read(f)
            assert row['index']==index and index not in rows
            assert len(row['eigenvalues_Ha'])==50 and all(math.isfinite(x) for x in row['eigenvalues_Ha']+row['residuals_Ha'])
            assert max(row['residuals_Ha'])<=g['eigen_residual_Ha']
            assert np.allclose(row['k_frac'],point['k_fractional'][index],rtol=0,atol=1e-10)
            assert row['vbm_Ha']==row['eigenvalues_Ha'][47] and row['cbm_Ha']==row['eigenvalues_Ha'][48]
            rows[index]=row;bindings.append(dict(index=index,path=str(f),sha256=sha(f),reused=entry.get('reused',False)))
    try:
        for entry in cfg['reused']:collect(entry,entry['available_indices'])
        workers=[]
        for entry in cfg['workers']:
            with (R/(entry['name']+'.log')).open('xb') as log:
                proc=subprocess.Popen(entry['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((entry,proc))
        dump(R/'worker_launch.json',[dict(name=e['name'],jobid=e['jobid'],gpu=e['gpu'],pid=p.pid) for e,p in workers])
        while any(p.poll() is None for e,p in workers):
            dump(R/'status.json',dict(stage='spectra_running',time=datetime.datetime.now().isoformat(),
                reused_points=len(bindings),new_points=sum(len(list(Path(e['output']).glob('k_*.json'))) for e,p in workers),
                workers=[dict(name=e['name'],returncode=p.poll()) for e,p in workers]))
            time.sleep(20)
        codes=[dict(name=e['name'],returncode=p.returncode) for e,p in workers];dump(R/'exit_codes.json',codes)
        if any(d['returncode'] for d in codes):raise RuntimeError('Recovery worker failed; logs and rows preserved')
        for entry,proc in workers:collect(entry,entry['indices'])
        assert sorted(rows)==list(range(64))
        v=max(rows.values(),key=lambda x:x['vbm_Ha']);c=min(rows.values(),key=lambda x:x['cbm_Ha'])
        gap=(c['cbm_Ha']-v['vbm_Ha'])*27.211386245988;assert gap>2e-7*27.211386245988
        gamma=[row for row in rows.values() if max(abs(x) for x in row['k_frac'])<1e-10];assert len(gamma)==1
        result=dict(source=str(src),density_sha256=density_sha,mask_sha256=original['parent_mask_sha256'],
            points=[rows[i] for i in range(64)],full_mesh=True,mesh=[4,4,4],gap_eV=gap,
            gamma_gap_eV=(gamma[0]['cbm_Ha']-gamma[0]['vbm_Ha'])*27.211386245988,vbm_k=v['k_frac'],cbm_k=c['k_frac'],
            max_eigen_residual_Ha=max(max(x['residuals_Ha']) for x in rows.values()),electronic_residual=point['residual'],
            raw_ASR_Ha_Bohr=point['acoustic_sum_rule_Ha_Bohr'],all_registered_gates_passed=True,
            scope='Original fixed-density k4 spectrum resumed after scratch exhaustion; original gates unchanged')
        dump(R/'row_provenance.json',bindings);dump(R/'corrected_endpoint_k4_complete.json',result)
        target=OLD/'spectra/corrected_endpoint_k4_complete.json';assert not target.exists();dump(target,result)
        completed=subprocess.run([sys.executable,str(OLD/'compare.py')],capture_output=True,text=True)
        (R/'comparison.log').write_text(completed.stdout+completed.stderr);assert completed.returncode==0
        dump(R/'status.json',dict(stage='completed',time=datetime.datetime.now().isoformat(),complete_points=64,
            original_target_passed=read(OLD/'comparison.json')['target_gate_passed']))
    except Exception as exc:
        dump(R/'status.json',dict(stage='failed',time=datetime.datetime.now().isoformat(),error=str(exc)));raise

def prepare():
    R.mkdir(exist_ok=False);shutil.copy2(Path(__file__),R/'controller.py')
    original=read(OLD/'plans/corrected_endpoint_k4_spectra.json');src=Path(original['source']);reused=[];used=set()
    for entry in original['shards']:
        available=sorted(int(f.stem.split('_')[1]) for f in Path(entry['output']).glob('k_*.json'))
        reused.append(dict(entry,available_indices=available,reused=True));used.update(available)
    missing=sorted(set(range(64))-used);assert len(used)==24 and len(missing)==40
    prefix=original['shards'][0]['command'];begin=prefix.index('env')+1;end=prefix.index('/bin/bash')
    env={x.split('=',1)[0]:x.split('=',1)[1] for x in prefix[begin:end]}
    runtime=P/'validation/results/response_design_20261002_resource_fill/comparison_runtime'
    wrapper=R/'private_tmp.sh'
    wrapper.write_text('''#!/bin/bash
set -eu
case "${TMPDIR:-}" in /dev/shm/d3ft_response_design_20261001_k4resume_*) ;; *) exit 90;; esac
[ ! -e "$TMPDIR" ] || exit 91
[ "$(df -Pk /dev/shm | awk 'NR==2 {print $4}')" -gt 104857600 ] || exit 92
mkdir -m 700 -p "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
ulimit -c 0
"$@"
''')
    workers=[];slot=0
    for job,count in [('250285',8),('247061',4)]:
        base=['srun','--jobid='+job,'--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32']
        inventory=subprocess.check_output(base+['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        assert all(int(row.split(',')[1])<=100 for row in inventory.strip().splitlines()),inventory
        allowed=json.loads(subprocess.check_output(base+['python3','-c','import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True));physical=[x for x in allowed if x<64]
        for gpu in range(count):
            indices=missing[slot::12];name=f'job{job}_gpu{gpu}';tmp=f'/dev/shm/d3ft_response_design_20261001_k4resume_{name}'
            settings=dict(env,CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=tmp,JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache',
                PYTHONPYCACHEPREFIX=tmp+'/python_cache',JAX_COMPILATION_CACHE_MAX_SIZE='0')
            mask=','.join(str(x) for c in physical[gpu::count] for x in [c,c+64] if x in allowed)
            view=R/'views'/name;output=R/'spectra'/name;view.parent.mkdir(exist_ok=True);output.parent.mkdir(exist_ok=True)
            command=base+[f'--gres=gpu:{count}','--gpu-bind=none','taskset','-c',mask,'timeout','--signal=TERM','--kill-after=30s','14400s','env']+[f'{k}={v}' for k,v in settings.items()]
            command+=['/bin/bash',str(wrapper),prefix[end+2],str(P/'validation/results/response_design_20261002_resource_fill/run_k4_spectrum_shard.py'),str(src),str(view),str(output),json.dumps(indices)]
            workers.append(dict(name=name,jobid=job,gpu=gpu,indices=indices,command=command,view=str(view),output=str(output)));slot+=1
    cfg=dict(source=str(src),original_failed_run=str(OLD),missing_indices=missing,reused=reused,workers=workers,
        frozen_at=datetime.datetime.now().isoformat(),scope='No new SCF, gates, solver, basis or density changes; only missing points')
    dump(R/'plan.json',cfg)
    with (R/'controller.log').open('xb') as log:proc=subprocess.Popen([sys.executable,'-u',str(R/'controller.py'),'--execute'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(R/'launch.json',dict(time=datetime.datetime.now().isoformat(),pid=proc.pid,worker_count=12,missing_points=40))
    print(json.dumps(dict(root=str(R),pid=proc.pid,workers=12,missing_points=40)))

if __name__=='__main__':
    execute() if '--execute' in sys.argv else prepare()
