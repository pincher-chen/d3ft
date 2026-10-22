"""Independent E8 path-point processes, immutable reuse, safe legacy handover."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from e1_three_parameter_queue import read,dump
from gpu_probe_retry import install_gpu_probe_retry
from run_identity import reserve_name
install_gpu_probe_retry()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def valid_point(row, index, c):
    return (row['index']==index and len(row['eigenvalues_Ha'])==56
        and len(row['residuals_Ha'])==56
        and np.isfinite(row['eigenvalues_Ha']).all()
        and np.isfinite(row['residuals_Ha']).all()
        and max(row['residuals_Ha'])<=1e-7
        and np.max(abs(np.asarray(row['k_frac'])-c['k_fractional'][index]))<=1e-10)


def main(T):
    os.close(os.open(T/'controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
    c=read(T/'protocol.json');P=Path(c['project']);Y=P/'validation/results/response_design_20261015_e7_ct_pilot'
    Q=Y/'gpu_reservations.json';old=Path(c['legacy_root']);oldout=old/'D3FT_start'
    active={};owned=set();accepted={};spent=0.;failures=[];legacy_stopped=False
    legacy_index=c['legacy_current_index'];legacy_start=dt.datetime.fromisoformat(read(old/'GPU_launch.json')['time'])
    if legacy_start.tzinfo is None:legacy_start=legacy_start.astimezone()
    controller_start=time.monotonic()

    def now():return dt.datetime.now().astimezone().isoformat()
    def state(stage,**kw):
        dump(T/'status.json',dict(time=now(),stage=stage,jobid=250285,controller_pid=os.getpid(),
            completed_points=len(accepted),total_points=61,new_GPU_hours_upper_bound=spent,
            legacy_worker_handover_complete=legacy_stopped,owned_gpus=sorted(owned),
            active={str(g):dict(index=j['index'],pid=j['proc'].pid,output=str(j['out'])) for g,j in active.items()},**kw))
    def note(s):
        for f in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md',P/'paper/response_design_E8_PARALLEL_20261016.md']:
            with f.open('a') as h:h.write('\n- '+now()+' '+s+'\n')
    def release(gpus):
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
            for g in gpus:
                if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):owners.pop(str(g))
                owned.discard(g)
            dump(Q,owners)
    def acquire():
        probe=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1',
            'nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=45)
        if probe.returncode:return
        devices={int(v[0]):dict(uuid=v[1].strip(),memory_MiB=int(v[2]))
            for line in probe.stdout.splitlines() if len(v:=line.split(','))==3}
        dump(T/'GPU_probe.json',dict(time=now(),devices=devices))
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
            for g in c['allowed_gpus']:
                if g in owned or str(g) in owners or devices.get(g,{}).get('memory_MiB',10**9)>3:continue
                owners[str(g)]=dict(owner='E8 independent single-point path queue',controller_pid=os.getpid(),status_file=str(T/'status.json'))
                owned.add(g)
            dump(Q,owners)
    def accept_file(f,index,metadata):
        row=read(f)
        if not valid_point(row,index,c):raise ValueError('Unchanged E8 spectrum/geometry gate failed '+str(index))
        target=T/'accepted'/f'k_{index:04d}.json'
        if index not in accepted:
            shutil.copy2(f,target)
            edge=f.parent/f'edge_{index:04d}.npz'
            if edge.exists():shutil.copy2(edge,T/'accepted'/edge.name)
            accepted[index]=dict(source=str(f),sha256=sha(f),metadata=str(metadata),metadata_sha256=sha(metadata))
            dump(T/'accepted_bindings.json',{str(k):v for k,v in sorted(accepted.items())})
    def stop_legacy_after_point():
        nonlocal legacy_stopped
        if legacy_stopped:return
        expected=oldout/f'k_{legacy_index:04d}.json'
        committed=expected.exists()
        if not committed and time.monotonic()-controller_start<c['point_timeout_seconds']:return
        # The serial worker commits each JSON atomically. Preserve that point
        # before stopping only the superseded, explicitly identified E8 PID.
        if committed:accept_file(expected,legacy_index,oldout/'metadata.json')
        code='''import os,sys,signal,json
from pathlib import Path
pid=int(sys.argv[1]);p=Path('/proc')/str(pid)
if not p.exists():print(json.dumps(dict(already_exited=True)));sys.exit(0)
args=p.joinpath('cmdline').read_bytes().decode().split('\\0')
env=dict(s.split('=',1) for s in p.joinpath('environ').read_bytes().decode().split('\\0') if '=' in s)
if sys.argv[2] not in args or sys.argv[3] not in args or env.get('SLURM_JOB_ID')!='250285':raise ValueError('PID ownership mismatch')
os.kill(pid,signal.SIGTERM)
print(json.dumps(dict(pid=pid,signal='TERM',reason='user-authorized E8 per-point migration after committed point',step=env.get('SLURM_STEP_ID'))))
'''
        r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1',
            'python3','-c',code,str(c['legacy_remote_pid']),str(old/'path_worker.py'),str(oldout)],
            capture_output=True,text=True,timeout=60)
        dump(T/'legacy_handover.json',dict(time=now(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
            committed_point=legacy_index if committed else None,all_old_results_retained=True,only_superseded_E8_signalled=True))
        if r.returncode:raise RuntimeError('Legacy PID handover failed; no foreign process signalled')
        legacy_stopped=True
        if not committed:pending.append(legacy_index)
        note('E8原串行进程在第'+str(legacy_index)+'点落盘后完成单点迁移交接；旧文件/日志保留，非物理门失败，不依赖19:55旧timeout续算。')
    def launch(g,index):
        if c['legacy_GPU_hour_reservation']+spent+sum(c['point_timeout_seconds']/3600 for _ in active)+c['point_timeout_seconds']/3600>120:
            raise RuntimeError('E8 total120 GPUh cannot reserve next independent point')
        tag=reserve_name(T,'k'+str(index)+'_gpu'+str(g),output_parent=T/'runs');out=T/'runs'/tag;tmp='/dev/shm/'+tag
        base=c['environment'];env={**os.environ,**base,'PYTHONPATH':c['runtime']+':'+c['jrystal'],
            'response_E8_RUNTIME':c['runtime'],'TMPDIR':tmp,'PYTHONPYCACHEPREFIX':tmp+'/pycache',
            'JAX_COMPILATION_CACHE_DIR':tmp+'/jax','D3FT_TRACE_STAGES':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c4','--gpu-bind=none',
            'bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E8-one-point','env','CUDA_VISIBLE_DEVICES='+str(g),
            'timeout','--signal=TERM','--kill-after=10s',str(c['point_timeout_seconds']),c['python'],'-u',
            str(T/'path_worker.py'),str(T/'protocol.json'),'start',str(out),json.dumps([index])]
        log=(T/(tag+'.log')).open('xb');start=time.monotonic();proc=subprocess.Popen(cmd,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rec=dict(time=now(),pid=proc.pid,jobid=250285,gpus=[g],index=index,command=cmd,output=str(out),
            timeout_seconds=c['point_timeout_seconds'],status='running')
        dump(T/(tag+'_launch.json'),rec);active[g]=dict(proc=proc,log=log,index=index,out=out,tag=tag,record=rec,start=start)
    def collect():
        nonlocal spent
        for g,j in list(active.items()):
            if j['proc'].poll() is None:continue
            j['log'].close();hours=(time.monotonic()-j['start'])/3600;spent+=hours
            j['record'].update(end=now(),returncode=j['proc'].returncode,GPU_hours_upper_bound=hours,
                status='completed' if j['proc'].returncode==0 else 'failed')
            dump(T/(j['tag']+'_launch.json'),j['record']);active.pop(g)
            if j['proc'].returncode:
                failures.append(dict(index=j['index'],returncode=j['proc'].returncode,output=str(j['out'])))
            else:accept_file(j['out']/f'k_{j["index"]:04d}.json',j['index'],j['out']/'metadata.json')
        dump(T/'budget.json',dict(new_GPU_hours_upper_bound=spent,legacy_GPU_hour_reservation=c['legacy_GPU_hour_reservation'],GPU_budget=120))

    try:
        for name,digest in c['frozen_file_sha256'].items():
            if sha(name)!=digest:raise ValueError('Frozen E8 program/input changed '+name)
        for index in c['initial_completed_indices']:
            accept_file(oldout/f'k_{index:04d}.json',index,oldout/'metadata.json')
        pending=[i for i in range(61) if i not in accepted and i!=legacy_index]
        state('parallel_path_starting',pending_indices=pending)
        while len(accepted)<61:
            stop_legacy_after_point();collect()
            if failures:raise RuntimeError('Independent point failed; preserve all records without relaxing gates')
            if pending:
                acquire()
                for g in sorted(owned-set(active)):
                    if not pending:break
                    launch(g,pending.pop(0))
            state('independent_single_point_spectra',pending_points=len(pending),completed_indices=sorted(accepted))
            time.sleep(10)
        rows=[read(T/'accepted'/f'k_{i:04d}.json') for i in range(61)]
        dft=np.asarray([r['eigenvalues_Ha'] for r in rows]);qe=np.load(c['qe_bands'])
        if sha(c['qe_bands'])!=c['qe_bands_sha256'] or dft.shape!=qe['energies_Ha'].shape:
            raise ValueError('QE reference archive/shape mismatch')
        if np.max(abs(qe['k_fractional']-np.asarray(c['k_fractional'])))>1e-8:raise ValueError('QE path order mismatch')
        HA=27.211386245988;left=(dft-dft[:,47].max())*HA;right=(qe['energies_Ha']-qe['energies_Ha'][:,47].max())*HA;delta=left-right
        result=dict(time=now(),completed=True,points=61,bands=56,all61_D3FT_spectra_completed=True,
            QE_SCF_NSFC_completed=True,SCF_sampling=[1,1,1],not_k_convergence=True,
            path_gap_D3FT_eV=float((dft[:,48].min()-dft[:,47].max())*HA),
            path_gap_QE_eV=float((qe['energies_Ha'][:,48].min()-qe['energies_Ha'][:,47].max())*HA),
            VBM_aligned_all_bands_MAE_eV=float(abs(delta).mean()),VBM_aligned_all_bands_max_abs_eV=float(abs(delta).max()),
            VBM_aligned_frontier44_55_MAE_eV=float(abs(delta[:,44:56]).mean()),
            maximum_D3FT_ritz_Ha=max(max(r['residuals_Ha']) for r in rows),numerical_agreement_certified=False,
            agreement_threshold=None,comparison_caveat='Own VBM alignment; same accepted density/model/path as serial E8; QE |G+k| vs fixed D3FT |G| basis; no registered cross-code agreement threshold',
            reused_points=len(c['initial_completed_indices'])+1,new_GPU_hours_upper_bound=spent,
            legacy_elapsed_hours_at_completion=(dt.datetime.now().astimezone()-legacy_start).total_seconds()/3600)
        dump(T/'comparison.json',result);np.savez_compressed(T/'aligned_bands_comparison.npz',
            D3FT_eV=left,QE_eV=right,k_fractional=c['k_fractional'],difference_eV=delta)
        state('E8_path_comparison_complete',comparison=result)
        note('E8并行路径61/61完成并保存独立QE差异，不称未预注册的跨代码PASS；释放本队列卡，E5八卡优先、之后E7三折。GPU5/6外项目仍须自行释放。')
    except Exception as exc:
        state('failed',error=repr(exc),failed_points=failures)
        note('E8单点并行停止：'+repr(exc)+'；保留所有已过点/失败，不放宽门。')
        raise
    finally:
        for g,j in list(active.items()):
            j['proc'].wait()
        collect();release(list(owned))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    main(ap.parse_args().root)
