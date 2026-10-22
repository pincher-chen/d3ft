"""Detached serial memory-only queue on hn35 GPU7; never schedules timing."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def dump(p,d):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def now():return datetime.datetime.now().astimezone().isoformat()


def main(root):
    root=root.resolve()
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=json.loads((root/'protocol.json').read_text());attempts=[]
        def note(text):
            for name in ['paper/response_design_EXPERIMENTS_20261014.md','response_design_to_do_list.md']:
                with (Path(p['project'])/name).open('a') as f:f.write('\n- '+now()+' '+text+'\n')
        def status(stage,**extra):
            used=sum(x.get('GPU_hours',(time.time()-x['start_epoch'])/3600) for x in attempts)
            dump(root/'status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),
                GPU_hours=used,node='hn35',jobid=250285,physical_gpu=7,**extra))
        for N,mode in p['modes']:
            # Query only our card. Unknown probes wait; they do not block E9/GaAs.
            status('waiting_assigned_GPU7',N=N,mode=mode)
            while True:
                try:
                    q=subprocess.run(['srun','--jobid=250285','--overlap','--nodelist=hn35',
                        '--cpu-bind=none','-N1','-n1','-c1','nvidia-smi',
                        '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,timeout=45)
                    memory={int(a):int(b) for a,b in (l.split(',') for l in q.stdout.splitlines())}
                    if q.returncode==0 and memory.get(7,10**9)<=64:break
                except (ValueError,OSError,subprocess.TimeoutExpired):pass
                time.sleep(60)
            for path,h in p['execution_SHA256'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:
                    raise ValueError('Memory execution source changed: '+path)
            tag=f'N{N}_{mode}__'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
            output=root/'runs'/tag
            tmp='/dev/shm/E3_memory_'+tag
            env={**os.environ,**p['environment']}
            env.update(TMPDIR=tmp,PYTHONPYCACHEPREFIX=tmp+'/pycache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax',
                PYTHONPATH=str(Path(__file__).parent)+':'+p['runtime']+':'+p['jrystal'],
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            for key in ['D3FT_CHECKPOINT_ROOT','D3FT_RESUME_SOURCE','D3FT_RESUME_REPORT',
                        'D3FT_EVALUATION_FREEZE_CERT','D3FT_YIELD_REQUEST']:
                env.pop(key,None)
            cmd=['srun','--jobid=250285','--overlap','--nodelist=hn35','--cpu-bind=none','--gpu-bind=none',
                '-N1','-n1','-c2','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','E3memory',
                'env','CUDA_VISIBLE_DEVICES=7','timeout','--signal=TERM','--kill-after=30s','43200',p['python'],'-u',
                str(Path(__file__).parent/'e3_mode_memory.py'),'--protocol',str(root/'protocol.json'),
                '--N',str(N),'--mode',mode,'--output',str(output)]
            record=dict(time=now(),start_epoch=time.time(),N=N,mode=mode,command=cmd,output=str(output),
                        physical_gpu=7,jobid=250285,node='hn35')
            dump(root/'launches'/(tag+'.json'),record)
            note('模式显存提交：GPU7 hn35/250285，N='+str(N)+' '+mode+'，独立新进程完整一次；非独占计时，不占E9/GaAs卡。')
            with (root/'logs'/(tag+'.log')).open('xb') as log:
                proc=subprocess.Popen(cmd,env=env,cwd=p['project'],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=proc.pid;attempts.append(record)
                dump(root/'launches'/(tag+'.json'),record)
                while proc.poll() is None:
                    status('mode_running',N=N,mode=mode,output=str(output),active_pid=proc.pid)
                    time.sleep(20)
            record.update(returncode=proc.returncode,end=now(),GPU_hours=(time.time()-record['start_epoch'])/3600)
            dump(root/'launches'/(tag+'.json'),record);dump(root/'GPU_budget.json',dict(attempts=attempts))
            if proc.returncode or not (output/'result.json').exists():
                status('interrupted_partial_retained_no_automatic_retry',returncode=proc.returncode)
                note('模式显存中断：N='+str(N)+' '+mode+' rc='+str(proc.returncode)+'；保留全部采样/切向，不拼接峰值，不自动冒充完整测量。')
                return
            note('模式显存完成：N='+str(N)+' '+mode+'，GPUh='+str(record['GPU_hours'])+'；同一模式完整新进程通过原梯度/PCG门。')
        with (root/'summary.csv').open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['N','mode','process_peak_GiB','device_peak_GiB','GPU_hours','source'])
            for a in attempts:
                d=json.loads((Path(a['output'])/'result.json').read_text());m=d['memory']['peaks']['7']
                w.writerow([a['N'],a['mode'],m['process_bytes']/2**30,m['device_bytes']/2**30,a['GPU_hours'],a['output']])
        dump(root/'acceptance.json',dict(passed=True,completed_modes=8,attempts=attempts,
            sampled_peaks_are_lower_bounds=True,formal_timing_performed=False))
        status('complete',completed_modes=8)
        note('八种模式显存测量全部完成（N1/6/24/54×forward/reverse各一次），全部原始采样/allocator/梯度及PCG保留；未做独占计时。')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);r=a.parse_args().root
    try:main(r)
    except Exception as exc:
        dump(r/'controller_error.json',dict(time=now(),error=repr(exc)))
        raise
