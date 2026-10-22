"""Budgeted one-device native k4 reference/hydro pair, compact k blocks.

The old k4 cache is initialization only: its Miller basis differs. Every
accepted SCF and 64-point spectrum is newly evaluated in the E6 physical model.
"""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import argparse,datetime as dt,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from run_identity import reserve_name
from e1_three_parameter_queue import read,dump,alive


def main(T):
    os.close(os.open(T/'controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
    c=read(T/'protocol.json');P=Path(c['project']);R=Path(c['runtime'])
    Q=P/'validation/results/response_design_20261015_e7_ct_pilot/gpu_reservations.json'
    spent=c['prior_E6_GPU_hours'];active=[];accepted=[];GPU=None
    def now():return dt.datetime.now().astimezone().isoformat()
    def state(stage,**kw):dump(T/'status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),jobid=250285,
        requested_gpus=[] if GPU is None else [GPU],spent_E6_GPU_hours=spent,E6_budget_GPU_hours=60,completed_geometries=len(accepted),**kw))
    def note(s):
        for p in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
            with p.open('a') as f:f.write('\n- '+now()+' '+s+'\n')
    def claim():
        nonlocal GPU
        r=subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c1','nvidia-smi',
            '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=45)
        mem={int(v[0]):int(v[1]) for l in r.stdout.splitlines() if len(v:=l.split(','))==2}
        dump(T/'GPU_probe.json',dict(time=now(),memory_MiB=mem))
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
            # Keep GPU2/5 for the approved diamond remainder handover.
            for g in c.get('gpu_candidates',[0,1,3,4,6,7]):
                if mem.get(g,10**9)<=3 and str(g) not in owners:
                    GPU=g;owners[str(g)]=dict(owner='E6 single-device compact k4 stream',status_file=str(T/'status.json'),controller_pid=os.getpid())
                    dump(Q,owners);return True
        return False
    def launch(script,args,gpus,logical,seconds):
        if spent+len(gpus)*seconds/3600+.04>60:raise RuntimeError('E6 remaining budget cannot reserve requested child')
        tag=reserve_name(T,logical,output_parent=T,log_parent=T);out=T/tag;tmp='/dev/shm/'+tag
        env={**os.environ,**c['environment'],'PYTHONPATH':str(R)+':'+c['jrystal'],'TMPDIR':tmp,
             'PYTHONPYCACHEPREFIX':tmp+'/pycache','JAX_COMPILATION_CACHE_DIR':tmp+'/jax','D3FT_TRACE_STAGES':'1'}
        cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"',
            'E6-k4','env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)),'timeout',str(seconds),c['python'],'-u',
            str(R/'validation/workflows/response_design/reverse_response'/script),*args,'--output',str(out)]
        f=(T/(tag+'.log')).open('xb');proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        rec=dict(time=now(),pid=proc.pid,jobid=250285,gpus=gpus,command=cmd,output=str(out),timeout_seconds=seconds,status='running')
        dump(T/(tag+'_launch.json'),rec)
        job=dict(proc=proc,log=f,record=rec,start=time.monotonic(),tag=tag,output=out)
        active.append(job)
        return job
    def wait_jobs(jobs,stage):
        nonlocal spent,active
        active=jobs
        while any(j['proc'].poll() is None for j in jobs):
            state(stage,pids=[j['proc'].pid for j in jobs],elapsed_seconds=max(time.monotonic()-j['start'] for j in jobs));time.sleep(15)
        codes=[]
        for j in jobs:
            j['log'].close();hours=len(j['record']['gpus'])*(time.monotonic()-j['start'])/3600;spent+=hours
            j['record'].update(end=now(),returncode=j['proc'].returncode,GPU_hours=hours,status='completed' if not j['proc'].returncode else 'failed')
            dump(T/(j['tag']+'_launch.json'),j['record']);codes.append(j['proc'].returncode)
        active=[];return codes
    def scf(nu,parent):
        checkpoint=None;start_cost=spent;previous_point=None
        for stage,epochs in enumerate([1,499,1000,2500]):
            remaining=10-(spent-start_cost)
            seconds=int(remaining*3600)-60
            if seconds<300:raise RuntimeError('Budget-limited k4 SCF: original native gates not reached')
            if stage>=2 and previous_point:
                rate=previous_point['solve_seconds']/previous_point['epochs']
                fit=int((seconds-120)/max(rate,1e-9))
                if fit<100:raise RuntimeError('Measured rate leaves insufficient E6 budget for another useful checkpoint')
                epochs=min(epochs,fit)
            # A read-only first checkpoint can qualify a compatible warm state.
            args=['--reference',c['k2_source'],'--parent',str(parent),'--nu',str(nu),'--epochs',str(epochs)]
            if checkpoint:args+=['--checkpoint',str(checkpoint)]
            if stage==0:args+=['--read-only']
            j=launch('e6_k4_point.py',args+['--single-stream'],[GPU],f'nu{nu:+g}_SCF{stage}',seconds)
            codes=wait_jobs([j],'native_k4_SCF')
            if any(codes):raise RuntimeError('k4 SCF child ended; preserve code (124 is budget-bound, not a physical FAIL)')
            checkpoint=j['output'];point=read(checkpoint/'point.json');previous_point=point
            dump(T/f'nu{nu:+g}_checkpoint_index.json',dict(source=str(checkpoint),physical_passed=point['physical_passed'],GPU_hours=spent-start_cost))
            if point['physical_passed']:return checkpoint
        raise RuntimeError('k4 SCF did not reach unchanged native gate within 4000 cumulative epochs')
    def spectra(source,nu):
        # The prior k4 workflow used 48 occupied+2 buffer eigenpairs. No unused
        # extra pair is required for this value/FD check; Ritz tolerance unchanged.
        seconds=int(5*3600)-60
        if spent+seconds/3600+.05>60:raise RuntimeError('Insufficient E6 budget for full64 spectrum')
        jobs=[launch('e6_k4_spectra.py',['--source',str(source),'--indices',json.dumps(list(range(64)))],[GPU],f'nu{nu:+g}_spectra_single',seconds)]
        codes=wait_jobs(jobs,'full64_local_k4_spectra')
        rows={}
        for j in jobs:
            for f in j['output'].glob('k_*.json'):
                v=read(f)
                if not v['passed']:continue
                if v['k_index'] in rows:raise ValueError('Duplicate k result')
                rows[v['k_index']]=v
        dump(T/f'nu{nu:+g}_spectrum_progress.json',dict(passed=len(rows)==64,completed_indices=sorted(rows),exit_codes=codes))
        if len(rows)!=64 or any(codes):raise RuntimeError('Full64 spectrum incomplete within registered budget; retain passed rows, no false k4 certificate')
        values=np.asarray([rows[k]['eigenvalues_Ha'] for k in range(64)]);nk=np.asarray(read(source/'point.json')['k_fractional'])
        vb=int(np.argmax(values[:,47]));cb=int(np.argmin(values[:,48]));gamma=int(np.argmin(np.linalg.norm(nk,axis=1)))
        result=dict(nu=nu,source=str(source),gap_eV=float((values[cb,48]-values[vb,47])*27.211386245988),
            Gamma_eV=float((values[gamma,48]-values[gamma,47])*27.211386245988),vbm_k=nk[vb].tolist(),cbm_k=nk[cb].tolist(),
            full_mesh=True,mesh=[4,4,4],points=[rows[k] for k in range(64)],maximum_ritz_Ha=max(max(v['residuals_Ha']) for v in rows.values()),
            note='Hard gap value/FD; no claim of a unique k4 adjoint at a degenerate extremum')
        dump(T/f'nu{nu:+g}_complete_spectrum.json',result);return result
    try:
        state('waiting_CPU_compact_DFT_equivalence')
        equivalence=Path(c['CPU_integration_acceptance'])
        while not equivalence.exists():
            probe_pid=c['CPU_integration_pid']
            if not alive(probe_pid):raise RuntimeError('CPU compact integration ended without acceptance; GPU not claimed')
            time.sleep(10)
        if not read(equivalence)['passed']:raise RuntimeError('CPU compact full-functional equivalence failed; GPU not claimed')
        dump(T/'CPU_integration_acceptance.json',read(equivalence))
        state('waiting_one_idle_GPU_compact_stream')
        while not claim():
            state('waiting_one_idle_GPU_compact_stream');time.sleep(30)
        # Full64 read-only energy/gradient plus native report tests real peak, not an FFT-only estimate.
        smoke=launch('e6_k4_point.py',['--reference',c['k2_source'],'--parent',c['old_k4_initialization'],'--nu','0','--epochs','1','--read-only','--single-stream','--smoke'],[GPU],'full64_memory_probe',1800)
        codes=wait_jobs([smoke],'full64_single_GPU_memory_probe')
        memory=read(smoke['output']/'memory_peak.json') if (smoke['output']/'memory_peak.json').exists() else {}
        peak=memory.get('peak_used_MiB');check=dict(passed=not any(codes) and peak is not None and peak<78*1024,peak_MiB=peak,threshold_MiB=78*1024,samples=memory.get('samples'),exit_codes=codes,scope='64 compact k blocks, real optimizer gradient and native force/stress report',source=str(smoke['output']))
        dump(T/'single_GPU_memory_acceptance.json',check)
        if not check['passed']:raise RuntimeError('Single-card k4 memory/short-run gate failed; no production calculation launched')
        if spent+45+.25>60:raise RuntimeError('Cannot reserve three native/spectrum pairs inside E6 cap')
        base=None
        for nu in (0.,-.001,.001):
            source=scf(nu,smoke['output'] if base is None else base)
            if nu==0.:base=source
            result=spectra(source,nu);accepted.append(result)
            note('E6新基组k4 ν='+str(nu)+'：真实SCF与完整64点谱PASS；只做初始化投影，不复用旧谱数据。')
        values={r['nu']:r for r in accepted};fd=(values[.001]['gap_eV']-values[-.001]['gap_eV'])/.002
        k2=read(c['k2_acceptance']);ad=k2['adjoint_fundamental_A_eV'];k2gap=c['k2_gap_eV']
        checks=dict(mesh_gap_1meV=abs(values[0.]['gap_eV']-k2gap)<.001,
            deformation_original_2percent=abs(fd-ad)<=.02*abs(ad))
        result=dict(time=now(),passed=all(checks.values()),checks=checks,k4_gap_eV=values[0.]['gap_eV'],k2_gap_eV=k2gap,
            k4_dEg_dlnV_eV=fd,k2_dEg_dlnV_eV=ad,k4_minus_k2_eV=fd-ad,
            literature_eV=-8.7,literature_difference_eV=fd+8.7,literature_relative_difference=abs(fd+8.7)/8.7,
            literature_definition='PBE/GGA fundamental indirect gap, dEg/dlnV',
            literal_reproduction_certified=False,literature_perturbed_volume_ions='not specified',
            own_ions='clamped fractional coordinates, prestrained reference',
            not_experimental_validation=True,spent_E6_GPU_hours=spent,full64_each_geometry=True,points=[str(r['source']) for r in accepted])
        dump(T/'k4_acceptance.json',result);state('k4_check_finished',passed=result['passed'],result=result)
        note('E6 k4既定gap/导数收敛门 '+('PASS' if result['passed'] else 'FAIL')+'；文献仅按定义条件对比，未认证完全重现；E6累计'+str(spent)+'GPUh。')
    except Exception as exc:
        state('failed_or_budget_limited',passed=False,error=repr(exc));note('E6 k4停止：'+repr(exc)+'；预算60GPUh/原物理门保持，未宣称k4闭合。');raise
    finally:
        for j in active:
            if j['proc'].poll() is None:j['proc'].wait()
            if j['record']['status']=='running':
                j['log'].close();hours=len(j['record']['gpus'])*(time.monotonic()-j['start'])/3600;spent+=hours
                j['record'].update(end=now(),returncode=j['proc'].returncode,GPU_hours=hours,status='failed_parent_stage_preserved')
                dump(T/(j['tag']+'_launch.json'),j['record'])
        if active:
            previous=read(T/'status.json');previous['spent_E6_GPU_hours']=spent;dump(T/'status.json',previous)
        with Q.with_name('gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
            for g in range(8):
                if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):owners.pop(str(g))
            dump(Q,owners)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
