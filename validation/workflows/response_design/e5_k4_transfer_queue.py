"""Budgeted E5 two-state transfer using ordinary four/eight-card k-pool ground states."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from gpu_probe_retry import install_gpu_probe_retry
from e1_three_parameter_queue import read, dump
from run_identity import reserve_name
install_gpu_probe_retry()


def file_sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize_transfer(start, endpoint, protocol):
    target = protocol['target_eV']
    error = endpoint['fundamental_gap_eV']-target
    gain = endpoint['fundamental_gap_eV']-start['fundamental_gap_eV']
    increment_error = gain-protocol['desired_gap_increment_eV']
    return dict(completed=True, numerical_gates_passed=True,
        mesh=[4,4,4], states=dict(start=start, endpoint=endpoint),
        fixed_original_k2_target_eV=target, target_error_eV=error,
        target_error_meV=1000*error,
        target_transfer_passed=abs(error)<=protocol['target_tolerance_eV'],
        fundamental_gap_change_eV=gain, increment_error_eV=increment_error,
        increment_error_meV=1000*increment_error,
        increment_transfer_passed=abs(increment_error)<=protocol['target_tolerance_eV'],
        start_k4_minus_k2_gap_eV=start['fundamental_gap_eV']-protocol['states']['start']['k2_gap_eV'],
        old_Gamma_only=protocol['old_Gamma_only'],
        comparison_caveat='Old -4.791 meV measures Gamma DIRECT gap at k4 density; '
            'new objective is full-mesh fundamental gap. Different target amplitude/basis: '
            'descriptive comparison only, not a controlled superiority test.',
        continuous_BZ_or_k_convergence_certified=False,
        structural_relaxation_performed=False, adjoint_computed=False)


class InfrastructureInterruption(RuntimeError):pass


def main(T):
    os.close(os.open(T/'controller.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY))
    c = read(T/'protocol.json')
    P = Path(c['project'])
    Y = P/'validation/results/response_design_20261015_e7_ct_pilot'
    Q = Y/'gpu_reservations.json'
    active = []
    records = []
    spent = c.get("initial_spent_GPU_hours",0.)
    claimed_at = None
    reserved_spent = 0.
    gpus = list(c.get("initial_repair_gpus", c.get("requested_gpus",range(8))))
    cap=float(c.get("GPU_budget",60))
    scf_seconds=c["SCF_timeout_seconds"]
    spectrum_seconds=c["spectra_timeout_seconds"]

    def now():
        return dt.datetime.now().astimezone().isoformat()

    def reserved_hours():
        return reserved_spent+(0. if claimed_at is None else len(gpus)*(time.monotonic()-claimed_at)/3600)

    def reserved_forecast(seconds):
        return reserved_hours()+len(gpus)*(seconds+25)/3600

    def state(stage, **kw):
        dump(T/'status.json', dict(time=now(), stage=stage, jobid=250285,
            requested_gpus=gpus, controller_pid=os.getpid(),
            GPU_budget=cap, active_GPU_hours_completed=spent,
            exclusive_reserved_GPU_hours=reserved_hours(), **kw))

    def note(text):
        for f in [P/'paper/response_design_EXPERIMENTS_20261014.md', P/'response_design_to_do_list.md',
                  P/'paper/response_design_E5_K4_TRANSFER_20261016.md']:
            with f.open('a') as h:
                h.write('\n- '+now()+' '+text+'\n')

    def claim():
        r = subprocess.run(['srun','--jobid=250285','--overlap','--cpu-bind=none',
            '-N1','-n1','-c1','nvidia-smi','--query-gpu=index,uuid,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=45)
        if r.returncode:
            state('GPU_probe_not_ready', returncode=r.returncode)
            return False
        devices = {int(v[0]):dict(uuid=v[1].strip(), memory_MiB=int(v[2]))
                   for line in r.stdout.splitlines() if len(v:=line.split(','))==3}
        memory = {g:v['memory_MiB'] for g,v in devices.items()}
        dump(T/'GPU_probe.json', dict(time=now(), devices=devices, memory_MiB=memory))
        if any(memory.get(g, 10**9)>3 for g in gpus):
            return False
        with (Y/'gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owners = read(Q)
            if any(str(g) in owners for g in gpus):
                return False
            for g in gpus:
                owners[str(g)] = dict(owner='E5 k4 two-state transfer, no adjoint',
                    controller_pid=os.getpid(), status_file=str(T/'status.json'))
            dump(Q, owners)
        return True

    def verify_sources():
        for path, digest in c['frozen_input_sha256'].items():
            if file_sha(path) != digest:
                raise ValueError('Frozen input changed: '+path)
        for relative, digest in c['runtime_sha256'].items():
            if file_sha(Path(c['runtime'])/relative) != digest:
                raise ValueError('Frozen runtime changed: '+relative)

    def launch(script, args, gpus, logical, seconds):
        nonlocal active
        estimate=(c.get('readout_estimate_seconds',1800) if '--read-only' in args else c.get('SCF_estimate_seconds',10800)) if script=='e5_k4_transfer_point.py' else c.get('spectra_estimate_seconds',2400)
        reservation = sum(len(j['record']['gpus'])*j['record'].get('estimated_seconds',estimate)/3600 for j in active)
        safe=None
        if script=='e6_k4_spectra.py' and logical.startswith('start_') and c.get('resume_start_source'):
            estimate=c.get('start_repair_estimate_seconds',3600)
        if script=='e5_k4_transfer_point.py' and c.get('resume_start_source') and '--read-only' not in args:
            # Reserve the measured endpoint spectrum and a checkpoint-interval
            # margin before spending any remaining budget on SCF. Never loosen
            # the cap or shorten the external six-hour safety timeout.
            remaining=min(cap-spent-reservation,(cap-.2)-c.get('prior_reserved_GPU_hours',0.)-reserved_hours())
            safe=remaining*3600/len(gpus)-c.get('endpoint_spectra_reserve_seconds',4100)-1800
            if safe<600:raise InfrastructureInterruption('Insufficient authorized GPU budget for a safe SCF checkpoint interval and endpoint spectra')
            estimate=min(estimate,safe)

        if spent+reservation+len(gpus)*estimate/3600+.2 > cap:
            raise RuntimeError(str(cap)+' GPUh cap cannot reserve this child')
        # Include every reserved pool device even for single-GPU spectra.
        if c.get("prior_reserved_GPU_hours",0.)+reserved_forecast(estimate) > cap-.2:
            raise RuntimeError('Full-node reservation cap cannot fit this phase')
        tag = reserve_name(T, logical, output_parent=T/'runs')
        out = T/'runs'/tag
        tmp = '/dev/shm/'+tag
        base = c['environment']
        runtime = Path(c['runtime'])
        env = {**os.environ, **base, 'PYTHONPATH':':'.join(map(str,[runtime,
            runtime/'validation/workflows/response_design/reverse_response',
            runtime/'validation/workflows/response_design',runtime/'validation/workflows/phase3',
            Path(c['jrystal'])])), 'TMPDIR':tmp,
            'PYTHONPYCACHEPREFIX':tmp+'/pycache','JAX_COMPILATION_CACHE_DIR':tmp+'/jax',
            'D3FT_TRACE_STAGES':'1', 'PYTHONUNBUFFERED':'1',
            'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
        cmd = ['srun','--jobid=250285','--overlap','--cpu-bind=none','--nodelist=hn35','-N1','-n1',
            '-c'+str(32 if len(gpus)>1 else 4),'--gpu-bind=none','bash','-c',
            'mkdir -p "$TMPDIR"; exec "$@"','E5-k4-transfer','env',
            'CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)),
            'timeout','--signal=TERM','--kill-after=30s',str(seconds),c['python'],
            '-u',str(runtime/'validation/workflows/response_design/reverse_response'/script),
            *args,'--output',str(out)]
        if script=='e5_k4_transfer_point.py':
            safe=safe if safe is not None else max(1.,(cap-spent-reservation)*3600/len(gpus)-3000)
            if safe<estimate:raise RuntimeError('Budget cannot safely cover predicted SCF plus checkpoint margin; no child launched')
            cmd+=['--safe-scf-wall-seconds',str(safe)]
        log = (T/(tag+'.log')).open('xb')
        start = time.monotonic()
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        rec = dict(time=now(), pid=proc.pid, jobid=250285, gpus=gpus,
            command=cmd, output=str(out), timeout_seconds=seconds, estimated_seconds=estimate, timeout_multiplier=seconds/estimate, checkpoint_interval=500, status='running')
        dump(T/(tag+'_launch.json'), rec)
        job = dict(proc=proc, log=log, start=start, record=rec, tag=tag, output=out)
        records.append(job)
        active.append(job)
        return job

    def wait_jobs(jobs, label, allow_failure=False):
        nonlocal spent, active
        while any(j['proc'].poll() is None for j in jobs):
            state(label, children=[dict(pid=j['proc'].pid, output=str(j['output'])) for j in jobs],
                  completed_k_points=sum(len(list(j['output'].glob('k_*.json'))) for j in jobs))
            time.sleep(15)
        for j in jobs:
            j['log'].close()
            hours=len(j['record']['gpus'])*(time.monotonic()-j['start'])/3600
            spent+=hours
            j['record'].update(end=now(), returncode=j['proc'].returncode,
                GPU_hours_upper_bound=hours, status='completed' if j['proc'].returncode==0 else 'failed')
            dump(T/(j['tag']+'_launch.json'),j['record'])
            active.remove(j)
        dump(T/'budget.json',dict(active_GPU_hours_upper_bound=spent,
            exclusive_reserved_GPU_hours=reserved_hours(),
            GPU_budget=cap, scope='New E5 two-state k4 transfer only'))
        if not allow_failure and any(j['proc'].returncode for j in jobs):
            codes=[j['proc'].returncode for j in jobs]
            if 124 in codes:raise InfrastructureInterruption(label+' exit124 infrastructure timeout, not physical failure')
            if any('Infrastructure budget pause' in (T/(j['tag']+'.log')).read_text() for j in jobs):raise InfrastructureInterruption(label+' checkpoint budget pause, not physical failure')
            raise RuntimeError(label+' child failed; preserve all partial records')

    def spectrum(source, label):
        jobs=[];rows={};provenance={}
        source_state_sha256=file_sha(source/'state.pkl')
        if label=='start':
            for folder in c.get('reuse_start_spectrum_roots',[]):
                for f in Path(folder).glob('k_*.json'):
                    r=read(f);k=r['k_index']
                    if not r.get('passed'):continue
                    if (r['source_state_sha256']!=source_state_sha256 or len(r['eigenvalues_Ha'])!=50
                        or max(r['residuals_Ha'])>1e-9 or r['orthogonality_max']>1e-10):
                        raise ValueError('Reused spectrum mismatched/failed original gates')
                    if k in rows:raise ValueError('Duplicate accepted reused k')
                    rows[k]=r;provenance[k]=dict(path=str(f),sha256=file_sha(f),reused=True)
        missing=sorted(set(range(64))-set(rows))
        workers=min(len(gpus),len(missing),c.get('start_repair_max_workers',len(gpus)) if label=='start' else len(gpus))
        dump(T/(label+'_spectra_reuse_manifest.json'),dict(source_state_sha256=source_state_sha256,
            accepted_reused_indices=sorted(rows),compute_only_indices=missing,provenance=provenance))
        for slot,g in enumerate(gpus[:workers]):
            args=['--source',str(source),'--indices',json.dumps(missing[slot::workers])]
            if c.get('ritz_restarts'):args+=['--ritz-restarts',str(c['ritz_restarts'])]
            jobs.append(launch('e6_k4_spectra.py',args,[g],label+'_spectra_gpu'+str(g),spectrum_seconds))
        wait_jobs(jobs,label+'_64k_spectra')
        for j in jobs:
            for f in j['output'].glob('k_*.json'):
                if '_attempt' in f.stem:continue
                r=read(f);k=r['k_index']
                if k in rows or not r['passed'] or r['source_state_sha256']!=source_state_sha256:
                    raise ValueError('Duplicate, failed, or mismatched k spectrum')
                if len(r['eigenvalues_Ha'])!=50 or max(r['residuals_Ha'])>1e-9 or r['orthogonality_max']>1e-10:
                    raise ValueError('Unchanged local Ritz/orthogonality gate failed')
                rows[k]=r;provenance[k]=dict(path=str(f),sha256=file_sha(f),reused=False)
        if sorted(rows)!=list(range(64)):
            raise ValueError('Incomplete full k4 mesh')
        dump(T/(label+'_64k_provenance.json'),dict(source_state_sha256=source_state_sha256,
            passed=True,all_50_Ritz_gate_Ha=1e-9,orthogonality_gate=1e-10,points=provenance))
        valence=max(rows,key=lambda k:rows[k]['eigenvalues_Ha'][47])
        conduction=min(rows,key=lambda k:rows[k]['eigenvalues_Ha'][48])
        HA=27.211386245988
        gap=(rows[conduction]['eigenvalues_Ha'][48]-rows[valence]['eigenvalues_Ha'][47])*HA
        if gap<=0:
            raise ValueError('Insulating occupied-band interpretation failed')
        gamma=next(r for r in rows.values() if max(abs(x) for x in r['k_fractional'])<1e-10)
        result=dict(source=str(source),source_state_sha256=source_state_sha256,
            mask_sha256=file_sha(source/'mask.npy'),density_sha256=file_sha(source/'density.npy'),
            physical_acceptance=read(source/'point.json'),fundamental_gap_eV=gap,
            gamma_direct_gap_eV=(gamma['eigenvalues_Ha'][48]-gamma['eigenvalues_Ha'][47])*HA,
            VBM_k_index=valence,CBM_k_index=conduction,
            VBM_k_fractional=rows[valence]['k_fractional'],
            CBM_k_fractional=rows[conduction]['k_fractional'],
            maximum_Ritz_Ha=max(max(r['residuals_Ha']) for r in rows.values()),
            full_mesh=True,bands=50,spectra=[rows[k] for k in range(64)])
        dump(T/(label+'_k4_spectrum.json'),result)
        return {k:v for k,v in result.items() if k!='spectra'}

    try:
        (T/'runs').mkdir(parents=True,exist_ok=True)
        verify_sources()
        state('waiting_requested_idle_GPUs_and_existing_queues')
        while not claim():
            state('waiting_requested_idle_GPUs_and_existing_queues')
            time.sleep(30)
        claimed_at=time.monotonic()
        verify_sources()
        note('E5 k4两态转移获250285/GPU'+','.join(map(str,gpus))+'空闲注册；先执行已注册显存门，60GPUh封顶，无伴随/无新E6。')
        if c.get('memory_probe_required') and len(gpus)==4:
            smoke=launch('e5_k4_transfer_point.py',['--reference',c['states']['start']['source'],
                '--parent',c['old_k4_initialization'],'--epochs','2','--devices','4','--smoke'],
                gpus,'four_card_memory_probe',c['probe_timeout_seconds'])
            wait_jobs([smoke],'four_card_memory_probe',allow_failure=True)
            memory=read(smoke['output']/'memory_peak.json') if (smoke['output']/'memory_peak.json').exists() else {}
            numerics=read(smoke['output']/'smoke_numerics.json') if (smoke['output']/'smoke_numerics.json').exists() else {}
            peak=memory.get('peak_used_MiB')
            allocator_peaks=[r['statistics']['peak_bytes_in_use']/1024**2
                for r in numerics.get('device_memory_statistics',[]) if 'peak_bytes_in_use' in r.get('statistics',{})]
            effective_peak=max([peak]+allocator_peaks) if peak is not None else None
            passed=bool(smoke['proc'].returncode==0 and numerics.get('numerical_ok')
                and numerics.get('native_force_stress_exercised')
                and len(memory.get('gpu_UUIDs',[]))==4
                and effective_peak is not None and effective_peak<78*1024)
            certificate=dict(passed=passed,returncode=smoke['proc'].returncode,
                requested_GPUs=gpus,peak_used_MiB=peak,effective_peak_MiB=effective_peak,
                threshold_GiB=78,numerics=numerics,memory=memory,
                scope='Two actual optimizer steps plus native force/stress; not SCF convergence',
                sampled_peak_is_lower_bound=True,output=str(smoke['output']))
            dump(T/'four_card_memory_acceptance.json',certificate)
            if passed:
                note('E5四卡显存短测PASS：GPU0/2/3/7峰值'+str(effective_peak/1024)+
                    'GiB<78，2真实优化步及原生导数已执行；自动接两态k4生产，数值门不变。')
            else:
                note('E5四卡短测未通过（exit='+str(smoke['proc'].returncode)+
                    '，峰值MiB='+str(effective_peak)+'），原记录保留；释放四卡后退回等待八卡，探针计入60GPUh。')
                reserved_spent=reserved_hours()
                with (Y/'gpu_reservations.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
                    for g in gpus:
                        if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):owners.pop(str(g))
                    dump(Q,owners)
                claimed_at=None;gpus=list(range(8));scf_seconds=9000;spectrum_seconds=3600
                state('four_card_probe_failed_waiting_eight_idle_GPUs',memory_acceptance=certificate)
                while not claim():
                    state('four_card_probe_failed_waiting_eight_idle_GPUs',memory_acceptance=certificate)
                    time.sleep(30)
                claimed_at=time.monotonic();verify_sources()
        summaries={}
        parent=Path(c.get('start_saved_parent',c['old_k4_initialization']))
        for label in ['start','endpoint']:
            readonly=label=='start' and bool(c.get('start_saved_parent'))
            if label=='start' and c.get('resume_start_source'):
                source=Path(c['resume_start_source'])
                if file_sha(source/'state.pkl')!=read(source/'metadata.json')['state_sha256']:
                    raise ValueError('Changed accepted start state')
            else:
                arguments=['--reference',c['states'][label]['source'],'--parent',str(parent),
                    '--epochs','1' if readonly else str(c.get('endpoint_remaining_epochs',3000)),'--devices',str(len(gpus))]
                if label=='endpoint' and c.get('endpoint_resume_checkpoint'):arguments+=['--resume-checkpoint',c['endpoint_resume_checkpoint']]
                if readonly:arguments+=['--read-only']
                job=launch('e5_k4_transfer_point.py',arguments,gpus,label+'_saved_readout' if readonly else label+'_SCF',
                    c.get('readout_timeout_seconds',7200) if readonly else scf_seconds)
                wait_jobs([job],label+'_kpool_readout' if readonly else label+'_kpool_SCF')
                source=job['output']
            if not read(source/'point.json')['physical_passed']:
                raise ValueError('Unaccepted SCF cannot enter spectra')
            if label=='start' and c.get('completed_start_spectrum'):
                cert=read(c['completed_start_spectrum'])
                if cert['source_state_sha256']!=file_sha(source/'state.pkl') or sorted(row['k_index'] for row in cert['spectra'])!=list(range(64)):
                    raise ValueError('Changed/incomplete accepted start spectrum')
                if any(not row['passed'] or len(row['eigenvalues_Ha'])!=50 or max(row['residuals_Ha'])>1e-9 or row['orthogonality_max']>1e-10 for row in cert['spectra']):
                    raise ValueError('Accepted start spectrum original gate mismatch')
                dump(T/'start_k4_spectrum.json',cert)
                summaries[label]={k:v for k,v in cert.items() if k!='spectra'}
            else:
                summaries[label]=spectrum(source,label)
            parent=source
            if label=='start' and c.get('initial_repair_gpus'):
                reserved_spent=reserved_hours()
                with (Y/'gpu_reservations.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);owners=read(Q)
                    for g in gpus:
                        if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):owners.pop(str(g))
                    dump(Q,owners)
                claimed_at=None;gpus=list(c['requested_gpus'])
                while not claim():state('start_PASS_waiting_eight_for_endpoint');time.sleep(20)
                claimed_at=time.monotonic()
            note('E5 k4 '+label+' 全64点数值门PASS，基本gap='+str(summaries[label]['fundamental_gap_eV'])+' eV；两态完成前不记转移结论。')
        result=summarize_transfer(summaries['start'],summaries['endpoint'],c)
        result.update(active_GPU_hours_upper_bound=spent,
            exclusive_reserved_GPU_hours=reserved_hours())
        dump(T/'acceptance.json',result)
        state('completed',target_transfer_passed=result['target_transfer_passed'],
              target_error_meV=result['target_error_meV'],gap_change_eV=result['fundamental_gap_change_eV'])
        note('E5 k4转移 '+('PASS' if result['target_transfer_passed'] else 'FAIL')+
             '：固定基本gap目标误差'+str(result['target_error_meV'])+' meV，基本gap变化'+
             str(result['fundamental_gap_change_eV'])+' eV；旧Γ直接gap -4.791meV只作定义注明的并列对照。')
        print('E5 k4两态64点数值验收完成，目标转移'+('PASS' if result['target_transfer_passed'] else 'FAIL')+'。\n'+
              '基本gap变化='+str(result['fundamental_gap_change_eV'])+' eV；目标误差='+str(result['target_error_meV'])+' meV。\n'+
              '旧Γ指标定义不同；全部原始数据、预算和失败记录已保留。',flush=True)
    except InfrastructureInterruption as exc:
        state('infrastructure_interrupted_recovery_pending',passed=None,classification='timeout_or_budget_not_physical_failure',error=repr(exc))
        note('E5基础设施中断：'+repr(exc)+'；检查点保留，不算物理失败、不作为失败科学结论，后续恢复。')
        raise
    except Exception as exc:
        physical=any((j['output']/'point.json').exists() and not read(j['output']/'point.json').get('physical_passed',False) for j in records)
        if physical:dump(T/'physical_failure_acceptance.json',dict(classification='physical_gate_failure',error=repr(exc),raw_points=[str(j['output']/'point.json') for j in records if (j['output']/'point.json').exists()]))
        state('failed_or_budget_limited',passed=False if physical else None,classification='physical_gate_failure' if physical else 'infrastructure_or_budget_unresolved',error=repr(exc))
        note('E5 k4两态队列停止：'+repr(exc)+'；保持原数值/预算门，未完成不称转移通过。')
        raise
    finally:
        # Never cancel a sibling worker when another fails. Every launched
        # worker already has a bounded timeout, included in the reservation.
        for j in list(active):
            j['proc'].wait()
            j['log'].close()
            hours=len(j['record']['gpus'])*(time.monotonic()-j['start'])/3600
            spent+=hours
            j['record'].update(end=now(),returncode=j['proc'].returncode,
                GPU_hours_upper_bound=hours,status='failed_or_interrupted_controller')
            dump(T/(j['tag']+'_launch.json'),j['record'])
        if claimed_at is not None:
            dump(T/'budget.json',dict(active_GPU_hours_upper_bound=spent,
                exclusive_reserved_GPU_hours=reserved_hours(),
                GPU_budget=cap))
        with (Y/'gpu_reservations.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            owners=read(Q)
            for g in gpus:
                if owners.get(str(g),{}).get('status_file')==str(T/'status.json'):
                    owners.pop(str(g))
            dump(Q,owners)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    main(ap.parse_args().root)
