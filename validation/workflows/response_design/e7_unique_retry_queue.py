"""Unique immutable diamond r2, followed by AlAs accounting/retry, on GPU6.

The duplicate dispatch error is preserved. Existing calculations run to their
natural end. Only the not-yet-submitted E6 CPU observer is held temporarily;
its two-card stage is restored on success or failure of this chain.
"""
from pathlib import Path
import json,os,signal,sys,time
import e7_ct_pilot as pilot
E=pilot.E;A=E/'retry_unique_20261015'
pilot.priority_pending=lambda:None
if '--resume-waiting' not in sys.argv:
    os.close(os.open(A/'controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
def status(stage,**kw):pilot.dump(A/'status.json',dict(time=pilot.now(),stage=stage,jobid=250285,gpu=6,**kw))
def original_done(material):
    # The original dispatcher owns its child's return code and GPU-hour record.
    # Wait for that record, rather than infer failure from an unrelated error.
    while True:
        row=pilot.load(E/(material+'_cheap_launch.json'))
        if row.get('status')=='failed':return row
        if row.get('status')=='completed' and (E/'runs'/(material+'_cheap')/'native_acceptance.json').exists():return row
        status('waiting_original_'+material,original_pid=row.get('pid'),original_status=row.get('status'))
        time.sleep(20)
def adopt(material,tag):
    work=E/'runs'/tag
    if not pilot.load(work/'native_acceptance.json').get('passed'):
        raise RuntimeError(tag+' did not pass native gates; retained, not selected for calibration')
    choices=pilot.load(E/'selected_runs.json');choices.setdefault(material,{})['cheap']=tag
    pilot.dump(E/'selected_runs.json',choices)
    pilot.report()
    pilot.dump(A/(material+'_selected.json'),dict(time=pilot.now(),material=material,selected_tag=tag,
        GPU_hours=pilot.load(E/(tag+'_launch.json')).get('GPU_hours'),native_passed=True,
        independent_fine_gate_pending=True,old_results_removed=False))
    return work

held=None
try:
    # Four-device E5 remains untouched. Holding this CPU-only gate prevents
    # E6 from racing the explicitly requested GPU6 retry during JAX startup.
    e6=pilot.P/'validation/results/response_design_20261015_e6_k2_hydro'
    state=pilot.load(e6/'status.json');launch=pilot.load(e6/'controller_launch.json')
    if state.get('stage')=='waiting_new_E5_k2_SCF_and_fixed_cards':
        pid=launch['pid'];proc=Path(f'/proc/{pid}')
        children=[v for q in (proc/'task').glob('*/children') for v in q.read_text().split()]
        if children:raise RuntimeError('E6 has a child; no CPU hold or overlapping retry launched')
        if b'e6_k2_hydro_queue.py' not in (proc/'cmdline').read_bytes():raise RuntimeError('E6 PID identity changed')
        os.kill(pid,signal.SIGSTOP);held=pid
        pilot.dump(A/'E6_CPU_hold.json',dict(time=pilot.now(),pid=pid,children=[],GPU_processes_signalled=[],
            reason='Reserve GPU6 for user-requested unique diamond→AlAs chain',resume_on_any_exit=True))
    else:
        # A live E6 stage owns both cards through its spectral waves. Wait for
        # its terminal state, not for a momentary zero-memory reading.
        while state.get('stage') not in ('failed','k2_hydro_pilot_finished_k4_literature_pending'):
            status('waiting_existing_E6_to_finish');time.sleep(20);state=pilot.load(e6/'status.json')
    status('diamond_cheap_r2_queued',previous_dispatch_error_preserved=True)
    original_done('diamond')
    status('diamond_cheap_r2_waiting_GPU6')
    work=pilot.run('diamond','cheap',[6],10800,pilot.initial('diamond'),run_tag='diamond_cheap_r2')
    if work is None:raise RuntimeError('diamond_cheap_r2 failed; no fake Ct accepted')
    adopt('diamond','diamond_cheap_r2')
    pilot.note('E7唯一diamond_cheap_r2已完成并通过原生门，原diamond_cheap及重复调度失败均保留；下一步AlAs接续。')
    status('AlAs_followup_waiting_original')
    old=original_done('AlAs')
    if old.get('status')=='completed' and pilot.load(E/'runs/AlAs_cheap/native_acceptance.json').get('passed'):
        adopt('AlAs','AlAs_cheap')
        action='existing AlAs completed and passed; counted once, no duplicate calculation'
    else:
        status('AlAs_cheap_r2_waiting_GPU6',original_terminal=old.get('status'))
        # A four-hour timeout can truncate several cold 9000-epoch roots/iso
        # points. Allow eight hours while keeping the same numerical schedule
        # and gates; all attempts remain inside the shared 120/400 GPUh caps.
        work=pilot.run('AlAs','cheap',[6],28800,pilot.initial('AlAs'),run_tag='AlAs_cheap_r2')
        if work is None:raise RuntimeError('AlAs_cheap_r2 failed; retain failure and preregistered reduction branch')
        adopt('AlAs','AlAs_cheap_r2');action='new unique AlAs_cheap_r2 completed'
    status('unique_retry_chain_complete',AlAs_action=action)
    pilot.note('E7唯一run接续链完成：diamond_cheap_r2；AlAs '+action+'。Ct逐项登记，Si/BN既有计量保留，独立加密门仍单独验收。')
    print('E7 diamond_cheap_r2原生门PASS，重复调度失败保留。\nAlAs接续完成并逐项计时，未重复计入Ct。\nE6 CPU门控恢复；独立数值门继续另验。',flush=True)
except Exception as exc:
    status('failed',error=str(exc));pilot.note('E7唯一run接续链失败：'+str(exc)+'；失败及原始数据保留，门不放宽。');raise
finally:
    if held is not None:
        proc=Path(f'/proc/{held}')
        if proc.exists() and b'e6_k2_hydro_queue.py' in (proc/'cmdline').read_bytes():
            os.kill(held,signal.SIGCONT)
            pilot.dump(A/'E6_CPU_release.json',dict(time=pilot.now(),pid=held,GPU_processes_signalled=[]))
