"""Continue three independent real LM refits after their first audited trials.

Uses the existing raw auditors and fixed retry/transition policies. No held
material is evaluated here. Review thresholds and scientific failures stop a
fold without a convergence claim; completed refits require a separate review.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_v3_lm_step import audit, read, sha, verify_hashes
from run_v3_numerical_validation_parallel import check_protocol_registration

WORKFLOW = Path(__file__).resolve().parent
FOLDS = ['all', 'leave_diamond_out', 'leave_BN_out']


def put(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def dependency_ready(dependency):
    record = Path(dependency['record'])
    path = record/'status.json'
    state = read(path).get('state') if path.exists() else None
    if state in dependency['success_states']:
        return True
    try:
        os.kill(dependency['controller_pid'], 0)
    except ProcessLookupError:
        raise ValueError('Predecessor missing without its required terminal record: '+str(record))
    if state in dependency.get('failure_states', []):
        raise ValueError('Predecessor failed: '+str(record)+' '+str(state))
    return False


def replay_decision(path, fold, campaign):
    decision = read(path)
    if decision.get('audit_script_sha256') != sha(WORKFLOW/'audit_v3_lm_step.py'):
        raise ValueError('LM auditor changed')
    if decision.get('decision') not in ['accepted_real_lm_step', 'rejected_actual_loss']:
        raise ValueError('LM decision is not eligible for iteration: '+str(decision.get('decision')))
    verify_hashes(decision['source_sha256'])
    i = decision['audit_inputs']
    replay = audit(i['trial'], i['before_numerical'], i['after_numerical'], i['fd_index'])
    if any(decision.get(k) != v for k,v in replay.items()):
        raise ValueError('Actual LM decision does not replay')
    if bool(decision.get('accepted')) != (decision['decision']=='accepted_real_lm_step'):
        raise ValueError('LM acceptance flag contradicts its decision')
    trial = read(i['trial'])
    held = read(campaign/'protocol.json')['folds'][fold]
    proposal = read(trial['proposal'])
    if trial['fold'] != fold or proposal['holdouts'] != held or set(trial['training_materials']).intersection(held):
        raise ValueError('Independent whole-material split changed')
    registries = [Path(p) for p in trial['preparation_source_hashes'] if Path(p).name=='fd_gate_registry.json']
    if len(registries) != 1:
        raise ValueError('No unique registered FD center')
    index = read(registries[0].parent/'protocol.json').get('iteration_index', 0)
    return decision, trial, index+1


class Runner:
    def __init__(self, plan, record):
        self.plan, self.record = plan, record
        self.campaign = Path(plan['campaign'])
        self.pins = dict(plan['source_sha256'])

    def status(self, state, **fields):
        value = dict(state=state, time=datetime.datetime.now().isoformat(),
                     refit_complete=False, held_materials_evaluated=False, **fields)
        temporary = self.record/'status.tmp'
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(self.record/'status.json')
        print(json.dumps(value), flush=True)

    def command(self, name, arguments, log):
        verify_hashes(self.pins)
        with Path(log).open('x') as handle:
            child = subprocess.Popen([sys.executable, '-u', str(WORKFLOW/name), *map(str,arguments)],
                                     stdout=handle, stderr=subprocess.STDOUT)
            put(Path(str(log)+'.launch.json'), dict(controller_pid=child.pid,
                command=[sys.executable,'-u',str(WORKFLOW/name),*map(str,arguments)],
                time=datetime.datetime.now().isoformat(), refit_complete=False))
            code = child.wait()
        verify_hashes(self.pins)
        if code:
            raise ValueError(name+' exited with '+str(code)+'; inspect '+str(log))

    def jacobian_policy(self, index, output):
        # Fixed prospective response distances/floors are retained. Only the
        # explicit source for this actual starting-center FD evidence changes.
        policy = read(self.plan['jacobian_protocol'])
        original = str(self.campaign/'material_fd_coverage_20261001/all_fd_evidence_index.json')
        policy['input_sha256'].pop(original)
        policy['input_sha256'][str(index)] = sha(index)
        policy['center_fd_source'] = str(index)
        policy['center_fd_source_sha256'] = sha(index)
        verify_hashes(policy['input_sha256'])
        put(output, policy)
        return policy

    def execute_trial(self, trial_path, before, fd_index, attempt):
        trial = read(trial_path)
        verify_hashes(trial['preparation_source_hashes'])
        directory = trial_path.parent
        registration = attempt/'prospective_numerical_protocol'
        self.command('prepare_v3_numerical_protocol.py', [trial_path,'--output',registration],
                     attempt/'register_numerical.log')
        prospective = read(registration/'protocol.json')
        policy = self.jacobian_policy(fd_index, attempt/'prospective_jacobian_policy.json')
        # No numerical descriptor or response gate is selected from the actual
        # candidate result. Above registrations precede the first GPU command.
        self.status('executing_real_iteration', fold=trial['fold'], trial=str(trial_path))
        self.command('run_v3_trial.py', [trial_path,'--record',directory/'execution'],
                     attempt/'trial.log')
        actual_path = directory/'execution/actual_trial.json'
        if not actual_path.exists() or not read(actual_path).get('actual_evaluation_complete'):
            raise ValueError('Actual trial did not complete; no retry from a worker/FD failure')
        actual = read(actual_path)
        numeric = directory/'candidate_numerical'
        if actual['nominal_actual_loss_gate']:
            self.command('run_v3_numerical_validation_parallel.py',
                [directory,'--campaign',self.campaign,'--record',numeric], attempt/'numerical.log')
            if not (numeric/'audit.json').exists():
                raise ValueError('Numerical execution ended without its complete audit')
            check_protocol_registration(prospective, read(numeric/'registered_protocol/protocol.json'))
            verify_hashes(policy['input_sha256'])
            put(numeric/'jacobian_resolution_protocol.json', policy)
        # Real nominal-loss rejection is audited without inventing an absent
        # numerical result. The unchanged auditor rejects before numeric gating.
        out = attempt/'lm_decision'
        self.command('audit_v3_lm_step.py', [trial_path,'--before-numerical',before,
            '--after-numerical',numeric,'--fd-index',fd_index,'--out',out], attempt/'lm_audit.log')
        return out/'acceptance.json'

    def fold(self, descriptor):
        fold = descriptor['fold']
        path = Path(descriptor['first_acceptance'])
        folder = self.record/fold
        folder.mkdir(exist_ok=False)
        history = []
        while True:
            verify_hashes(self.pins)
            decision, prior, index = replay_decision(path, fold, self.campaign)
            proposal = read(prior['proposal'])
            if not history and (Path(decision['audit_inputs']['trial']).resolve()!=Path(descriptor['first_trial']).resolve()
                    or proposal!=read(self.campaign/(fold+'_proposal.json'))
                    or proposal['starting_vector']!=[0.]*7):
                raise ValueError('First decision is not this fold\'s independent zero initialization')
            history.append(dict(acceptance=str(path),decision=decision['decision'],
                accepted=decision['accepted'],actual_reduction=decision['actual_reduction'],
                loss_before=decision.get('loss_before',proposal['loss_before']),
                candidate_vector=proposal['candidate_vector']))
            # Reports remain explicitly incomplete even after accepted steps.
            (folder/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            attempt = folder/f'attempt_{index:02d}'
            attempt.mkdir(exist_ok=False)
            if decision['accepted']:
                policy = self.plan['accepted_stopping_review_policy']
                small = decision['actual_reduction']/decision['loss_before'] < policy['loss_relative_review_threshold']
                if small or index>=policy['maximum_trials']:
                    self.command('prepare_v3_refit_transition.py', [path,'--campaign',self.campaign,
                        '--fd-registry',attempt/'unexecuted_center_fd/fd_gate_registry.json',
                        '--trial-index',index,'--out',attempt/'transition'], attempt/'transition.log')
                    state = read(attempt/'transition/status.json')
                    if state.get('refit_complete') is not False:
                        raise ValueError('Stopping review was promoted to refit completion')
                    return dict(fold=fold,state=state['state'],last_acceptance=str(path),
                        refit_complete=False,held_material_evaluated=False)
                center = attempt/(fold+'_center_fd_'+str(index))
                self.status('preparing_new_center_fd', fold=fold, iteration=index)
                self.command('prepare_v3_center_fd.py', [path,'--campaign',self.campaign,'--out',center],
                             attempt/'prepare_center.log')
                center_run = attempt/'center_fd_execution'
                self.command('run_v3_center_fd.py', [center,path,'--record',center_run], attempt/'center_fd.log')
                if read(center_run/'status.json')['state']!='completed_center_fd' or not read(center_run/'coverage.json')['passed']:
                    raise ValueError('New center FD coverage failed; no next trial')
                self.command('prepare_v3_refit_transition.py', [path,'--campaign',self.campaign,
                    '--fd-registry',center/'fd_gate_registry.json','--trial-index',index,
                    '--out',attempt/'transition'], attempt/'transition.log')
                before = Path(decision['audit_inputs']['after_numerical'])
                fd_index = center/'fd_evidence_index.json'
            else:
                self.status('preparing_real_loss_retry', fold=fold, iteration=index)
                self.command('prepare_v3_rejected_trial.py', [path,'--campaign',self.campaign,
                    '--trial-index',index,'--out',attempt/'transition'], attempt/'transition.log')
                if (attempt/'transition/status.json').exists():
                    state = read(attempt/'transition/status.json')
                    if state.get('refit_complete') is not False:
                        raise ValueError('Retry review was promoted to completion')
                    return dict(fold=fold,state=state['state'],last_acceptance=str(path),
                        refit_complete=False,held_material_evaluated=False)
                before = Path(decision['audit_inputs']['before_numerical'])
                fd_index = Path(decision['audit_inputs']['fd_index'])
            transition = read(attempt/'transition/transition.json')
            if transition['fold']!=fold or transition['holdouts']!=read(self.campaign/'protocol.json')['folds'][fold]:
                raise ValueError('Prepared transition changes fold')
            path = self.execute_trial(Path(transition['trial']), before, fd_index, attempt)


def validate_plan(plan):
    verify_hashes(plan['source_sha256'])
    if [d['fold'] for d in plan['folds']]!=FOLDS:
        raise ValueError('Whole-material refit order changed')
    if plan['accepted_stopping_review_policy']!={'loss_relative_review_threshold':1e-3,'maximum_trials':20}:
        raise ValueError('Original accepted-transition review policy changed')
    if not plan['dependencies']:
        raise ValueError('No explicit resource predecessors')
    if len({d['first_acceptance'] for d in plan['folds']})!=3:
        raise ValueError('First acceptance reused across independent folds')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan',type=Path);ap.add_argument('--record',type=Path)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();plan=read(args.plan);validate_plan(plan)
    if args.check_only:
        print(json.dumps(dict(registered=True,computations_launched=0,refit_complete=False)));return
    if args.record is None:ap.error('--record required')
    args.record.mkdir(parents=True,exist_ok=False)
    put(args.record/'plan.json',plan)
    (args.record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    runner=Runner(plan,args.record)
    runner.pins[str(args.plan.resolve())]=sha(args.plan)
    try:
        while True:
            verify_hashes(runner.pins)
            pending=[d['record'] for d in plan['dependencies'] if not dependency_ready(d)]
            if not pending:break
            runner.status('waiting_for_audited_first_trials',pending=pending);time.sleep(30)
        results=[]
        for descriptor in plan['folds']:
            results.append(runner.fold(descriptor))
            put(args.record/(descriptor['fold']+'_review.json'),results[-1])
        runner.status('all_refits_require_stopping_review',results=results)
    except Exception as error:
        runner.status('refit_iteration_queue_failed',error=str(error));raise


if __name__=='__main__':main()
