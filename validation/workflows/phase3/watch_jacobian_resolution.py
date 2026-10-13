"""Re-audit completed numerical campaigns, exporting all parameter entries.

Waits for the complete recorded forward campaign, then reconstructs Jacobians
from raw points. This CPU observer does not launch, restart, or accept DFT jobs.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_v3_lm_step import numerical_envelope
from check_jacobian_resolution import audit_pair
from fit_step import dataset
from run_v3_numerical_validation import sha, verify_hashes


def read(path):return json.loads(Path(path).read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign',type=Path)
    ap.add_argument('--record',type=Path,required=True)
    args=ap.parse_args();campaign=args.campaign.resolve()
    args.record.mkdir(exist_ok=False,parents=True)
    (args.record/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    watcher_hash=sha(__file__);finished={}

    def status(state,**fields):
        d=dict(state=state,time=datetime.datetime.now().isoformat(),accepted=False,**fields)
        tmp=args.record/'status.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(args.record/'status.json')
        print(json.dumps(d),flush=True)

    while True:
        if sha(__file__)!=watcher_hash:raise ValueError('Jacobian observer source changed')
        candidate=read(campaign/'candidate_numerical_execution_launch.json')
        assembly=read(campaign/'reference_numerical_20261001/assembly_launch.json')
        targets={'reference':(Path(assembly['record']),assembly['controller_pid']),
                 'candidate':(Path(candidate['record']),candidate['controller_pid'])}
        pending=[]
        for name,(record,pid) in targets.items():
            if name in finished:continue
            path=record/'audit.json'
            if not path.exists():
                try:os.kill(pid,0)
                except ProcessLookupError:
                    state=read(record/'status.json') if (record/'status.json').exists() else {}
                    finished[name]=dict(state='numerical_campaign_terminal_without_complete_audit',details=state)
                    continue
                pending.append(name);continue
            completed=read(path)
            if completed.get('evidence_kind')=='frozen_reference_resolution':
                manifest=completed['manifest'];numeric_path=Path(completed['protocol'])
            else:
                trial=read(Path(completed['trial'])/'trial.json')
                manifest=trial['candidate'];numeric_path=Path(completed['trial'])/'numerical_protocol_executable/protocol.json'
            numeric=read(numeric_path);proto_path=record/'jacobian_resolution_protocol.json'
            try:
                proto=read(proto_path);verify_hashes(proto['input_sha256'])
                verify_hashes(numeric['input_sha256'])
                results=[];sources={str(path):sha(path),str(proto_path):sha(proto_path),str(numeric_path):sha(numeric_path)}
                for pair in numeric['comparisons']:
                    key=pair['material']+'_'+pair['axis'];result_path=record/(key+'_result.json')
                    if not result_path.exists():
                        results.append(dict(material=pair['material'],axis=pair['axis'],passed=False,state='missing_pair_result'));continue
                    result=read(result_path)
                    if not result.get('fine'):
                        results.append(dict(material=pair['material'],axis=pair['axis'],passed=False,state='missing_fine_endpoint'));continue
                    audited=audit_pair(pair['coarse_candidate'],result['fine'],pair,proto)
                    results.append(audited);sources.update(audited['source_sha256'])
                # This independently binds spectra/core/input identities as well
                # as the forward and Jacobian gates, rather than trusting flags.
                gate_error=None
                try:
                    _,rows,_,_=dataset(manifest)
                    _,bound_sources=numerical_envelope(record,manifest,rows)
                    sources.update(bound_sources)
                except (ValueError,KeyError,OSError,AssertionError) as exc:gate_error=str(exc)
                report=dict(passed=gate_error is None and all(r.get('passed') for r in results),
                    gate_error=gate_error,numerical_record=str(record),results=results,
                    registered_pairs=len(numeric['comparisons']),
                    recorded_derivative_entries=sum(len(r.get('rows',[])) for r in results),
                    weak_or_zero_entries=sum(not row['signal_resolved'] for r in results for row in r.get('rows',[])),
                    source_sha256=sources,accepted=False,entrywise_finite_difference_validated=False,
                    weak_responses_claimed_resolved=False,certified_continuum_error_bounds=False,
                    scope='Complete empirical projected-response Jacobian resolution, with full forward/spectrum provenance re-audit; not an LM acceptance or entrywise FD test.')
                with (args.record/(name+'_resolution.json')).open('x') as handle:json.dump(report,handle,indent=2)
                finished[name]=dict(state='completed_derivative_resolution_audit',passed=report['passed'],entries=report['recorded_derivative_entries'])
            except (ValueError,KeyError,OSError,AssertionError) as exc:
                finished[name]=dict(state='invalid_derivative_resolution_evidence',error=str(exc))
        status('waiting_for_completed_numerical_campaigns' if pending else 'completed_jacobian_observation',pending=pending,results=finished)
        if not pending:
            # Audit the first real transition only when both complete numerical
            # records exist. Invalid evidence remains a rejection/pending result.
            before=targets['reference'][0];after=targets['candidate'][0]
            if (before/'audit.json').exists() and (after/'audit.json').exists():
                completed=read(after/'audit.json')
                trial=Path(completed['trial'])/'trial.json'
                command=[sys.executable,'-u',str(Path(__file__).with_name('audit_v3_lm_step.py')),
                    str(trial),'--before-numerical',str(before),'--after-numerical',str(after),
                    '--fd-index',str(campaign/'material_fd_coverage_20261001/all_fd_evidence_index.json'),
                    '--out',str(args.record/'first_lm_step')]
                with (args.record/'first_lm_step.log').open('x') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                status('completed_jacobian_observation_and_lm_audit',results=finished,
                    lm_audit_returncode=result.returncode,refit_complete=False)
            return
        time.sleep(30)


if __name__=='__main__':main()
