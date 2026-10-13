"""Join independent reference lanes without promoting partial or failed evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

from audit_trial import audit_spectrum
from run_v3_reference_numerical import validate_reference
from run_v3_numerical_validation import verify_hashes


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    args = ap.parse_args(); directory=args.directory.resolve()
    launches=read(directory/'launches.json'); record=directory/'evidence'
    inputs={str(directory/'launches.json'):sha(directory/'launches.json')}
    for launch in launches:
        inputs[launch['plan']]=launch['plan_sha256']
    (record/'assembly_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    def status(state,**kw):
        obj=dict(state=state,time=datetime.datetime.now().isoformat(),accepted=False,**kw)
        tmp=record/'status.tmp';tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(record/'status.json')
        print(json.dumps(obj),flush=True)
    try:
        while True:
            verify_hashes(inputs); pending=[]; failed=[]
            for launch in launches:
                source=Path(launch['record']);state=read(source/'status.json') if (source/'status.json').exists() else {}
                if state.get('state')=='completed_reference_numerical_lane':continue
                if state.get('state') in ['reference_numerical_controller_failed','dependency_controller_missing']:
                    failed.append(dict(record=str(source),details=state));continue
                try:os.kill(launch['controller_pid'],0)
                except ProcessLookupError:failed.append(dict(record=str(source),details='controller missing without terminal result'));continue
                pending.append(str(source))
            status('waiting_for_reference_lanes',pending=pending,failed=failed,
                recorded_pair_results=len(list(record.glob('*_result.json'))),registered_pairs=35)
            if not pending:break
            time.sleep(30)
        if failed:
            status('reference_lanes_incomplete',failures=failed);return
        protocol_path=directory/'protocol/protocol.json';protocol=read(protocol_path)
        plans=[read(launch['plan']) for launch in launches]
        materials=[];rows=[];source_hashes=dict(inputs)
        for launch,plan in zip(launches,plans):
            verify_hashes(plan['source_hashes']);validate_reference(plan,protocol)
            materials+=plan['materials']; audit_path=Path(launch['record'])/'audit.json'
            lane=read(audit_path);rows+=lane['rows'];source_hashes[str(audit_path)]=sha(audit_path)
            if lane['protocol_sha256']!=sha(protocol_path):raise ValueError('Reference lane protocol changed')
        if len(materials)!=7 or len(set(materials))!=7 or len(rows)!=35:
            raise ValueError('Incomplete or overlapping reference lanes')
        reference=read(plans[0]['manifest']);baseline_spectra=plans[0]['baseline_spectra']
        spectral_evidence={m:audit_spectrum(Path(reference[m]),Path(baseline_spectra[m])) for m in reference}
        result=dict(evidence_kind='frozen_reference_resolution',protocol=str(protocol_path),protocol_sha256=sha(protocol_path),
            manifest=plans[0]['manifest'],baseline_spectra=baseline_spectra,baseline_spectral_evidence=spectral_evidence,
            rows=rows,all_registered_pairs_passed=all(row.get('passed') for row in rows),
            source_hashes=source_hashes,accepted=False,certified_total_error_bounds=False,
            nonlinear_refit_complete=False,script_sha256=sha(__file__),
            scope='Zero-parameter reference resolution evidence across seven materials and five axes. Each actual pair and spectrum must be recomputed by the LM auditor; not a fit acceptance or certified continuum bound.')
        with (record/'audit.json').open('x') as handle:json.dump(result,handle,indent=2)
        status('completed_reference_numerical_validation',all_registered_pairs_passed=result['all_registered_pairs_passed'])
    except Exception as exc:
        status('reference_assembly_failed',error=str(exc));raise


if __name__=='__main__':main()
