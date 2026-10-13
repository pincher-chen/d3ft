"""Compare equilibrium parameter derivatives across a registered numerical pair.

Derivative errors are expressed as responses over the frozen FD comparison
distances. This is a numerical-resolution test of AD Jacobians, not an entrywise
FD test, a statistical bound, or proof that weak responses are resolved.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

from check_elastic_convergence import compare
from equilibrium_jacobian import derive
from run_v3_numerical_validation import check_registered_settings


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_gradients(source):
    source=Path(source); summary=read(source/'summary.json'); jacobian=derive(source)
    gradients={'a0_A':np.asarray(jacobian['dlogV_equilibrium_dp'])*summary['a0_A']/3,
               'B_GPa':np.asarray(jacobian['dB_equilibrium_dp'])}
    for observable,key in [('C11_minus_C12_GPa','dD_equilibrium_dp'),('C44_GPa','dC44_equilibrium_dp')]:
        if observable in summary:gradients[observable]=np.asarray(jacobian[key])
    if any(g.shape!=(7,) or not np.isfinite(g).all() for g in gradients.values()):
        raise ValueError('Incomplete or nonfinite seven-parameter equilibrium Jacobian')
    return gradients


def component(coarse,fine,span,rtol,floor):
    if not np.isfinite([coarse,fine,span,rtol,floor]).all() or span<=0 or rtol<0 or floor<=0:
        raise ValueError('Invalid derivative-resolution comparison')
    error=abs(fine-coarse)*span; budget=rtol*abs(fine*span)+floor
    return dict(coarse_derivative=float(coarse),fine_derivative=float(fine),
        physical_comparison_distance=float(span),coarse_projected_response=float(coarse*span),
        fine_projected_response=float(fine*span),projected_absolute_error=float(error),
        projected_agreement_budget=float(budget),passed=bool(error<=budget))


def audit_pair(coarse,fine,pair,protocol):
    coarse,fine=Path(coarse),Path(fine)
    metadata=[read(p/'metadata.json') for p in [coarse,fine]]
    if any(meta['parameter_vector']!=pair['parameter_vector'] for meta in metadata):
        raise ValueError('Derivative pair metadata uses another parameter vector')
    reference=Path(read(protocol['frozen_manifest'])[pair['material']])
    expected=read(reference/'metadata.json')
    proof=read(protocol['core_equivalence'])
    if (sha(protocol['core_equivalence'])!=protocol['core_equivalence_sha256'] or
        not proof['passed'] or not all(case['exact_elementwise_equal'] for case in proof['cases'])):
        raise ValueError('Unverified preconditioner implementation equivalence')
    for path,digest in proof['input_files'].items():
        if sha(path)!=digest:raise ValueError('Equivalence implementation source changed')
    for meta in metadata:
        differences={p for p in set(meta['code_sha256'])|set(expected['code_sha256'])
            if meta['code_sha256'].get(p)!=expected['code_sha256'].get(p)}
        if differences and (differences!={proof['path']} or
            {meta['code_sha256'][proof['path']],expected['code_sha256'][proof['path']]}!=
            {proof['old_sha256'],proof['new_sha256']}):
            raise ValueError('Derivative-resolution core differs from frozen v3')
        if meta['workflow_sha256']['worker.py']!=expected['workflow_sha256']['worker.py']:
            raise ValueError('Derivative-resolution worker changed')
        if meta['pseudopotentials']!=expected['pseudopotentials']:
            raise ValueError('Derivative-resolution pseudopotentials changed')
    for path,digest in expected['pseudopotentials'].items():
        if sha(path)!=digest:raise ValueError('Pseudopotential source changed')
    check_registered_settings(read(coarse/'metadata.json'),pair['coarse_settings'])
    check_registered_settings(read(fine/'metadata.json'),pair['fine_settings'])
    forward=compare(coarse,fine,pair['axis'],pair['gate']['lattice_relative'],
        pair['gate']['elastic_absolute_GPa'],
        pair['axis_definition']=='coarse FFT grid with physical XC grid held constant')
    if read(coarse/'metadata.json')['parameter_vector']!=pair['parameter_vector']:
        raise ValueError('Derivative pair is for another parameter vector')
    if forward['parameter_vector']!=pair['parameter_vector']:
        raise ValueError('Derivative pair summary uses another parameter vector')
    if forward['material']!=pair['material']:
        raise ValueError('Derivative pair material differs from numerical registration')
    a,b=physical_gradients(coarse),physical_gradients(fine)
    targets=pair['required_training_observables']
    if len(targets)!=len(set(targets)) or not targets or any(obs not in a or obs not in b for obs in targets):
        raise ValueError('Incomplete or duplicated original training observations')
    spans=protocol['physical_comparison_distances']; names=protocol['parameter_names']
    if len(spans)!=7 or len(names)!=7 or len(set(names))!=7:
        raise ValueError('Incomplete seven-parameter resolution registration')
    rows=[]
    a0=read(fine/'summary.json')['a0_A']
    for observable in targets:
        floor=protocol['projected_absolute_floors'][observable]
        signal_budget=1e-4*a0 if observable=='a0_A' else .1
        for j,name in enumerate(names):
            result=component(a[observable][j],b[observable][j],spans[j],
                protocol['relative_agreement_tolerance'],floor)
            resolved=abs(result['fine_projected_response'])>signal_budget
            rows.append(dict(material=pair['material'],axis=pair['axis'],observable=observable,
                parameter=name,parameter_index=j,**result,
                forward_signal_budget=float(signal_budget),signal_resolved=bool(resolved),
                relative_accuracy_at_resolved_signal=bool(resolved and result['passed']),
                assessment='resolved_response' if resolved else 'absolute_agreement_at_weak_or_zero_response'))
    sources={str(Path(p)/name):digest for p,files in forward['sources'].items() for name,digest in files.items()}
    sources.update(proof['input_files'])
    sources[str(Path(protocol['core_equivalence']))]=sha(protocol['core_equivalence'])
    sources[str(reference/'metadata.json')]=sha(reference/'metadata.json')
    for name in ['check_jacobian_resolution.py','check_elastic_convergence.py','equilibrium_jacobian.py','force_acceptance.py']:
        path=Path(__file__).with_name(name);sources[str(path.resolve())]=sha(path)
    return dict(material=pair['material'],axis=pair['axis'],forward_pair_passed=forward['passed'],
        passed=bool(forward['passed'] and all(row['passed'] for row in rows)),rows=rows,
        source_sha256=sources,entrywise_finite_difference_validated=False,
        weak_responses_claimed_resolved=False,certified_continuum_error_bounds=False,
        scope='AD equilibrium Jacobian resolution at this pair/vector under frozen projected-response tolerances; weak/zero entries have absolute agreement only. Not entrywise FD validation or an LM acceptance.')
