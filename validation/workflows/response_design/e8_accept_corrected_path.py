"""Accept the corrected E8 path using the user's extension of Gamma gates.

Raw spectra, rejected comparisons, repaired comparison and Gamma protocol are
read-only. Only a separate acceptance certificate and per-point CSV are emitted.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def main(T):
    protocol=read(T/'protocol.json');gamma=read(T/'Gamma_acceptance.json')
    comparison=read(T/'corrected_comparison.json');gates=protocol['Gamma_gates']
    assert gamma['thresholds']==gates and protocol['Gamma_gates_registered_before_repaired_output']
    source=Path(protocol['D3FT_accepted']);bindings=read(source.parent/'accepted_bindings.json')
    dft_rows=[read(source/f'k_{i:04d}.json') for i in range(61)]
    for i,row in enumerate(dft_rows):
        if row['index']!=i or sha(source/f'k_{i:04d}.json')!=bindings[str(i)]['sha256']:
            raise ValueError('D3FT point provenance changed')
    dft=np.asarray([r['eigenvalues_Ha'] for r in dft_rows])
    with np.load(T/'corrected_QE_bands.npz') as q:
        qe=np.array(q['energies_Ha']);k=np.array(q['k_fractional'])
    if dft.shape!=(61,56) or qe.shape!=dft.shape or not np.isfinite([dft,qe]).all():
        raise ValueError('Incomplete or nonfinite path')
    if np.max(abs(k-np.asarray(protocol['k_fractional'])))>1e-8:
        raise ValueError('QE path order mismatch')
    if np.max(abs(k-np.asarray([r['k_frac'] for r in dft_rows])))>1e-8:
        raise ValueError('D3FT path order mismatch')
    HA=27.211386245988
    left=(dft-dft[:,47].max())*HA;right=(qe-qe[:,47].max())*HA;delta=left-right
    gap_d=float((dft[:,48].min()-dft[:,47].max())*HA)
    gap_q=float((qe[:,48].min()-qe[:,47].max())*HA)
    metrics=dict(path_gap_D3FT_eV=gap_d,path_gap_QE_eV=gap_q,
        absolute_path_gap_difference_eV=abs(gap_d-gap_q),
        VBM_aligned_all_bands_MAE_eV=float(abs(delta).mean()),
        VBM_aligned_all_bands_max_abs_eV=float(abs(delta).max()),
        Gamma_total_energy_difference_Ha=gamma['energy_error_Ha'],
        maximum_D3FT_Ritz_Ha=max(max(r['residuals_Ha']) for r in dft_rows))
    for key in ['path_gap_D3FT_eV','path_gap_QE_eV','VBM_aligned_all_bands_MAE_eV','VBM_aligned_all_bands_max_abs_eV']:
        if abs(metrics[key]-comparison[key])>1e-12:raise ValueError('Stored/recomputed metric mismatch '+key)
    checks=dict(Gamma_original_gates_passed=gamma['passed'],
        Gamma_total_energy_gate=abs(gamma['energy_error_Ha'])<=gates['total_energy_Ha'],
        path_gap_uses_same_Gamma_gate=metrics['absolute_path_gap_difference_eV']<=gates['gap_eV'],
        all61x56_bands_use_same_Gamma_max_gate=metrics['VBM_aligned_all_bands_max_abs_eV']<=gates['VBM_aligned_max_eV'],
        all61_D3FT_Ritz_original_gate=metrics['maximum_D3FT_Ritz_Ha']<=1e-7,
        paths_have_positive_gap=min(gap_d,gap_q)>0)
    if not all(checks.values()):raise ValueError('Unchanged Gamma-derived full-path gate failed '+str(checks))
    files=[T/'protocol.json',T/'Gamma_acceptance.json',T/'corrected_comparison.json',
        T/'corrected_QE_bands.npz',T/'input_audit.json',T/'scf.in',T/'nscf.in',
        T/'scf_data-file-schema.xml',T/'nscf_data-file-schema.xml',
        Path(protocol['original_error_comparison']),source.parent/'accepted_bindings.json']
    hashes={str(f):sha(f) for f in files}
    result=dict(time=dt.datetime.now().astimezone().isoformat(),E8_complete=True,passed=True,
        scope='MgSiN2 61-point/56-band D3FT-QE electronic-structure comparison from Gamma SCF',
        thresholds=gates,checks=checks,metrics=metrics,
        criterion_statement='门沿用 Γ 预注册值，未另设更松的路径门',
        Gamma_thresholds_preregistered=True,full_path_application_authorized_after_results=True,
        separate_looser_path_gate=False,MAE_is_reported_not_an_added_threshold=True,
        total_energy_gate_applies_to_Gamma_SCF_only=True,
        decision='User explicitly approved extending the unchanged Gamma numerical thresholds to the full path',
        limitations=dict(SCF_sampling=[1,1,1],k_convergence_performed=False,
            experimental_comparison=False,continuous_BZ_gap_certified=False,
            parent_reference_path_in_prestrained_cell=True),
        original_error_comparison_rejected_and_retained=True,doublegrid_diagnosis_retained=True,
        no_new_calculations_submitted=True,additional_GPU_hours=0,input_sha256=hashes)
    target=T/'full_path_acceptance.json'
    if target.exists():
        old=read(target)
        for key in ['thresholds','checks','metrics','input_sha256']:
            if old[key]!=result[key]:raise ValueError('Acceptance rerun differs; preserve original')
        result=old
    else:
        with target.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        with (T/'full_path_point_comparison.csv').open('x') as f:
            writer=csv.writer(f);writer.writerow(['k_index','kx','ky','kz','D3FT_direct_gap_eV',
                'QE_direct_gap_eV','VBM_aligned_MAE_eV','VBM_aligned_max_abs_eV'])
            for i in range(61):writer.writerow([i,*k[i],(dft[i,48]-dft[i,47])*HA,
                (qe[i,48]-qe[i,47])*HA,float(abs(delta[i]).mean()),float(abs(delta[i]).max())])
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Read-only audit modified a source')
    print(json.dumps(dict(passed=True,certificate=str(target),thresholds=gates,metrics=metrics),indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    main(ap.parse_args().root)
