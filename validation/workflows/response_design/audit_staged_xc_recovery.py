"""Audit the completed recovery against raw FD data and publication records."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np

P=Path('${D3FT_ROOT}')
R=P/'validation/results/response_design_20261005_numerical_verification'
G=R/'gradient_v7'
D=P/'paper_rewriting_output/final_paper'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit():
    cfg=read(G/'gradient_protocol.json');report=read(G/'gradient_result.json')
    raw=read(G/'response/fine_reverse/result.json')
    parity=read(G/'staged_xc_geometry_parity.json')
    direction=np.asarray(cfg['direction']);gradient=np.asarray(raw['results']['gap_gradient_eV'])
    gaps={n:read(G/'orbitals'/n/'summary.json')['gap_eV']
          for n in ['plus_h','minus_h','plus_half','minus_half']}
    fd=(gaps['plus_h']-gaps['minus_h'])/.002
    half=(gaps['plus_half']-gaps['minus_half'])/.001
    richardson=(4*half-fd)/3
    projection=float(gradient@direction)
    adjoint=np.load(G/'response/fine_reverse/adjoint.npy')
    memory=read(G/'memory_observation_summary.json')
    binding=read(R/'source_binding_audit.json')
    publication=read(R/'publication_status.json')
    log=(G/'logs/fine_reverse_v7.log').read_text()
    seconds=[float(x) for x in re.findall(r'staged_geometry_complete seconds=([0-9.]+)',log)]
    checks=dict(raw_fd_matches_report=abs(fd-report['fine_fd_h_eV'])<1e-12,
        raw_half_fd_matches_report=abs(half-report['fine_fd_half_eV'])<1e-12,
        raw_richardson_matches_report=abs(richardson-report['fine_fd_richardson_eV'])<1e-12,
        raw_projection_matches_report=abs(projection-report['fine_directional_gradient_eV'])<1e-12,
        original_gradient_budgets_pass=bool(report['passed']) and all(report['checks'].values()),
        adjoint_converged=raw['results']['response_record']['converged'],
        adjoint_full_basis_finite=adjoint.shape==(56087,48) and bool(np.isfinite(adjoint).all()),
        geometry_contractions_completed=len(seconds)==2,
        no_resource_exhaustion='RESOURCE_EXHAUSTED' not in log,
        original_kernel_symmetry_gate=raw['kernel_symmetry_relative_error']<=1e-8,
        parity_passed=parity['passed'],
        parity_helper_binding=sha(G/'drivers/staged_xc_geometry.py')==parity['helper_sha256'],
        source_binding_audit_passed=binding['passed'],
        new_helper_in_publication_audit=any(c['path']==str(G/'drivers/staged_xc_geometry.py') for c in binding['checks']),
        publication_completed=publication['stage']=='completed',
        both_numerical_groups_complete=publication.get('scientific_campaigns_complete',False),
        numerical_checks_passed=publication.get('all_numerical_checks_passed',False),
        original_k6_target_failure_retained=not read(R/'kmesh_v3/result.json')['original_target_pass'])
    builds=read(R/'publication_builds.json')
    checks['four_builds_pass']=len(builds)==4 and all(b['returncode']==0 for b in builds)
    record=dict(time=datetime.datetime.now().isoformat(),passed=all(checks.values()),checks=checks,
        fine_directional_gradient_eV=projection,fine_half_fd_eV=half,
        adjoint_vs_half_fd_absolute_difference_eV=abs(projection-half),
        relative_difference=abs(projection-half)/abs(half),
        production_to_refined_gradient_difference_eV=report['gradient_setting_difference_eV'],
        response_record=raw['results']['response_record'],
        geometry_contraction_seconds=seconds,
        gpu0_observed_peak_GiB=memory['gpu0_observed_peak_MiB']/1024,
        memory_scope=memory['scope'],source_binding_count=len(binding['checks']),
        hashes={str(f):sha(f) for f in [G/'gradient_result.json',G/'response/fine_reverse/result.json',
            G/'drivers/staged_xc_geometry.py',G/'drivers/staged_scalar_gradient.py',G/'drivers/gamma_response.py',
            D/'main.tex',D/'main.pdf',D/'supplementary.tex',D/'supplementary.pdf']},
        scope='XC720 geometry memory recovery and one-direction refinement acceptance; not continuum convergence, broad acceleration or multi-k target success',submission_ready=False)
    for f in [G/'recovery_acceptance.json',D/'response_design_XC720_RECOVERY_CHECK_20261006.json']:
        f.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if not record['passed']:raise RuntimeError('Recovery/publication acceptance incomplete or failed')

if __name__=='__main__':audit()
