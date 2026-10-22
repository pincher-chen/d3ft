"""Rebuild derived CSV/JSON from immutable response pilot records; no raw edits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def refinement_checks(campaign, sources):
    plan_path = campaign/'mgsin2_matched_grid_refinement_plan.json'
    if not plan_path.exists():
        return {}
    plan = json.loads(plan_path.read_text())
    protocol = plan['protocol']; gates = protocol['gates']; results = []
    sources[str(plan_path.resolve())] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for test in protocol['tests']:
        item = dict(test, state='pending', passed=False)
        paths = [campaign/run/'point.json' for run in test['runs']]
        if not all(path.exists() for path in paths):
            results.append(item); continue
        try:
            points = [json.loads(path.read_text()) for path in paths]
            metas = [json.loads((path.parent/'metadata.json').read_text()) for path in paths]
        except json.JSONDecodeError:
            item['state'] = 'point_output_still_flushing'; results.append(item); continue
        allowed = {'cutoff'} if test['type'] == 'cutoff' else {'xc_grid_factor'}
        physical = [{k: v for k, v in m['physical'].items() if k not in allowed} for m in metas]
        matched = physical[0] == physical[1] and metas[0]['code_sha256'] == metas[1]['code_sha256']
        a, b = points
        errors = dict(energy_difference_Ha=abs(a['energy_Ha']-b['energy_Ha']),
            max_force_difference_Ha_Bohr=float(np.abs(np.asarray(a['forces_Ha_Bohr'])-np.asarray(b['forces_Ha_Bohr'])).max()),
            max_stress_difference_GPa=float(np.abs(np.asarray(a['stress_Ha_Bohr3'])-np.asarray(b['stress_Ha_Bohr3'])).max()*29421.015697))
        point_gates = all(x['residual'] <= gates['electronic_residual_max'] and
            x['acoustic_sum_rule_Ha_Bohr'] <= gates['raw_asr_max_Ha_Bohr'] for x in points)
        differences_pass = (errors['energy_difference_Ha'] <= gates['energy_difference_max_Ha'] and
            errors['max_force_difference_Ha_Bohr'] <= gates['max_force_difference_Ha_Bohr'] and
            errors['max_stress_difference_GPa'] <= gates['max_stress_difference_GPa'])
        item.update(state='computed', matched_other_physical_inputs=matched,
            stationary_and_raw_asr_gates=point_gates, differences_pass=differences_pass,
            passed=bool(matched and point_gates and differences_pass), **errors)
        for path in paths:
            for f in [path, path.parent/'metadata.json']:
                sources[str(f.resolve())] = hashlib.sha256(f.read_bytes()).hexdigest()
        results.append(item)
    return dict(tests=results, protocol=protocol, full_numerical_acceptance=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign', type=Path)
    args = ap.parse_args()
    rows, sources = [], {}
    for path in sorted(args.campaign.glob('*/point.json')):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # Julia may still be flushing its newly created point.
        meta_path = path.parent/'metadata.json'
        meta = json.loads(meta_path.read_text())
        force = np.asarray(data.get('forces_Ha_Bohr', []), float)
        rows.append(dict(run=path.parent.name, energy_Ha=data['energy_Ha'],
            pressure_GPa=data['pressure_GPa'],
            residual=data.get('residual', data.get('residual_norm')),
            stationary=data.get('stationarity_pass', data.get('converged', False)),
            asr_Ha_Bohr=float(np.abs(force.sum(axis=0)).max()) if force.size else None,
            max_force_Ha_Bohr=float(np.abs(force).max()) if force.size else None,
            total_seconds=data.get('total_seconds'),
            xc_grid_factor=meta.get('physical', {}).get('xc_grid_factor'),
            accepted_for_relaxation=False))
        for item in [path, meta_path]:
            sources[str(item.resolve())] = hashlib.sha256(item.read_bytes()).hexdigest()
    target = args.campaign/'analysis'
    target.mkdir(exist_ok=True)
    if rows:
        with (target/'points.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    # A same cutoff label does not prove same PW space away from Gamma.
    checks = {}
    for tag, other in [('mgsin2_k1_xc2_e3000','dftk_mgsin2_20ha_grid128_k1'),
                       ('mgsin2_k1_grid128_xc1_e1500','dftk_mgsin2_20ha_grid128_k1'),
                       ('mgsin2_k1_grid128_simpson_e3000','dftk_mgsin2_20ha_grid128_k1'),
                       ('mgsin2_xc_refinement_e1500','dftk_mgsin2_20ha_grid64_k2')]:
        p, q = args.campaign/tag/'point.json', args.campaign/other/'point.json'
        if not (p.exists() and q.exists()):
            checks[tag] = dict(state='pending', dftk_run=other)
            continue
        try:
            d, b = json.loads(p.read_text()), json.loads(q.read_text())
        except json.JSONDecodeError:
            checks[tag]=dict(state='point_output_still_flushing',dftk_run=other)
            continue
        checks[tag] = dict(state='diagnostic_comparison', dftk_run=other,
            energy_difference_Ha=d['energy_Ha']-b['energy_Ha'],
            pressure_difference_GPa=d['pressure_GPa']-b['pressure_GPa'],
            max_stress_difference_GPa=float(np.abs(np.asarray(d['stress_Ha_Bohr3'])-
                                                      np.asarray(b['stress_Ha_Bohr3'])).max()*29421.015697),
            max_force_difference_Ha_Bohr=float(np.abs(np.asarray(d['forces_Ha_Bohr'])-
                                                       np.asarray(b['forces_Ha_Bohr'])).max()),
            comparable_basis='Gamma-only' if '_k1_' in tag else 'NOT identical: |G| vs |G+k|',
            full_numerical_acceptance=False)
        protocol_path=args.campaign/'input_parity_protocol_v1.json'
        if '_k1_' in tag and protocol_path.exists():
            pm=json.loads((p.parent/'metadata.json').read_text())
            qm=json.loads((q.parent/'metadata.json').read_text())
            protocol=json.loads(protocol_path.read_text())
            spec_path=Path(qm['structure_file'])
            spec=json.loads(spec_path.read_text())
            cell=np.asarray(spec['cell_angstrom'])*qm['angstrom_to_bohr']
            positions=np.asarray(spec['fractional_positions'])@cell
            physical=pm['physical']
            same=(physical['cutoff']==qm['Ecut'] and qm['kgrid']==[1,1,1]
                  and qm['k_fractional']==[[0.,0.,0.]]
                  and qm['plane_waves_per_k']==[d['plane_waves']]
                  and pm['source_sha256']==hashlib.sha256(spec_path.read_bytes()).hexdigest()
                  and np.allclose(d['cell_bohr'],cell,rtol=0,atol=1e-10)
                  and np.allclose(d['positions_bohr'],positions,rtol=0,atol=1e-10)
                  and all(v['sha256']==hashlib.sha256((Path(qm['pp_dir'])/(s+'.upf')).read_bytes()).hexdigest()
                          for s,v in physical['pp'].items()))
            c=checks[tag]
            c['matched_gamma_inputs_and_basis_size']=bool(same)
            c['passes_initial_input_parity']=bool(same and d['stationarity_pass'] and b['converged']
                and abs(c['energy_difference_Ha'])<=protocol['energy_difference_max_Ha']
                and abs(c['pressure_difference_GPa'])<=protocol['pressure_difference_max_GPa']
                and c['max_stress_difference_GPa']<=protocol['max_stress_difference_max_GPa']
                and c['max_force_difference_Ha_Bohr']<=protocol['max_force_difference_max_Ha_Bohr'])
            c['protocol_sha256']=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    (target/'input_parity.json').write_text(json.dumps(checks, indent=2)+'\n')
    refinement = refinement_checks(args.campaign, sources)
    if refinement:
        (target/'convergence_budget.json').write_text(json.dumps(refinement, indent=2)+'\n')
    (target/'provenance.json').write_text(json.dumps(dict(sources=sources,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Unrelaxed pilots; no claimed gap adjoint, full force/elasticity or fit acceptance.'),
        indent=2)+'\n')
    print(json.dumps(dict(points=len(rows), comparisons=checks), indent=2))


if __name__ == '__main__':
    main()
