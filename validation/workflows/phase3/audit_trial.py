"""Audit a shared candidate without accepting it or overwriting calculation data.

Missing results remain pending. Full numerical convergence at new parameters and
derivative coverage are separate requirements, even if every available gate passes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from equilibrium_jacobian import derive
from fit_step import validate_trial


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_spectrum(source, spectrum):
    """Recompute the mesh gate from all per-k records and bind it to the density."""
    meta = read(spectrum / 'metadata.json')
    summary = read(spectrum / 'summary.json')
    eq = [p for p in read(source / 'points.json') if p['label'].startswith('eq_')][-1]
    require(Path(meta['source']).resolve() == source.resolve(), 'Spectrum source mismatch')
    require(meta['source_metadata_sha256'] == sha(source / 'metadata.json'), 'Spectrum metadata hash mismatch')
    require(meta['density_sha256'] == eq['density_sha256'] == sha(source / eq['density_file']), 'Spectrum density mismatch')
    config = read(source / 'metadata.json')['config']
    mesh = config['k_grid_sizes']
    expected = mesh ** 3 if isinstance(mesh, int) else math.prod(mesh)
    require(len(meta['mesh']) == expected, 'Spectrum mesh size differs from source')
    require(sorted(meta['indices']) == list(range(expected)), 'Incomplete spectrum indices')
    require(meta['solver'] == summary['solver'] == 'block', 'Unsupported spectrum solver')
    require(abs(meta['density_electrons'] - meta['electrons']) <= meta['charge_budget'], 'Density charge gate failed')
    vbm, cbm, residual = -math.inf, math.inf, 0.
    nocc = meta['occupied_bands']
    for i in range(expected):
        row = read(spectrum / f'k_{i:04d}.json')
        require(row['index'] == i and row['k_frac'] == meta['mesh'][i], 'Per-k identity mismatch')
        values, errors = row['eigenvalues_Ha'], row['residuals_Ha']
        require(len(values) == len(errors) and len(values) >= nocc + 1, 'Missing eigenpairs')
        require(all(math.isfinite(v) for v in values + errors), 'Nonfinite eigenpair')
        require(values == sorted(values) and min(errors) >= 0, 'Invalid eigenpair ordering/residual')
        require(row['vbm_Ha'] == values[nocc-1] and row['cbm_Ha'] == values[nocc], 'Band edge mismatch')
        vbm, cbm = max(vbm, values[nocc-1]), min(cbm, values[nocc])
        residual = max(residual, max(errors))
    require(summary['full_mesh'] and summary['mesh_fixed_occupations_passed'], 'Stored mesh gate failed')
    require(cbm - vbm > 2e-7 and residual <= 1e-7, 'Recomputed gap/residual gate failed')
    require(abs(summary['gap_eV'] - (cbm-vbm)*27.211386245988) < 1e-9, 'Stored gap mismatch')
    require(abs(summary['max_eigen_residual_Ha'] - residual) < 1e-14, 'Stored residual mismatch')
    return {'gap_eV': (cbm-vbm)*27.211386245988, 'max_eigen_residual_Ha': residual,
            'spectrum_summary_sha256': sha(spectrum/'summary.json')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ['baseline', 'proposal', 'candidate', 'spectra', 'out']:
        ap.add_argument('--'+arg, type=Path, required=True)
    args = ap.parse_args()
    require(not args.out.exists(), 'Use a new output directory for each audit snapshot')
    base, candidate, spectra, prop = map(read, [args.baseline, args.candidate, args.spectra, args.proposal])
    expected = set(prop['input_fingerprints'])
    require(set(candidate) == set(spectra) == expected, 'Candidate/spectrum maps must cover all training materials')
    for m in expected:
        for file, digest in prop['input_fingerprints'][m].items():
            require(sha(Path(base[m])/file) == digest, f'Frozen input changed: {m}/{file}')
    rows = []
    for m in sorted(expected):
        source, spectral = Path(candidate[m]), Path(spectra[m])
        row = {'material': m, 'calculation': 'pending', 'spectrum': 'pending'}
        if (source/'summary.json').exists():
            try:
                d = read(source/'summary.json')
                require(d['material'] == m and d['parameter_vector'] == prop['candidate_vector'], 'Candidate identity mismatch')
                derive(source)
                terms = [t for t in prop['training_rows'] if t['material'] == m]
                before = sum((t['residual']/t['sigma'])**2 for t in terms)
                after = sum(((d[t['observable']]/t['reference']-1)/t['sigma'])**2 for t in terms)
                require(math.isfinite(after), 'Nonfinite loss')
                row.update(calculation='passed_local_checks', loss_before=before, loss_after=after)
                row['summary_sha256'] = sha(source/'summary.json')
            except (ValueError, KeyError, OSError, AssertionError) as exc:
                row.update(calculation='invalid', error=str(exc))
        if (spectral/'summary.json').exists():
            try:
                row.update(audit_spectrum(source, spectral), spectrum='passed')
            except (ValueError, KeyError, OSError, AssertionError) as exc:
                row.update(spectrum='invalid', spectrum_error=str(exc))
        rows.append(row)
    complete = all(x['calculation'] == 'passed_local_checks' for x in rows)
    result = {'accepted': False, 'rows': rows, 'all_calculations_complete': complete,
              'all_spectra_passed': all(x['spectrum'] == 'passed' for x in rows),
              'scope': 'Audit only; candidate convergence budgets and derivative coverage still require review.',
              'inputs': {k: {'path': str(getattr(args,k).resolve()), 'sha256': sha(getattr(args,k))}
                         for k in ['baseline','proposal','candidate','spectra']},
              'audit_script_sha256': sha(__file__)}
    if complete:
        validate_trial(args.baseline, args.candidate, prop['holdouts'])
        actual = prop['loss_before'] - sum(x['loss_after'] for x in rows)
        predicted = prop['loss_before'] - prop['predicted_loss']
        result.update(actual_reduction=actual, predicted_reduction=predicted,
                      rho=actual/predicted if predicted > 0 else None)
    args.out.mkdir(parents=True)
    (args.out/'audit.json').write_text(json.dumps(result, indent=2))
    fields = sorted(set().union(*(x.keys() for x in rows)))
    with (args.out/'materials.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
