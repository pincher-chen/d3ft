"""Recompute numerical pairs and identify their relationship to frozen rows.

All five independent axes remain explicit. Matching settings are not the same
as having tested the exact frozen endpoint. Pair differences are empirical
resolution evidence, never certified total error bounds or derivative bounds.
"""
import argparse
import hashlib
import json
from pathlib import Path

from check_elastic_convergence import compare


AXES = ['cutoff', 'grid', 'k', 'xc_grid', 'strain']
FILES = ['metadata.json', 'summary.json', 'points.json', 'mask.npy']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(meta):
    cfg, args = meta['config'], meta['arguments']
    return dict(cutoff=cfg['cutoff_energy'], grid=cfg['grid_sizes'],
        k=cfg['k_grid_sizes'], xc_grid=cfg['d3ft'].get('xc_grid_factor', 1),
        strain=args.get('strain_step', .01), force_floor=args.get('asr_force_floor', 0),
        xc=cfg['xc'], empty_bands=cfg['empty_bands'], spin=cfg['spin'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--certificates', type=Path, nargs='+', required=True)
    ap.add_argument('--core-equivalence', type=Path, nargs='*', default=[])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('Use a new audit output directory')
    root = Path(__file__).resolve().parents[3]
    baseline = {m: Path(p) for m, p in read(args.manifest).items()}
    metadata = {m: read(p/'metadata.json') for m, p in baseline.items()}
    fingerprints = {m: {f: sha(p/f) for f in FILES} for m, p in baseline.items()}
    equivalences = []
    for path in args.core_equivalence:
        evidence = read(path)
        if (not evidence.get('passed') or not evidence.get('cases') or
            not all(c['exact_elementwise_equal'] for c in evidence['cases'])):
            raise ValueError('Core equivalence lacks exact implementation comparison')
        for source, digest in evidence['input_files'].items():
            if sha(source) != digest:
                raise ValueError('Equivalence implementation source changed')
        equivalences.append(dict(path=evidence['path'], old=evidence['old_sha256'],
            new=evidence['new_sha256'], evidence=str(path.resolve()), sha256=sha(path)))
    def equivalent_core(actual, expected):
        differences = [k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)]
        return all(any(k == e['path'] and {actual.get(k), expected.get(k)} ==
            {e['old'], e['new']} for e in equivalences) for k in differences)
    candidates = set()
    for path in args.certificates:
        candidates.update(path.rglob('*certificate.json') if path.is_dir() else [path])
    records = []
    for path in sorted(candidates):
        cert = read(path)
        axis, material = cert.get('axis'), cert.get('material')
        if axis not in AXES or material not in baseline or len(cert.get('sources', {})) != 2:
            continue
        row = dict(certificate=str(path.resolve()), certificate_sha256=sha(path),
            material=material, axis=axis, recomputed_pair_passed=False,
            exact_frozen_endpoint=False, same_setting_endpoint=False)
        try:
            sources = [Path(p) for p in cert['sources']]
            sources = [p if p.is_absolute() else root/p for p in sources]
            for source, (_, saved) in zip(sources, cert['sources'].items()):
                if set(saved) != set(FILES):
                    raise ValueError('Different solver/source-file format; not a D3FT pair')
                if any(sha(source/f) != saved[f] for f in FILES):
                    raise ValueError('Stored source fingerprints do not match raw evidence')
            recomputed = compare(*sources, axis,
                fixed_physical_xc_grid=cert.get('fixed_physical_xc_grid', False))
            # Keep failures; only an actual re-comparison can qualify a pair.
            row['recomputed_pair_passed'] = recomputed['passed']
            row['rows'] = recomputed['rows']
            row['sources'] = recomputed['sources']
            expected = metadata[material]
            base_vector = read(baseline[material]/'summary.json')['parameter_vector']
            endpoints = []
            for source in sources:
                meta = read(source/'metadata.json')
                endpoints.append(dict(path=str(source),
                    vector_matches=meta['parameter_vector'] == base_vector,
                    settings_match=settings(meta) == settings(expected),
                    core_matches=meta['code_sha256'] == expected['code_sha256'],
                    pp_matches=meta['pseudopotentials'] == expected['pseudopotentials'],
                    worker_matches=meta['workflow_sha256'].get('worker.py') ==
                        expected['workflow_sha256'].get('worker.py'),
                    exact_fingerprint=all(sha(source/f) == fingerprints[material][f] for f in FILES)))
            row['endpoints'] = endpoints
            compatible = lambda e: all(e[k] for k in ['vector_matches', 'settings_match',
                'core_matches', 'pp_matches', 'worker_matches'])
            row['same_setting_endpoint'] = any(compatible(e) for e in endpoints)
            row['exact_frozen_endpoint'] = any(e['exact_fingerprint'] for e in endpoints)
            # Both numerical endpoints must use the same frozen physical core.
            pair_cores = [read(p/'metadata.json')['code_sha256'] for p in sources]
            row['pair_core_matches_frozen'] = all(x == expected['code_sha256'] for x in pair_cores)
            row['pair_core_matches_or_explicit_equivalence'] = all(
                equivalent_core(x, expected['code_sha256']) for x in pair_cores)
            row['pair_worker_matches_frozen'] = all(read(p/'metadata.json')['workflow_sha256'].get('worker.py') ==
                expected['workflow_sha256'].get('worker.py') for p in sources)
            row['direct_frozen_axis_passed'] = bool(recomputed['passed'] and
                row['exact_frozen_endpoint'] and row['pair_core_matches_frozen'] and
                row['pair_worker_matches_frozen'])
            row['frozen_axis_with_equivalence_passed'] = bool(recomputed['passed'] and
                row['exact_frozen_endpoint'] and row['pair_core_matches_or_explicit_equivalence'] and
                row['pair_worker_matches_frozen'])
        except (ValueError, KeyError, OSError, AssertionError) as exc:
            row['error'] = str(exc)
            row['direct_frozen_axis_passed'] = False
            row['frozen_axis_with_equivalence_passed'] = False
        records.append(row)
    coverage = {}
    for material in baseline:
        coverage[material] = {axis: [r['certificate'] for r in records if
            r['material'] == material and r['axis'] == axis and r['direct_frozen_axis_passed']]
            for axis in AXES}
    equivalent_coverage = {m: {axis: [r['certificate'] for r in records if
        r['material'] == m and r['axis'] == axis and r['frozen_axis_with_equivalence_passed']]
        for axis in AXES} for m in baseline}
    report = dict(manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
        baseline_fingerprints=fingerprints, axes=AXES, coverage=coverage, records=records,
        documented_core_equivalences=equivalences, coverage_with_equivalence=equivalent_coverage,
        numerical_axes_fully_screened=all(coverage[m][a] for m in baseline for a in AXES),
        certified_total_error_bounds=False, parameter_derivative_convergence_established=False,
        accepted=False, script_sha256=sha(__file__),
        comparison_script_sha256=sha(Path(__file__).with_name('check_elastic_convergence.py')),
        scope='Exact frozen forward-observable pair coverage only. Missing axes and failed/mismatched evidence stay visible. Individual pair tolerances do not bound total numerical error or candidate-parameter errors.')
    args.out.mkdir(parents=True)
    (args.out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(pairs_examined=len(records), direct_passed_axes={m:
        [a for a in AXES if coverage[m][a]] for m in baseline},
        fully_screened=report['numerical_axes_fully_screened'])))


if __name__ == '__main__':
    main()
