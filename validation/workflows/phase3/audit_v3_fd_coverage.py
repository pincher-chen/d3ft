"""Audit historical FD endpoints against frozen v3 inputs without promoting them.

Compatibility permits a fresh Jacobian comparison; it is not FD acceptance.
Only certificates recomputed from v3 and passing the joint signal gate count.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--historical', type=Path, required=True)
    ap.add_argument('--revalidated', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows, inputs, directions = [], {}, []
    inputs[str(args.manifest.resolve())] = sha(args.manifest)
    for certificate in sorted(args.historical.glob('*fd_certificate.json')):
        old = json.loads(certificate.read_text())
        paths = [Path(old[key]) for key in ('minus', 'plus')]
        material = json.loads((paths[0] / 'summary.json').read_text())['material']
        if material not in manifest:
            continue
        base = Path(manifest[material])
        metas = [json.loads((path / 'metadata.json').read_text()) for path in [base] + paths]
        summaries = [json.loads((path / 'summary.json').read_text()) for path in [base] + paths]
        issues = set()
        for path, meta in zip(paths, metas[1:]):
            for key in ('grid_sizes', 'cutoff_energy', 'k_grid_sizes', 'empty_bands'):
                if meta['config'][key] != metas[0]['config'][key]:
                    issues.add(key)
            for key, default in [('strain_step', .01), ('asr_force_floor', 0.)]:
                if meta['arguments'].get(key, default) != metas[0]['arguments'].get(key, default):
                    issues.add(key)
            if meta['config']['d3ft'].get('xc_grid_factor', 1) != metas[0]['config']['d3ft'].get('xc_grid_factor', 1):
                issues.add('xc_grid_factor')
            for key in ('pseudopotentials', 'code_sha256'):
                if meta[key] != metas[0][key]:
                    issues.add(key)
            if not np.array_equal(np.load(base / 'mask.npy'), np.load(path / 'mask.npy')):
                issues.add('ordered_Miller_mask')
        vectors = np.array([s['parameter_vector'] for s in summaries])
        if not np.allclose(vectors[0], (vectors[1] + vectors[2]) / 2, rtol=0, atol=1e-14):
            issues.add('parameter_center')
        fresh_path = args.revalidated / certificate.name
        fresh = json.loads(fresh_path.read_text()) if fresh_path.exists() else None
        if fresh is not None and Path(fresh['base']).resolve() != base.resolve():
            raise ValueError('Revalidation does not use the frozen v3 base')
        joint = bool(not issues and fresh and fresh.get('joint_gate_passed'))
        if joint:
            directions.append(vectors[2] - vectors[1])
        rows.append(dict(certificate=certificate.name, material=material,
            compatible=not issues, incompatible_fields=';'.join(sorted(issues)),
            fresh_v3_comparison=fresh is not None,
            agreement=bool(fresh and fresh.get('agreement_passed')),
            joint_gate=joint,
            unresolved=';'.join(r['observable'] for r in fresh['rows']
                if not r.get('joint_gate_passed')) if fresh else '',
            delta=(vectors[2] - vectors[1]).tolist()))
        for path in [base] + paths:
            for name in ('summary.json', 'metadata.json', 'points.json', 'mask.npy'):
                inputs[str((path / name).resolve())] = sha(path / name)
        inputs[str(certificate.resolve())] = sha(certificate)
        if fresh_path.exists():
            inputs[str(fresh_path.resolve())] = sha(fresh_path)
    rank = int(np.linalg.matrix_rank(directions)) if directions else 0
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(rows=rows, accepted_direction_rank=rank,
        all_seven_parameter_derivatives_accepted=False,
        scope='Rank measures independent jointly resolved directions in this audited archive only. It does not imply every parameter must respond in every material, nor certify candidate spectra or numerical convergence.',
        input_sha256=inputs, script_sha256=sha(Path(__file__)))
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'coverage.csv').open('x') as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(json.dumps(dict(certificates=len(rows), compatible=sum(r['compatible'] for r in rows),
        revalidated=sum(r['fresh_v3_comparison'] for r in rows),
        jointly_resolved=sum(r['joint_gate'] for r in rows), independent_rank=rank)))


if __name__ == '__main__':
    main()
