"""Remap an exact FFT-ordered Miller set, with explicit safe-shrink validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def remap(mask, grid, allow_shrink=False):
    if mask.ndim != 3 or len(set(mask.shape)) != 1 or mask.dtype != bool:
        raise ValueError('Expected a cubic Boolean mask')
    if grid < 3 or not mask.any():
        raise ValueError('Need a nonempty mask and grid >= 3')
    old = mask.shape[0]
    axis = np.rint(np.fft.fftfreq(old) * old).astype(int)
    active = np.column_stack(np.nonzero(mask))
    miller = axis[active]
    bound = np.abs(miller).max(axis=0)
    if grid < old:
        if not allow_shrink:
            raise ValueError('Shrinking requires --allow-shrink')
        # Density contains differences of two wavefunction G vectors. Keep
        # both signs strictly inside Nyquist; do not collapse Nyquist pairs.
        if np.any(4 * bound >= grid):
            raise ValueError('Target grid cannot represent the full density frequency support')
    new = np.zeros((grid,) * 3, dtype=bool)
    new[tuple((miller % grid).T)] = True
    if int(new.sum()) != int(mask.sum()):
        raise ValueError('Miller indices collide in target grid')
    target_axis = np.rint(np.fft.fftfreq(grid) * grid).astype(int)
    recovered = target_axis[np.column_stack(np.nonzero(new))]
    if set(map(tuple, recovered)) != set(map(tuple, miller)):
        raise ValueError('Target grid changes Miller frequencies')
    return new, bound


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--grid', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--allow-shrink', action='store_true')
    args = ap.parse_args()
    mask = np.load(args.source)
    new, bound = remap(mask, args.grid, args.allow_shrink)
    with args.out.open('xb') as f:
        np.save(f, new)
    meta = dict(source=str(args.source.resolve()),
                source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                original_grid=mask.shape[0], new_grid=args.grid,
                num_miller_indices=int(mask.sum()), max_abs_miller=bound.tolist(),
                density_support_bound=(2 * bound).tolist(),
                output_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                scope='Exact Miller identity; XC quadrature and radial interpolation convergence remain separate checks')
    with args.out.with_suffix('.provenance.json').open('x') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
