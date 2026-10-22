#!/usr/bin/env python3
"""Render exact archived S2 structures with Figure 3c colours, on CPU only."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import journal_style as js

ROOT = Path(__file__).resolve().parents[1]
STATES = [('start', 'Design start'), ('reverse', 'Reverse endpoint'),
          ('fd_candidate', 'Separate FD candidate')]
OVITO_PYTHON = Path(os.environ.get('OVITO_PYTHON', sys.executable))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render_one(name, output, width, samples):
    output.mkdir(parents=True, exist_ok=True)
    command = [str(OVITO_PYTHON), str(ROOT/'script/render_structure.py'),
               '--data-dir', str(ROOT/'data'/name), '--output-dir', str(output),
               '--width', str(width), '--samples', str(samples), '--bond-mode', 'both']
    with (output/'render.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Rendered {name}: {width}px / {samples} samples', flush=True)
    return json.loads((output/'render_report.json').read_text())


def triad(ax, cell, style):
    """Project real cell-axis directions with the same orthographic camera."""
    elev, azim = np.deg2rad([style['camera']['elevation_degrees'],
                            style['camera']['azimuth_degrees']])
    view = -np.array([np.cos(elev)*np.cos(azim), np.cos(elev)*np.sin(azim), np.sin(elev)])
    right = np.cross(view, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, view)
    projected = np.column_stack((cell@right, cell@up))
    projected *= .12 / np.linalg.norm(projected, axis=1).max()
    origin = np.array([.17, .17])
    for label, delta in zip('abc', projected):
        end = origin+delta
        ax.annotate('', xy=end, xytext=origin, xycoords='axes fraction',
                    arrowprops={'arrowstyle': '-|>', 'lw': .8, 'color': '#111111',
                                'mutation_scale': 6})
        unit = delta/np.linalg.norm(delta)
        ax.text(*(end+unit*.026), label, transform=ax.transAxes, fontsize=7,
                ha='center', va='center', color='#111111')


def main(args):
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT/'data/scalar_path_protocol.json').read_text())
    style = json.loads((ROOT/'data/style.json').read_text())
    if not args.compose_only:
        # Small checks also establish one common scale across all three cells.
        previews = [render_one(name, output/'preview'/name, 700, 4) for name, _ in STATES]
        fixed_fov = max(r['fov'] for r in previews)
        style['camera']['fixed_fov_A'] = fixed_fov
        for name, _ in STATES:
            (ROOT/'data'/name/'ovito_panel_c_style.json').write_text(json.dumps(style, indent=2)+'\n')
            render_one(name, output/'renders'/name, args.width, args.samples)
    else:
        style = json.loads((ROOT/'data/start/ovito_panel_c_style.json').read_text())
    images = [np.asarray(Image.open(output/'renders'/name/'structure_endpoint_clean.png').convert('RGB'))
              for name, _ in STATES]
    # One union crop: never independently resize geometries and hide cell changes.
    masks = [np.any(a < 248, axis=2) for a in images]
    bounds = []
    for mask in masks:
        ys, xs = np.where(mask); bounds.append((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
    x0 = max(0, min(b[0] for b in bounds)-32); y0 = max(0, min(b[1] for b in bounds)-32)
    x1 = min(images[0].shape[1], max(b[2] for b in bounds)+32)
    y1 = min(images[0].shape[0], max(b[3] for b in bounds)+32)
    js.apply()
    fig = plt.figure(figsize=js.mm(183, 84))
    for i, ((name, title), image) in enumerate(zip(STATES, images)):
        ax = fig.add_axes([.012+i*.329, .23, .318, .66])
        ax.imshow(image[y0:y1, x0:x1]); ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_off()
        js.panel_title(ax, 'abc'[i], title)
        state = protocol['states'][name]; cell = np.asarray(state['cell_angstrom'])
        triad(ax, cell, style)
        a, b, c, alpha, beta, gamma = state['cell_parameters']
        fig.text(.171+i*.329, .175,
                 rf'$a,b,c$ = {a:.3f}, {b:.3f}, {c:.3f} Å'+'\n'+
                 rf'$\alpha,\beta,\gamma$ = {alpha:.2f}, {beta:.2f}, {gamma:.2f}°',
                 ha='center', va='top', fontsize=7, linespacing=1.5)
    js.export(fig, output/'Figure_S2')
    checks = {}
    for name, _ in STATES:
        report = json.loads((output/'renders'/name/'render_report.json').read_text())
        assert report['atom_counts'] == {'Mg':4, 'Si':4, 'N':8}, report['atom_counts']
        assert report['actual_endpoint_equals_affine_plus_u_max_error_A'] < 1e-10
        assert report['geometric_bond_counts'] == {'Mg-N':16, 'N-Si':16}, report['geometric_bond_counts']
        checks[name] = {k:report[k] for k in ['atom_counts','geometric_bond_counts','fov',
                        'actual_endpoint_equals_affine_plus_u_max_error_A']}
    assert np.ptp([r['fov'] for r in checks.values()]) < 1e-12
    report = {'states':checks, 'common_camera_and_scale':True, 'union_crop_px':[int(x) for x in [x0,y0,x1,y1]],
              'no_displacement_arrows':True, 'no_GPU_or_DFT_calculation':True,
              'colours':style['colors'], 'size_mm':[183,84],
              'source_protocol_sha256':sha(ROOT/'data/scalar_path_protocol.json'),
              'outputs':{p.name:sha(p) for p in output.glob('Figure_S2.*')}}
    (output/'QA.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Complete:', output/'Figure_S2.png', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--width', type=int, default=2400)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--compose-only', action='store_true')
    main(parser.parse_args())
