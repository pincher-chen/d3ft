#!/usr/bin/env python3
"""Publication-style OVITO/Tachyon MgSiN2 panel, preserving original atomic data.

Place this script under Figure_3/script/ (or pass --data-dir).
Requires: fig3_structure_source.csv, cell_A.json, endpoint_input.xyz,
          ovito_panel_c_style.json in the data directory.
"""
import argparse
import copy
import csv
from collections import Counter
import ctypes
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

# Must precede importing OVITO / Qt.
os.environ.setdefault('QT_QPA_PLATFORM', 'ovitoheadless')
software_icd = Path('/usr/share/vulkan/icd.d/lvp_icd.x86_64.json')
if software_icd.exists():
    os.environ.setdefault('VK_ICD_FILENAMES', str(software_icd))
os.environ.setdefault('OVITO_THREAD_COUNT', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import ovito
try:
    ctypes.CDLL('libOpenGL.so.0', mode=ctypes.RTLD_GLOBAL)
except OSError:
    fallback = Path(ovito.__file__).parent / 'plugins/syslibs/libOpenGL.so.0'
    if fallback.exists():
        ctypes.CDLL(str(fallback), mode=ctypes.RTLD_GLOBAL)

from ovito.data import DataCollection, ParticleType, Particles, SimulationCell
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier
from ovito.pipeline import Pipeline, StaticSource
from ovito.qt_compat import QtCore, QtGui
from ovito.vis import TachyonRenderer, VectorVis, Viewport


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(color):
    return tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))


def save_image(image, path, dpi):
    dpm = round(dpi / 0.0254)
    image.setDotsPerMeterX(dpm)
    image.setDotsPerMeterY(dpm)
    if not image.save(str(path)):
        raise RuntimeError('Failed to save ' + str(path))


def save_pdf(image, path):
    writer = QtGui.QPdfWriter(str(path))
    writer.setResolution(600)
    writer.setPageSize(QtGui.QPageSize(QtCore.QSizeF(90, 90*image.height()/image.width()), QtGui.QPageSize.Unit.Millimeter))
    writer.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
    painter = QtGui.QPainter(writer)
    painter.drawImage(QtCore.QRectF(0, 0, writer.width(), writer.height()), image)
    painter.end()


def optimized_style(original, args):
    s = copy.deepcopy(original)
    for element in s['radii_A']:
        s['radii_A'][element] *= args.atom_scale
    s['cell_line_width_A'] *= args.cell_scale
    s['arrow_width_A'] *= args.arrow_scale
    s['bonds']['width_A'] *= args.bond_scale
    if args.arrow_amplification is not None:
        s['arrow_amplification'] = args.arrow_amplification
    if args.elevation is not None:
        s['camera']['elevation_degrees'] = args.elevation
    if args.azimuth is not None:
        s['camera']['azimuth_degrees'] = args.azimuth
    # Fixed framing for the paired reference; more whitespace than default.
    s['camera']['zoom_padding'] *= args.padding_scale
    if args.samples is not None:
        s['raytracing']['antialiasing_samples'] = args.samples
    if args.width is not None:
        ow, oh = s['size_px']
        s['size_px'] = [args.width, round(args.width * oh / ow)]
    return s


def build_scene(data_dir, style, no_bonds, bond_mode):
    with (data_dir / 'fig3_structure_source.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    symbols = [r['species'] for r in rows]
    affine = np.array([[float(r[k]) for k in ('x_A','y_A','z_A')] for r in rows])
    u = np.array([[float(r[k]) for k in ('ux_A','uy_A','uz_A')] for r in rows])
    cell = np.asarray(json.loads((data_dir / 'cell_A.json').read_text())['lattice_row_vectors_A'], dtype=float)
    endpoint = import_file(str(data_dir / 'endpoint_input.xyz')).compute()
    actual = np.asarray(endpoint.particles.positions).copy()
    assert len(rows)>0 and len(actual)==len(rows)
    if not set(symbols).issubset(style['colors']) or not set(symbols).issubset(style['radii_A']):
        raise ValueError('Provide display colors/radii for every species in the style JSON')
    mismatch = float(np.max(np.abs(actual - affine - u)))
    assert mismatch < 1e-10, f'Endpoint != affine + displacement, max error {mismatch:g} Å'
    assert np.max(np.abs(np.asarray(endpoint.cell)[:,:3] - cell.T)) < 1e-10
    # Only original label scaling is checked if not modified in CLI.
    if 'arrow_amplification' in rows[0]:
        archive_factors = {float(r['arrow_amplification']) for r in rows}
        assert len(archive_factors) == 1, 'Inconsistent archive arrow factors'

    particles = Particles(count=len(rows))
    particles.create_property('Position', data=actual)
    identifiers = {name:i+1 for i,name in enumerate(dict.fromkeys(symbols))}
    types = particles.create_property('Particle Type', data=[identifiers[s] for s in symbols])
    for name, identifier in identifiers.items():
        types.types.append(ParticleType(id=identifier, name=name,
            color=rgb(style['colors'][name]), radius=style['radii_A'][name]))
    lattice = SimulationCell(pbc=tuple(endpoint.cell.pbc))
    lattice[...] = np.column_stack((cell.T, np.zeros(3)))
    lattice.vis.line_width = style['cell_line_width_A']
    lattice.vis.rendering_color = rgb(style['colors']['cell'])
    atoms = Pipeline(source=StaticSource(data=DataCollection(objects=[particles, lattice])))
    if style['bonds']['enabled'] and not no_bonds:
        bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
        pairs=style['bonds'].get('pair_cutoffs_A', {})
        if not pairs:
            raise ValueError('Set explicit species pair_cutoffs_A or disable bonds; no invented cutoffs')
        for pair, cutoff in pairs.items():
            left,right=pair.split('-')
            if bond_mode=='si-n' and set([left,right])!=set(['Si','N']):
                continue
            if left in identifiers and right in identifiers:
                bonds.set_pairwise_cutoff(left,right,cutoff)
        bonds.vis.width = style['bonds']['width_A']
        bonds.vis.color = rgb(style['colors']['bonds'])
        atoms.modifiers.append(bonds)
    atoms.add_to_scene()

    bases = Particles(count=len(rows))
    bases.create_property('Position', data=affine)
    bases.vis.enabled = False
    vector_prop = bases.create_property('Internal displacement', data=u, components=3)
    vector_prop.vis = VectorVis(
        enabled=True, alignment=VectorVis.Alignment.Base,
        color=rgb(style['colors']['arrow']), scaling=style['arrow_amplification'],
        width=style['arrow_width_A'], flat_shading=False)
    arrows = Pipeline(source=StaticSource(data=DataCollection(objects=[bases])))
    arrows.add_to_scene()
    return atoms, arrows, symbols, affine, actual, u, cell, mismatch


def make_viewport(style):
    elev = math.radians(style['camera']['elevation_degrees'])
    azim = math.radians(style['camera']['azimuth_degrees'])
    direction = -np.array((math.cos(elev)*math.cos(azim),
                           math.cos(elev)*math.sin(azim), math.sin(elev)))
    v = Viewport(type=Viewport.Type.Ortho,
                 camera_dir=tuple(direction), camera_up=(0,0,1))
    v.zoom_all(size=tuple(style['size_px']))
    v.fov *= style['camera']['zoom_padding']
    if 'fixed_fov_A' in style['camera']:
        v.fov = style['camera']['fixed_fov_A']
    return v


def main(args):
    started = time.monotonic()
    figure = Path(__file__).resolve().parent.parent
    data = (args.data_dir or figure / 'assets/example-data').resolve()
    style_path = (args.style or data / 'ovito_panel_c_style.json').resolve()
    missing = [str(p) for p in (style_path, data/'fig3_structure_source.csv',
                data/'cell_A.json', data/'endpoint_input.xyz') if not p.exists()]
    if missing:
        raise FileNotFoundError('Required archived inputs not found:\n' + '\n'.join(missing) +
                                '\nUse --data-dir and --style to locate the original files.')
    original = json.loads(style_path.read_text())
    style = optimized_style(original, args)
    output = (args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    atoms, arrows, symbols, affine, actual, u, cell, mismatch = build_scene(data, style, args.no_bonds, args.bond_mode)
    computed = atoms.compute()
    topology = np.asarray(computed.particles.bonds.topology) if computed.particles.bonds is not None else []
    bond_counts = Counter('-'.join(sorted((symbols[int(i)], symbols[int(j)]))) for i,j in topology)
    print('Geometric neighbour counts:', dict(bond_counts), flush=True)
    viewport = make_viewport(style)
    renderer = TachyonRenderer(**style['raytracing'])
    print('Rendering clean OVITO structure', ovito.version_string, 'size', style['size_px'], flush=True)
    image = viewport.render_image(size=tuple(style['size_px']), background=(1,1,1), renderer=renderer)
    save_image(image, output/'structure_panel_clean.png', style['dpi'])
    save_image(image, output/'structure_panel_clean.tiff', style['dpi'])
    save_pdf(image, output/'structure_panel_clean.pdf')
    ovito.scene.save(str(output/'structure_panel_clean.ovito'))
    arrows.remove_from_scene()
    paired = viewport.render_image(size=tuple(style['size_px']), background=(1,1,1), renderer=renderer)
    save_image(paired, output/'structure_endpoint_clean.png', style['dpi'])
    ovito.scene.save(str(output/'structure_endpoint_clean.ovito'))
    with (output/'rendered_atom_source.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('atom','species','endpoint_x_A','endpoint_y_A','endpoint_z_A',
                         'affine_x_A','affine_y_A','affine_z_A','ux_A','uy_A','uz_A'))
        for i in range(len(symbols)):
            writer.writerow((i, symbols[i], *actual[i], *affine[i], *u[i]))
    source_files = [data/'fig3_structure_source.csv',data/'cell_A.json',data/'endpoint_input.xyz',style_path]
    report = {
        'ovito':ovito.version_string, 'python':platform.python_version(),
        'renderer':'Tachyon CPU ray tracer; Qt labels',
        'headless_platform':os.environ.get('QT_QPA_PLATFORM'),
        'vulkan_icd':os.environ.get('VK_ICD_FILENAMES'),
        'size_px':style['size_px'],
        'atom_counts':dict(Counter(symbols)),
        'geometric_bond_counts':dict(bond_counts),
        'text_annotations':False,
        'actual_endpoint_equals_affine_plus_u_max_error_A':mismatch,
        'visible_atom_positions':'actual endpoint',
        'arrow_bases':'affinely strained start',
        'arrow_amplification_display_only':style['arrow_amplification'],
        'bond_mode': 'none' if args.no_bonds else args.bond_mode,
        'bonds_are_geometric_neighbours_only':True,
        'sphere_radii_are_display_parameters':True,
        'no_new_ionic_relaxation_performed':True, 'PDF_is_embedded_raster':True,
        'camera_pos':list(viewport.camera_pos), 'camera_dir':list(viewport.camera_dir),
        'fov':viewport.fov, 'style_used':style,
        'script_sha256':sha(__file__),
        'source_sha256':{p.name:sha(p) for p in source_files},
        'seconds':time.monotonic()-started,
        'outputs_sha256':{p.name:sha(p) for p in output.iterdir()
                          if p.is_file() and p.suffix in ('.png','.tiff','.pdf','.ovito','.csv')},
    }
    (output/'render_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Completed:', output, 'elapsed:', round(report['seconds'],2),'s', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--style', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--width', type=int)
    p.add_argument('--samples', type=int)
    p.add_argument('--no-bonds', action='store_true')
    p.add_argument('--bond-mode', choices=('both','si-n'), default='both',
                   help='Default both Mg-N and Si-N geometric neighbours; si-n hides Mg-N')
    p.add_argument('--atom-scale', type=float, default=1.0)
    p.add_argument('--cell-scale', type=float, default=1.0)
    p.add_argument('--bond-scale', type=float, default=1.0)
    p.add_argument('--arrow-scale', type=float, default=1.0)
    p.add_argument('--arrow-amplification', type=float,
                   help='Override original arrow scale; displayed vectors only')
    p.add_argument('--elevation', type=float, help='Degrees, default original camera')
    p.add_argument('--azimuth', type=float, help='Degrees, default original camera')
    p.add_argument('--padding-scale', type=float, default=1.0)
    main(p.parse_args())
