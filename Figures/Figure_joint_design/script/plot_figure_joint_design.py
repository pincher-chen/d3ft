#!/usr/bin/env python3
"""Standalone CPU plot; only this figure's data/ are read. No DFT jobs."""
import argparse, csv, hashlib, itertools, json, platform, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Local journal_style.py is shipped with this figure.
import journal_style as js
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, help='Optional separate output directory for trial styling')
parser.add_argument('--verify-data', action='store_true', help='Verify supplied immutable input copies against data/SOURCES.json')
args = parser.parse_args()
HERE = Path(__file__).resolve().parents[1]
DATA = HERE / 'data'
OUT = (args.output_dir or HERE).resolve()
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_rows(name):
    with (DATA / name).open(newline='') as f:
        return list(csv.DictReader(f))

def numeric(name):
    return [{k: float(v) for k, v in r.items()} for r in read_rows(name)]

def read_json(name):
    return json.loads((DATA / name).read_text())
if args.verify_data:
    for r in read_json('SOURCES.json')['files']:
        assert sha(DATA / r['path']) == r['sha256'], r['path']
audits = {}
js.apply()

def export(fig, name):
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    outside = []
    for ax in fig.axes:
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        for axis in [ax.xaxis, ax.yaxis]:
            lo, hi = sorted(axis.get_view_interval())
            texts.extend((t.label1 for t in axis.get_major_ticks() if lo - 1e-12 <= t.get_loc() <= hi + 1e-12))
        for t in texts:
            if t.get_visible() and t.get_text():
                b = t.get_window_extent(ren)
                if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > fig.bbox.width + 0.5 or (b.y1 > fig.bbox.height + 0.5):
                    outside.append(t.get_text())
    assert not outside, (name, outside)
    for ext in ['pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}')
    fig.savefig(OUT / f'{name}.png', dpi=450)
    fig.savefig(OUT / f'{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    audits[name] = {'size_mm': list(fig.get_size_inches() * 25.4), 'text_outside_canvas': outside, 'pdf_fonttype': 42, 'svg_editable_text': True, 'files': {f.name: sha(f) for ext in ['pdf', 'svg', 'png', 'tiff'] for f in [OUT / f'{name}.{ext}']}}
    plt.close(fig)
rows = numeric('fig3_joint54_trajectory_source.csv')
atomrows = numeric('fig3_joint54_atomic_source.csv')
assert len(rows) == 4 and len(atomrows) == 64
target = rows[0]['target_eV']
# LAYOUT: physical figure dimensions and panel spacing.
fig = plt.figure(figsize=js.mm(183, 122))
gs = fig.add_gridspec(2, 2, left=0.09, right=0.88, bottom=0.09, top=0.93, hspace=0.48, wspace=0.55)
a = fig.add_subplot(gs[0, 0])
b = fig.add_subplot(gs[0, 1])
structure = fig.add_subplot(gs[1, 0])
c = fig.add_subplot(gs[1, 1])
x = np.arange(4)
a.axhline(target, color=js.TEAL, ls='--', lw=1.25, label='Target')
a.plot(x, [q['gap_eV'] for q in rows], 's-', color=js.MAGENTA, ms=5.0, lw=1.45, label='Actual state')
a.set_ylim(4.395, 4.54)
a.set_xticks(x)
a.set_ylabel('Γ direct gap (eV)')
a.set_xlabel('Design step')
js.panel_title(a, 'a', 'Joint design')
js.framed_legend(a, loc='lower right')
js.style_ax(a, minor='y')
a.text(0.97, 0.34, 'Final error\n3.4 × 10⁻⁴ meV', ha='right', va='bottom', transform=a.transAxes, fontsize=7.5)
for j in range(4):
    vv = [r['displacement_norm_A'] * 1000 for r in atomrows if r['step'] == j]
    b.scatter(j + np.linspace(-0.09, 0.09, 16), vv, s=12, color='#C5CDD4', zorder=1)
b.plot(x, [q['max_internal_displacement_A'] * 1000 for q in rows], 's-', color=js.MAGENTA, ms=5.0, lw=1.45, label='Maximum')
b.plot(x, [q['RMS_internal_displacement_A'] * 1000 for q in rows], 's--', color=js.INK, ms=4.5, lw=1.25, label='RMS')
b.set_ylim(-0.15, 4.7)
b.set_xticks(x)
b.set_xlabel('Design step')
b.set_ylabel('Internal displacement (10⁻³ Å)')
js.panel_title(b, 'b', 'Internal coordinates')
js.framed_legend(b, loc='upper left')
js.style_ax(b, minor='y')
c.plot(x, [q['energy_change_meV_atom'] for q in rows], 's-', color=js.MAGENTA, ms=5.0, lw=1.45)
c.set_ylabel('ΔE (meV atom⁻¹)', color=js.MAGENTA)
c.set_ylim(-0.6, 0.12)
c.set_xticks(x)
c.set_xlabel('Design step')
js.panel_title(c, 'd', 'Energy and force cost')
cc = c.twinx()
cc.plot(x, [q['max_single_atom_force_eV_A'] for q in rows], 'o-', color=js.INK, ms=5.0, lw=1.45)
cc.set_ylabel('Maximum force (eV Å⁻¹)', color=js.INK)
cc.set_ylim(0.95, 1.02)
cc.set_yticks([0.96, 0.98, 1.0, 1.02])
js.style_ax(c, minor='y')
c.spines['right'].set_visible(False)
cc.spines['left'].set_visible(False)
cc.spines['top'].set_visible(False)
cc.spines['bottom'].set_visible(False)
cc.spines['right'].set_color('black')
cc.spines['right'].set_linewidth(0.8)
c.tick_params(axis='y', colors=js.MAGENTA)
c.yaxis.label.set_color(js.MAGENTA)
cc.tick_params(axis='y', direction='in', colors=js.INK, width=0.7, length=3.0)
cc.yaxis.label.set_color(js.INK)
c.text(0.22, 0.88, 'Start not relaxed', transform=c.transAxes, fontsize=7.5, color=js.INK)
# OVITO panel c: crop only the white border; no atom geometry/image pixels altered.
image = plt.imread(DATA / 'ovito_panel_c_clean.png')
foreground = np.any(image[:, :, :3] < .96, axis=2)
yy, xx = np.nonzero(foreground)
pad = round(max(xx.max()-xx.min(), yy.max()-yy.min())*.035)
x0, x1 = max(0, xx.min()-pad), min(image.shape[1], xx.max()+pad+1)
y0, y1 = max(0, yy.min()-pad), min(image.shape[0], yy.max()+pad+1)
structure.imshow(image[y0:y1, x0:x1], interpolation='antialiased')
structure.set_axis_off()
structure.set_anchor('W')
js.panel_title(structure, 'c', 'Endpoint structure')
(OUT / 'panel_c_image_usage.json').write_text(json.dumps(dict(source='data/ovito_panel_c_clean.png',sha256=sha(DATA/'ovito_panel_c_clean.png'),crop_xyxy=[int(x0),int(y0),int(x1),int(y1)],processing='White-border crop only; aspect ratio preserved',source_render=(read_json('ovito_panel_c_provenance.json')['render_source'] if (DATA/'ovito_panel_c_provenance.json').exists() else 'ovito_render_v3'),arrow_amplification=400),indent=2)+'\n')
export(fig, 'Figure_joint_design')
# OUTPUT RECORD: local input hashes and render checks.
report = {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'script_sha256': sha(__file__), 'figures': audits, 'input_sha256': {str(p.relative_to(DATA)): sha(p) for p in sorted(DATA.rglob('*')) if p.is_file()}, 'scope': 'Rendering from archived accepted data only; no electronic/ionic optimization.'}
(OUT / 'render_report.json').write_text(json.dumps(report, indent=2) + '\n')
print('Outputs:', OUT)
