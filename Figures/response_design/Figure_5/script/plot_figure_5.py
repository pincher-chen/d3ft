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
# DATA: E7 fit histories, frozen external evaluations and leave-one-material-out losses.
import ast
fit = read_json('raw/e7_fit_result.json')['full_fit_and_three_LOO']
sv = read_json('raw/e7_summary_values.json')


def history(run):
    out = []
    for h in run['history']:
        acc = h['acceptance']
        if isinstance(acc, str):
            acc = ast.literal_eval(acc)
        assert acc['accepted']
        out.append((h['proposal']['loss_before'], h['proposal']['predicted_loss'], acc['loss_after']))
    return out


runs = [('full', 'Full fit', js.MAGENTA, 'o'), ('LOO_Si', 'Without Si', js.INK, 's'),
        ('LOO_BN', 'Without BN', js.TEAL, '^'), ('LOO_diamond', 'Without diamond', js.BLUE, 'D')]
ext = {r['material']: r for r in sv['external']}
for material in ['AlP', 'GaAs']:
    accepted = {model: read_json(f'raw/{material}_{model}_acceptance.json')
                for model in ['PBE', 'fitted']}
    assert all(row['passed'] and row['material'] == material
               and row['model'] == model for model, row in accepted.items())
    assert accepted['PBE']['reference_a0_A'] == accepted['fitted']['reference_a0_A']
    ext[material] = {
        'material': material,
        'PBE_error_percent': accepted['PBE']['relative_error_percent'],
        'fitted_error_percent': accepted['fitted']['relative_error_percent'],
    }
loo = {r['material']: r for r in sv['LOO']}
# LAYOUT
base = {}
for r in read_rows('response-baseline_E7_references.csv'):
    base[(r['material'], r['observable'])] = 100 * float(r['relative_residual'])
fig = plt.figure(figsize=js.mm(183, 132))
gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.05], left=0.08, right=0.985, bottom=0.14, top=0.94, hspace=0.55)
p = fig.add_subplot(gs[0])
bot = gs[1].subgridspec(1, 3, width_ratios=[1.25, 0.85, 1.0], wspace=0.48)
a = fig.add_subplot(bot[0, 0]); b = fig.add_subplot(bot[0, 1]); c = fig.add_subplot(bot[0, 2])
# Borrow inter-panel whitespace for four external groups; preserve panels b/d.
external_box = b.get_position()
b.set_position([external_box.x0 - 0.03, external_box.y0,
                external_box.width + 0.06, external_box.height])
order7 = [('Si', 'Si'), ('BN', 'c-BN'), ('diamond', 'Diamond'), ('SiC', '3C-SiC'), ('GaP', 'GaP'), ('AlP', 'AlP'), ('AlAs', 'AlAs')]
props = [('a0_A', '$a_0$', 'o', js.MAGENTA), ('B_GPa', '$B$', 's', js.INK), ('C11_minus_C12_GPa', '$C_{11}-C_{12}$', '^', js.TEAL), ('C44_GPa', '$C_{44}$', 'D', js.BLUE)]
for lo, hi, lab in [(-0.5, 2.5, 'Training'), (2.5, 5.5, 'External'), (5.5, 6.5, 'Baseline only')]:
    p.axvspan(lo, hi, color='#F3F3F3' if lab != 'External' else '#E7E7E7', zorder=0, lw=0)
    p.text((lo + hi) / 2, 3.35, lab, ha='center', va='bottom', fontsize=7.5, color=js.INK)
p.axhline(0, color=js.INK, lw=0.7)
for k, (key, lab, mk, col) in enumerate(props):
    xs = [i + (k - 1.5) * 0.13 for i, (m, _) in enumerate(order7) if (m, key) in base]
    ys = [base[(m, key)] for m, _ in order7 if (m, key) in base]
    p.scatter(xs, ys, marker=mk, s=28, color=col, label=lab, zorder=3)
p.set_xticks(range(7), [n for _, n in order7]); p.set_xlim(-0.5, 6.5); p.set_ylim(-16, 5.2)
p.set_ylabel('PBE error (%)')
js.panel_title(p, 'a', 'Unmodified PBE/SG15')
js.framed_legend(p, loc='lower left', ncol=4, handletextpad=0.3, columnspacing=0.8)
js.style_ax(p, minor='y')
for key, label, color, marker in runs:
    h = history(fit[key])
    loss = [h[0][0]] + [q[2] for q in h]
    a.plot(range(4), loss, '-', marker=marker, color=color, ms=5, lw=1.35, label=label)
    a.scatter(range(1, 4), [q[1] for q in h], s=32, facecolor='none', edgecolor=color, lw=1.1, zorder=4)
    for q in h:
        assert abs(q[2] / q[1] - 1) < 0.02
a.set_yscale('log')
a.set_ylim(0.1, 18)
a.set_xticks(range(4))
a.set_xlabel('Fitting iteration')
a.set_ylabel('Training loss')
js.panel_title(a, 'b', 'Filled vs open')
js.framed_legend(a, loc='upper right', handlelength=1.4)
js.style_ax(a, minor='y')
mats = ['SiC', 'GaP', 'AlP', 'GaAs']
xx = np.arange(len(mats))
b.bar(xx - 0.25, [ext[m]['PBE_error_percent'] for m in mats], 0.44, color='#C5CDD4', label='PBE')
b.bar(xx + 0.25, [ext[m]['fitted_error_percent'] for m in mats], 0.44, color=js.MAGENTA, label='Frozen fit')
for i, m in enumerate(mats):
    for dx, v in [(-0.25, ext[m]['PBE_error_percent']), (0.25, ext[m]['fitted_error_percent'])]:
        b.annotate(f'{v:.2f}', xy=(i + dx, v), xytext=(-2 if dx < 0 else 2, 3),
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=7.5, color=js.INK)
b.set_xticks(xx, ['3C-SiC', 'GaP', 'AlP', 'GaAs'])
b.set_ylim(0, 2.1)
b.set_ylabel('$a_0$ error (%)')
js.panel_title(b, 'c', 'External materials')
js.framed_legend(b, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, borderaxespad=0)
js.style_ax(b, minor='y')
order = ['Si', 'BN', 'diamond']
xx = np.arange(3)
c.bar(xx - 0.18, [loo[m]['PBE_loss'] for m in order], 0.34, color='#C5CDD4', label='PBE')
c.bar(xx + 0.18, [loo[m]['fitted_loss'] for m in order], 0.34, color=js.MAGENTA, label='Fitted without it')
for i, m in enumerate(order):
    c.text(i + 0.18, loo[m]['fitted_loss'] * 1.18, f'×{loo[m]["ratio"]:.2f}',
           ha='right' if i == len(order) - 1 else 'center', fontsize=7.5)
    if not loo[m]['fine_passed']:
        c.text(0.05, 0.70, '$B$ check failed', transform=c.transAxes,
               ha='left', va='bottom', fontsize=7, color=js.INK)
c.set_yscale('log')
c.set_ylim(0.04, 220)
c.set_xticks(xx, ['Si', 'c-BN', 'Diamond'])
c.set_ylabel('Held-out loss')
js.panel_title(c, 'd', 'Leave one out')
js.framed_legend(c, loc='upper left')
js.style_ax(c, minor='y')
export(fig, 'Figure_5')
# OUTPUT RECORD: local input hashes and render checks.
report = {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'script_sha256': sha(__file__), 'figures': audits, 'input_sha256': {str(p.relative_to(DATA)): sha(p) for p in sorted(DATA.rglob('*')) if p.is_file()}, 'scope': 'Rendering from archived accepted data only; no electronic/ionic optimization.'}
(OUT / 'render_report.json').write_text(json.dumps(report, indent=2) + '\n')
print('Outputs:', OUT)
