#!/usr/bin/env python3
"""Standalone CPU plot; only this figure's data/ are read. No DFT jobs."""
import argparse, csv, hashlib, itertools, json, platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
# Saturated series colours in the AD--DFPT figure style: one accent for the
# reverse result, black for forward, and teal for the ideal reference.
MAGENTA, INK, TEAL, BLUE = ('#D81B60', '#222222', '#00838F', '#3949AB')
audits = {}
# STYLE: boxed axes, bold labels, larger type.
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8, 'axes.labelsize': 8.5, 'axes.labelweight': 'bold', 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'legend.fontsize': 7.5, 'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': 'black', 'legend.framealpha': 1, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': 0.8, 'axes.spines.top': True, 'axes.spines.right': True, 'lines.linewidth': 1.4, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.minor.width': 0.45, 'ytick.minor.width': 0.45, 'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6, 'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans', 'savefig.facecolor': 'white', 'axes.facecolor': 'white'})

def style_ax(ax, minor='both'):
    ax.tick_params(which='major', direction='in', top=True, right=True)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', top=True, right=True)
        if minor == 'y':
            ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

def framed_legend(ax, *args, **kwargs):
    leg = ax.legend(*args, frameon=True, fancybox=False, edgecolor='black', framealpha=1, borderpad=0.35, handlelength=1.7, **kwargs)
    leg.get_frame().set_linewidth(0.6)
    return leg

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
d = read_rows('fig1_derivative_source.csv')
assert len(d) == 54
rg = np.array([float(r['reverse']) for r in d])
fg = np.array([float(r['forward']) for r in d])
diff = abs(fg - rg)
assert np.allclose(diff, [float(r['absolute_difference']) for r in d], rtol=1e-12, atol=1e-16)
tr = [dict(N=int(r['N']), repeat=int(r['repeat']), reverse_seconds=float(r['reverse_seconds']), forward_seconds=float(r['forward_seconds'])) for r in read_rows('fig1_timing_source.csv')]
ns = np.array([1, 6, 24, 54])
assert len(tr) == 12 and all((sum((r['N'] == n for r in tr)) == 3 for n in ns))
means = np.array([[np.mean([r[k] for r in tr if r['N'] == n]) for k in ['reverse_seconds', 'forward_seconds']] for n in ns])
rat = means[:, 1] / means[:, 0]
# LAYOUT: physical figure dimensions and panel spacing.
fig = plt.figure(figsize=(183 / 25.4, 88 / 25.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.35, 1.05], height_ratios=[1, 1], left=0.095, right=0.985, bottom=0.13, top=0.90, wspace=0.46, hspace=0.55)
a = fig.add_subplot(gs[0, 0])
aa = fig.add_subplot(gs[1, 0])
b = fig.add_subplot(gs[:, 1])
c = fig.add_subplot(gs[:, 2])
for ax, sel, x, ylabel, marker, ms in [(a, slice(0, 6), np.arange(6), '|Forward − reverse|\n(eV / strain)', 's', 5.0), (aa, slice(6, 54), np.arange(48), '|Forward − reverse|\n(eV / Bohr)', 'o', 3.1)]:
    values = diff[sel]
    assert np.all(values > 0)
    ax.semilogy(x, values, marker, color=BLUE, ms=ms, mfc=BLUE, mec=BLUE, mew=0)
    ax.set_ylim(1e-12, 1e-08)
    ax.set_yticks([1e-11, 1e-10, 1e-9])
    ax.set_ylabel(ylabel)
    style_ax(ax, minor='y')
a.set_xticks(range(6), ['xx', 'yy', 'zz', 'yz', 'xz', 'xy'])
a.set_title('a   Same-program derivatives', loc='left', pad=5)
aa.set_xticks([0, 15, 31, 47], [1, 16, 32, 48])
aa.set_xlabel('Cartesian input index')
b.axvspan(46, 62, facecolor='#F3F3F3', edgecolor='#C8C8C8', hatch='///', lw=0.3, zorder=0)
for j, (key, color, label) in enumerate([('reverse_seconds', MAGENTA, 'Reverse'), ('forward_seconds', INK, 'Forward')]):
    for n in ns:
        vv = [r[key] for r in tr if r['N'] == n]
        b.scatter(n * np.exp(np.array([-0.04, 0, 0.04])), vv, s=18, facecolor='white', edgecolor=color, linewidths=0.9, zorder=3)
    b.plot(ns, means[:, j], color=color, marker='s', ms=5.2, lw=1.45, mfc=color, mec=color, label=label, zorder=4)
ideal_x = np.array([1.0, 60.0])
single = means[0, 1]
b.plot(ideal_x, ideal_x * single, linestyle='--', color=TEAL, lw=1.25, label='Ideal', zorder=2)
b.set_xscale('log')
b.set_yscale('log')
b.set_ylim(400, 60000)
b.set_xlim(0.7, 78)
b.set_xticks(ns, ns)
b.set_yticks([600, 3000, 10000, 30000], ['600', '3,000', '10,000', '30,000'])
b.set_xlabel('Number of inputs, N')
b.set_ylabel('Complete-gradient time (s)')
b.set_title('b   Raw timings and means', loc='left', pad=5)
handles, labels = b.get_legend_handles_labels()
handles.append(__import__('matplotlib.patches', fromlist=['Patch']).Patch(facecolor='#F3F3F3', edgecolor='#C8C8C8', hatch='///', linewidth=0.4))
labels.append('Shared node')
framed_legend(b, handles, labels, loc='upper left')
style_ax(b)
c.plot(ns, rat, 's-', color=MAGENTA, ms=5.2, lw=1.45, mfc=MAGENTA, mec=MAGENTA, label='Measured')
c.plot([1, 60], [1, 60], linestyle='--', color=TEAL, lw=1.25, label='Ideal: y = N')
c.set_xscale('log')
c.set_yscale('log')
c.set_xticks(ns, ns)
c.set_ylim(0.55, 160)
c.set_xlim(0.7, 78)
c.set_yticks([1, 10, 50], ['1', '10', '50'])
c.set_xlabel('Number of inputs, N')
c.set_ylabel('Forward / reverse time')
c.set_title('c   Time ratio', loc='left', pad=5)
framed_legend(c, loc='upper left')
style_ax(c)
for n, y, dx, dy, ha in zip(ns[1:], rat[1:], (-6, -6, -8), (10, 10, 1), ('center', 'center', 'right')):
    c.annotate(f'{y:.1f}×', (n, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va='bottom', fontsize=7.5, color=MAGENTA, fontweight='bold')
export(fig, 'Figure_response_scaling')
# OUTPUT RECORD: local input hashes and render checks.
report = {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'script_sha256': sha(__file__), 'figures': audits, 'input_sha256': {str(p.relative_to(DATA)): sha(p) for p in sorted(DATA.rglob('*')) if p.is_file()}, 'scope': 'Rendering from archived accepted data only; no electronic/ionic optimization.'}
(OUT / 'render_report.json').write_text(json.dumps(report, indent=2) + '\n')
print('Outputs:', OUT)
