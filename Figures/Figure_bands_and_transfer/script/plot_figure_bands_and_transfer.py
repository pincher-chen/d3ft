#!/usr/bin/env python3
"""Standalone CPU plot; only this figure's data/ are read. No DFT jobs."""
import argparse, csv, hashlib, itertools, json, platform, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from band_panel import draw_band_panel
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
# DATA: E8 matched-model bands, E6 hydrostatic points, E5 sampling transfer.
HA = 27.211386245988
z = np.load(DATA / 'raw/e8_corrected_aligned_bands.npz')
D3, QE, kf = z['D3FT_eV'], z['QE_eV'], z['k_fractional']
D3 = D3 - D3[:, 47].max()
QE = QE - QE[:, 47].max()
acc8 = read_json('raw/e8_full_path_acceptance.json')['metrics']
diff = np.abs(D3 - QE)
assert np.isclose(diff.max(), acc8['VBM_aligned_all_bands_max_abs_eV'], rtol=1e-9)
assert np.isclose(diff.mean(), acc8['VBM_aligned_all_bands_MAE_eV'], rtol=1e-9)
e6 = read_json('raw/e6_k2_acceptance.json')
e6l = read_json('raw/e6_latest_acceptance.json')
pts = sorted(e6['points'], key=lambda p: p['nu'])
nu = np.array([p['nu'] for p in pts])
eg = np.array([p['fundamental_eV'] for p in pts])
slope = e6['adjoint_fundamental_A_eV']
fd = {f['h']: f['fundamental_eV'] for f in e6['FD']}
lit = e6l['literature']['reference_eV']
intercept = np.mean(eg - slope * nu)
with (DATA / 'e5_kmesh_transfer_source.csv').open(newline='') as f:
    transfer = list(csv.DictReader(f))
err2 = float(transfer[0]['target_error_meV'])
err4 = float(transfer[1]['target_error_meV'])
# E10: 4^3 bands of the joint-design start and endpoint on the parent-reference path.
e10 = {}
for st in ('E4_initial', 'E4_joint54_endpoint'):
    zz = np.load(DATA / f'raw/e10_{st}_path_bands.npz')
    rr = read_json(f'raw/e10_{st}_path_result.json')
    assert rr['passed'] in (True, 'True') and zz['eigenvalues_Ha'].shape == (130, 56)
    e10[st] = (zz['eigenvalues_Ha'] * HA, zz['k_fractional'], rr)
kf10 = e10['E4_initial'][1]
assert np.allclose(kf10, e10['E4_joint54_endpoint'][1])
# Main parent-reference path, matching panel c: Γ–X–S–Y–Γ–Z–U–R–T–Z.
# The archived point list stores each shared vertex once; vertices are re-inserted here.
order10 = list(range(0, 36)) + [0] + list(range(36, 82)) + [46]
labels_at = {0: 'Γ', 10: 'X', 18: 'S', 28: 'Y', 36: 'Γ', 47: 'Z', 57: 'U', 65: 'R', 75: 'T', 83: 'Z'}
_cl = (DATA / 'raw/e8_scf.in').read_text().splitlines()
_ci = next(n for n, l in enumerate(_cl) if l.strip().startswith('CELL_PARAMETERS'))
_A = np.array([[float(x) for x in _cl[_ci + m].split()] for m in (1, 2, 3)])
_B = 2 * np.pi * np.linalg.inv(_A).T / 0.529177210903
kk = kf10[order10]
x10 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kk, axis=0) @ _B, axis=1))])
t10 = [x10[p_] for p_ in labels_at]; l10 = list(labels_at.values())
segs10 = [np.arange(len(order10))]
# LAYOUT
fig = plt.figure(figsize=(183 / 25.4, 175 / 25.4))
gs = fig.add_gridspec(3, 1, left=0.105, right=0.985, bottom=0.085, top=0.955, hspace=0.62)
r0 = gs[0].subgridspec(1, 2, width_ratios=[2.3, 1], wspace=0.28)
r1 = gs[1].subgridspec(1, 2, width_ratios=[2.3, 1], wspace=0.28)
r2 = gs[2].subgridspec(1, 2, width_ratios=[1.3, 1], wspace=0.42)
p = fig.add_subplot(r0[0, 0]); q = fig.add_subplot(r0[0, 1])
a = fig.add_subplot(r1[0, 0]); b = fig.add_subplot(r1[0, 1])
c = fig.add_subplot(r2[0, 0]); d = fig.add_subplot(r2[0, 1])
for st, col, ls, lw, lab in [('E4_initial', '#9AA5AF', '--', 0.8, 'Start'), ('E4_joint54_endpoint', js.MAGENTA, '-', 1.05, 'Endpoint')]:
    ev = e10[st][0]; ev = ev - ev[:, 47].max(); ev = ev[order10]
    for seg in segs10:
        for j in range(56):
            if ev[seg, j].max() >= -5 and ev[seg, j].min() <= 6:
                p.plot(x10[seg], ev[seg, j], color=col, ls=ls, lw=lw, zorder=3 if st != 'E4_initial' else 2, label=lab if (j == 0 and seg is segs10[0]) else None)
for t in t10:
    p.axvline(t, color='#DADFE3', lw=0.4, zorder=0)
p.axhline(0, color=js.INK, lw=0.5, ls=':')
p.set_xticks(t10, l10); p.set_xlim(x10[0], x10[-1]); p.set_ylim(-5, 6)
p.set_ylabel('$E-E_{\\rm VBM}$ (eV)')
p.set_xlabel('Parent-reference path')
js.panel_title(p, 'a', 'Start and endpoint bands')
js.framed_legend(p, handles=[Line2D([], [], color='#9AA5AF', ls='--', lw=1.1, label='Start'), Line2D([], [], color=js.MAGENTA, lw=1.3, label='Endpoint')], loc='center left', bbox_to_anchor=(0.18, 0.72), borderaxespad=0)
js.style_ax(p, minor='y')
p.tick_params(axis='x', length=0)
g0, g1 = e10['E4_initial'][2], e10['E4_joint54_endpoint'][2]
dG = 1000 * (float(g1['Gamma_transition_eV']) - float(g0['Gamma_transition_eV']))
dP = 1000 * (float(g1['path_gap_eV']) - float(g0['path_gap_eV']))
vals = [('$\\Gamma$ objective\n(design)', 100.0003, js.MAGENTA), ('$\\Gamma$ transition\n($4\\times4\\times4$)', dG, js.BLUE), ('Path gap\n($4\\times4\\times4$)', dP, js.TEAL)]
q.bar(range(3), [v for _, v, _ in vals], 0.62, color=[c_ for _, _, c_ in vals])
q.axhline(100, color=js.INK, ls='--', lw=1.1)
for i, (_, v, _) in enumerate(vals):
    q.text(i, v - 3, f'{v:.1f}', ha='center', va='top', color='white', fontsize=7.5, fontweight='bold')
q.set_xticks(range(3), ['$\\Gamma$\ndesign', '$4{\\times}4{\\times}4$\n$\\Gamma$', '$4{\\times}4{\\times}4$\npath'])
q.set_ylim(0, 125); q.set_ylabel('Gap change (meV)')
js.panel_title(q, 'b', '+100 meV transfer')
js.style_ax(q, minor='y')
band_report, band_rows = draw_band_panel(a, DATA)
js.panel_title(a, 'c', 'Matched-model bands')
js.style_ax(a, minor='y')
a.tick_params(axis='x', length=0)
b.hist(np.log10(diff.ravel() + 1e-12), bins=40, color='#C5CDD4', edgecolor='white', lw=0.3)
b.axvline(np.log10(diff.mean()), color=js.MAGENTA, lw=1.3, label='Mean')
b.axvline(np.log10(diff.max()), color=js.TEAL, lw=1.2, ls='--', label='Maximum')
b.set_xlim(-7.5, -4.2)
b.set_xlabel('log$_{10}$ |ΔE| (eV)')
b.set_ylabel('Band energies')
js.panel_title(b, 'd', '61 points × 56 bands')
js.framed_legend(b, loc='upper left')
js.style_ax(b, minor='y')
grid = np.linspace(-0.0026, 0.0026, 20)
c.plot(grid * 1e3, intercept + slope * grid, '-', color=js.MAGENTA, lw=1.45, label=f'Adjoint {slope:.4f}', zorder=2)
c.plot(grid * 1e3, intercept + lit * grid, '--', color=js.TEAL, lw=1.25, label=f'PBE {lit:.1f}', zorder=1)
c.scatter(nu * 1e3, eg, s=28, color=js.INK, zorder=3, label='Reoptimized')
c.set_xlabel('ln(V/V$_0$) (10$^{-3}$)')
c.set_ylabel('$2{\\times}2{\\times}2$ sampled gap (eV)')
js.panel_title(c, 'e', 'Deformation potential')
js.framed_legend(c, loc='upper right')
js.style_ax(c)
c.text(0.03, 0.05, f'Central FD: {fd[0.002]:.4f} (h = 0.002)\n{fd[0.001]:.4f} eV (h = 0.001)',
       transform=c.transAxes, fontsize=7, va='bottom')
d.axhspan(-1, 1, color=js.BAND, zorder=0)
d.axhline(0, color=js.INK, lw=0.7)
for yv in (1, -1):
    d.axhline(yv, color=js.INK, ls='--', lw=0.9)
d.scatter([2, 4], [err2, err4], color=js.MAGENTA, s=36, zorder=3)
d.set_xticks([2, 4], ['$2{\\times}2{\\times}2$\nDesign', '$4{\\times}4{\\times}4$\nTransfer'])
d.set_xlim(1.2, 4.8); d.set_ylim(-1.6, 2.15)
d.set_xlabel('k mesh'); d.set_ylabel('Target error (meV)')
js.panel_title(d, 'f', 'Design and transfer')
js.style_ax(d, minor=None)
for n, zv in [(2, err2), (4, err4)]:
    d.annotate(f'{zv:+.3f}', (n, zv), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=7.5, color=js.MAGENTA, fontweight='bold')
export(fig, 'Figure_bands_and_transfer')
# OUTPUT RECORD: local input hashes and render checks.
(OUT / 'band_panel_report.json').write_text(json.dumps(band_report, indent=2) + '\n')
with (OUT / 'band_plot_source.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['branch', 'band', 'distance_inverse_A', 'energy_relative_VBM_eV'])
    writer.writerows(band_rows)

report = {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'script_sha256': sha(__file__), 'figures': audits, 'band_backend': band_report['backend'], 'band_tool_versions': band_report['versions'], 'band_helper_sha256': sha(Path(__file__).with_name('band_panel.py')), 'input_sha256': {str(p.relative_to(DATA)): sha(p) for p in sorted(DATA.rglob('*')) if p.is_file()}, 'scope': 'Rendering from archived accepted data only; no electronic/ionic optimization.'}
(OUT / 'render_report.json').write_text(json.dumps(report, indent=2) + '\n')
print('Outputs:', OUT)
