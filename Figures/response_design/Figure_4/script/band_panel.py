"""E8 panel using pymatgen electronic structures and Sumo's band plotter."""
from pathlib import Path
import json
import hashlib
import importlib.metadata

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from pymatgen.core import Lattice
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
from sumo.plotting.bs_plotter import SBSPlotter

BLUE, ORANGE = '#D81B60', '#222222'
BOHR = .529177210903
# SeeK-path special points of the standard Pna2_1 setting, in fractional
# coordinates of the actual reciprocal cell.
SEEKPATH_PNA21 = {'G': (0, 0, 0), 'X': (.5, 0, 0), 'Y': (0, .5, 0), 'Z': (0, 0, .5),
                  'S': (.5, .5, 0), 'T': (0, .5, .5), 'U': (.5, 0, .5), 'R': (.5, .5, .5)}


def draw_band_panel(ax, data_dir, path_scope="main"):
    data_dir = Path(data_dir)
    raw = np.load(data_dir/'raw/e8_corrected_aligned_bands.npz')
    protocol = json.loads((data_dir/'raw/e8_parent_path.json').read_text())
    k = raw['k_fractional']
    assert k.shape == (61,3)
    assert np.allclose(k, protocol['k_fractional'], rtol=0, atol=1e-12)
    lines = (data_dir/'raw/e8_scf.in').read_text().splitlines()
    i = next(i for i,line in enumerate(lines) if line.strip().startswith('CELL_PARAMETERS'))
    assert 'bohr' in lines[i].lower()
    real_lattice = np.array([[float(v) for v in lines[i+j].split()] for j in [1,2,3]])*BOHR
    reciprocal = Lattice(real_lattice).reciprocal_lattice
    # The archived E8 path was generated with ASE Cell.bandpath, whose
    # Setyawan-Curtarolo orthorhombic convention reorders axes (a<b<c).  The
    # manuscript and the E10 path use SeeK-path labels in the standard Pna2_1
    # setting, which coincides with this cell's axes (verified with SeeK-path
    # 2.2.2 on the start structure: identity primitive transformation).  Only
    # the labels are translated; k points and energies are unchanged.
    labels = {}
    relabel = {}
    for name, point in protocol['special_points'].items():
        matches = [std for std, coord in SEEKPATH_PNA21.items()
                   if np.allclose(point, coord, atol=1e-12)]
        assert len(matches) == 1, (name, point)
        relabel[name] = matches[0]
        labels[r'\Gamma' if matches[0]=='G' else matches[0]] = point
    assert len(set(relabel.values())) == len(relabel)
    arrays = {name:raw[name+'_eV']-raw[name+'_eV'][:,47].max() for name in ['D3FT','QE']}
    assert path_scope in ('main', 'full')
    original_kpoints = len(k)
    if path_scope == 'main':
        # The first continuous component ends at the second Z (index 44).
        # Full-path alignment is computed above, before selecting the display.
        assert protocol['path'] == 'GXSYGZURTZ,YT,UX,SR'
        assert np.allclose(k[44], protocol['special_points']['Z'])
        selected = np.arange(45)
        shown_path = 'GXSYGZURTZ'
    else:
        selected = np.arange(len(k))
        shown_path = protocol['path']
    shown_path = ''.join(relabel.get(c, c) for c in shown_path)
    for name, values in arrays.items():
        assert np.isclose(values[selected,47].max(), 0, atol=1e-10)
        assert np.isclose(values[selected,48].min(), values[:,48].min(), atol=1e-10)
    k = k[selected]
    arrays = {name:values[selected] for name,values in arrays.items()}
    structures = {}
    plotters = {}
    for name, values in arrays.items():
        gap = values[:,48].min()
        bs = BandStructureSymmLine(k, {Spin.up:values.T}, reciprocal,
                                  efermi=float(gap/2), labels_dict=labels)
        assert not bs.is_metal()
        assert np.isclose(bs.get_band_gap()['energy'],gap,atol=1e-10)
        structures[name] = bs
        plotters[name] = SBSPlotter(bs)

    plt.sca(ax)
    plotters['D3FT'].get_plot(plt=plt, ymin=-5, ymax=6, zero_energy=0,
        zero_line=False, vbm_cbm_marker=False, no_base_style=True, aspect=False,
        ylabel=r'$E-E_\mathrm{VBM}$ (eV)')
    for line in ax.lines:
        if len(line.get_xdata())>2:
            line.set_color(BLUE)
            line.set_linewidth(.9)
            line.set_zorder(2)
    qe = plotters['QE'].bs_plot_data(zero_to_efermi=True)
    for x, bands in zip(qe['distances'],qe['energy'][str(Spin.up)]):
        for band in bands:
            if np.max(band)>=-5 and np.min(band)<=6:
                ax.plot(x,band,color=ORANGE,lw=.75,ls='--',zorder=3)
    for tick in ax.get_xticks()[1:-1]:
        ax.axvline(tick,color='#DAE0E5',lw=.4,zorder=0)
    ax.axhline(0,color='#222222',lw=.55,ls=':',zorder=1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.8)
        spine.set_color('black')
    ax.grid(False)
    ax.tick_params(axis='x',pad=4,length=0)
    ax.tick_params(axis='y',direction='in',right=True)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel('Parent-reference path',labelpad=4)
    ax.set_ylabel(r'$E-E_\mathrm{VBM}$ (eV)')
    ax.set_title('MgSiN₂: matched-model bands',pad=5,loc='left')
    forced = plotters['D3FT'].bs
    vbm = forced.get_vbm()
    cbm = forced.get_cbm()
    gap = float(cbm['energy']-vbm['energy'])
    for edge in [vbm,cbm]:
        # One occurrence suffices when the same endpoint is duplicated between branches.
        index = edge['kpoint_index'][0]
        ax.scatter(forced.distance[index],edge['energy'],s=16,facecolor='white',
                   edgecolor=BLUE,linewidth=.9,zorder=5,clip_on=False)
    ax.axhspan(0,gap,facecolor='#F3F3F3',zorder=-3)
    xmax = ax.get_xlim()[1]
    ax.annotate('',xy=(.75*xmax,gap),xytext=(.75*xmax,0),
                arrowprops=dict(arrowstyle='<->',color='#222222',lw=.9))
    ax.text(.77*xmax,gap*.50,f'$E_g$ = {gap:.3f} eV',fontsize=7.5,
            va='center',ha='left',color='#222222')
    legend = ax.legend(handles=[Line2D([],[],color=BLUE,lw=1.2,label='D3FT'),
                       Line2D([],[],color=ORANGE,lw=1.1,ls='--',label='QE')],
              frameon=True,fancybox=False,edgecolor='black',framealpha=1,
              loc='upper left',bbox_to_anchor=(.02,.62),
              fontsize=7.5,ncol=2,handlelength=1.8,columnspacing=1.0,borderpad=0.3)
    legend.get_frame().set_linewidth(0.6)
    # Export exact input values and library branch distances without interpolation.
    branch_rows = []
    plotted = plotters['D3FT'].bs_plot_data(zero_to_efermi=True)
    for branch,(x,vals) in enumerate(zip(plotted['distances'],plotted['energy'][str(Spin.up)])):
        for b,ys in enumerate(vals):
            for distance,energy in zip(x,ys):
                branch_rows.append([branch,b+1,float(distance),float(energy)])
    report = dict(
        backend='pymatgen BandStructureSymmLine + sumo SBSPlotter.get_plot',
        versions={name:importlib.metadata.version(name) for name in ['pymatgen','sumo','matplotlib','numpy']},
        original_kpoints=original_kpoints,plotted_kpoints=len(k),bands=56,
        path_scope=path_scope,plotted_path=shown_path,source_kpoint_indices=selected.tolist(),
        path=''.join(relabel.get(c, c) for c in protocol['path']),archived_ASE_path=protocol['path'],
        label_translation_ASE_to_SeeKpath=relabel,label_convention='SeeK-path, standard Pna2_1 setting',
        parent_labels=protocol['labels'],
        labels_scope=protocol['scope'],reciprocal_distance_units='1/Angstrom',
        tick_labels=[label.get_text() for label in ax.get_xticklabels()],
        tick_positions=[float(v) for v in ax.get_xticks()],
        gap_eV=gap,vbm_indices=vbm['kpoint_index'],cbm_indices=cbm['kpoint_index'],
        sampled_gap_is_direct=forced.get_band_gap()['direct'],
        energy_zero='each code own sampled VBM',interpolation='None; exact stored k points',
        DOS_available=False,DOS_not_estimated_from_path=True,
        source_sha256={name:hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
                       for name in ['raw/e8_corrected_aligned_bands.npz','raw/e8_parent_path.json','raw/e8_scf.in']})
    return report,branch_rows


if __name__ == '__main__':
    import argparse,csv
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--path-scope',choices=['main','full'],default='main',
                        help='Main continuous path or all branches for SI')
    args=parser.parse_args()
    figure=Path(__file__).resolve().parents[1]
    output=args.output_dir or figure
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(183/25.4,100/25.4))
    fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.90)
    report,rows=draw_band_panel(ax,figure/'data',path_scope=args.path_scope)
    name = 'Figure_4a_bands' if args.path_scope == 'main' else 'SI_full_path_bands'
    stem = 'band_plot' if args.path_scope == 'main' else 'SI_full_path_band_plot'
    for ext in ['pdf','svg']:fig.savefig(output/f'{name}.{ext}')
    fig.savefig(output/f'{name}.png',dpi=450)
    fig.savefig(output/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    with (output/f'{stem}_source.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['branch','band','distance_inverse_A','energy_relative_VBM_eV']);writer.writerows(rows)
    (output/f'{stem}_report.json').write_text(json.dumps(report,indent=2)+'\n')
