"""Source-grounded posterior bands QA, comparison CSV and two-panel preview only."""
import csv,hashlib,json,importlib.metadata
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]
M=json.loads((R/'metadata/protocol.json').read_text())
P=json.loads((R/'inputs/path.json').read_text())
for f,h in M['source_files_sha256'].items():
 assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,('Source changed',f)
for f,h in M['input_files_sha256'].items():
 assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,('Input changed',f)
energies=[];checks=[]
for i in range(56):
 with np.load(R/'raw/candidate'/f'point_{i:03d}.npz') as z:
    energies.append(z['energies_Ha'].copy())
    assert np.max(abs(z['k_fractional']-P['fractional_calculation'][i]))<1e-12
 checks.append(json.loads((R/'raw/candidate'/f'point_{i:03d}.json').read_text()))
a=np.array(energies)
with np.load(R/'raw/QE/bands.npz') as z:b=z['energies_Ha'].copy();qk=z['k_fractional'].copy()
assert a.shape==b.shape==(56,40)
assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
assert np.max(abs(qk-np.array(P['fractional_calculation'])))<1e-8
private_density=R/'raw/QE/tmp/calc.save/charge-density.dat'
original_density=M['QE_reference']+'/rep2/tmp/calc.save/charge-density.dat'
assert hashlib.sha256(private_density.read_bytes()).hexdigest()==M['source_files_sha256'][original_density], 'QE NSCF changed the copied SCF density'
maximum_ritz=max(max(c['Ritz_residuals_Ha']) for c in checks)
assert maximum_ritz<=1e-8
nb=17;conversion=27.211386245988
delta=a[:,:nb]-b[:,:nb];shift=float(delta.mean());aligned=(delta-shift)*conversion

def stats(bands):
 d=aligned[:,bands]
 return dict(bands_one_based=[i+1 for i in bands],points=56,values=d.size,
             MAE_eV=float(np.mean(abs(d))),maximum_abs_eV=float(np.max(abs(d))),
             RMSE_eV=float(np.sqrt(np.mean(d*d))))

with np.load(R/'inputs/original_audit_spectrum.npz') as z:ae=z['energies_Ha'];ak=z['k_fractional']
with np.load(R/'inputs/QE_original_spectrum.npz') as z:qe=z['energies_Ha'];qmesh=z['k_fractional']
kp=np.array(P['fractional_calculation']);overlaps=[]
for i,p in enumerate(kp):
 differences=p-ak;differences-=np.rint(differences);idx=int(np.argmin(np.max(abs(differences),axis=1)))
 if np.max(abs(differences[idx]))<1e-9:
    qdiff=p-qmesh;qdiff-=np.rint(qdiff);qi=int(np.argmin(np.max(abs(qdiff),axis=1)))
    assert np.max(abs(qdiff[qi]))<1e-8
    # Equivalent reciprocal representatives differ for a finite common-G sphere.
    exact=np.max(abs(p-ak[idx]))<1e-9
    ca=float(np.max(abs(a[i,:nb]-ae[idx,:nb])))
    qa=float(np.max(abs(b[i,:nb]-qe[qi,:nb])))
    overlaps.append(dict(point=i,k_fractional=p.tolist(),SCF_audit_point=idx,QE_mesh_point=qi,
         same_reciprocal_representative=bool(exact),candidate_audit_max_abs_Ha=ca,
         candidate_audit_max_abs_eV=ca*conversion,QE_original_max_abs_Ha=qa,
         QE_original_max_abs_eV=qa*conversion))
exact=[v for v in overlaps if v['same_reciprocal_representative']]
assert exact and max(v['candidate_audit_max_abs_Ha'] for v in exact)<1e-8, 'Existing audit consistency failed'
assert max(v['QE_original_max_abs_Ha'] for v in overlaps)<1e-7, 'QE mesh consistency failed'
gamma=[v for v in overlaps if np.linalg.norm(v['k_fractional'])<1e-12]
assert len(gamma)==2
gamma_max=max(v['candidate_audit_max_abs_eV'] for v in gamma)
repeated_gamma=float(np.max(abs(a[0,:nb]-a[40,:nb]))*conversion)
assert repeated_gamma<1e-7
labels={p:l for p,l in zip(P['ticks'],P['labels'])}
with (R/'processed/band_comparison.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['point','band','distance_inverse_A','candidate_Ha','QE_Ha','aligned_delta_eV','label'])
 for i in range(56):
    for j in range(nb):w.writerow([i,j,P['distance_inverse_A'][i],a[i,j],b[i,j],aligned[i,j],labels.get(i,'')])
np.savez_compressed(R/'processed/bands_all40.npz',candidate_Ha=a,QE_Ha=b,k_fractional=kp,
                    distance_inverse_A=P['distance_inverse_A'])
# Energy origin is QE's path valence maximum, distinct from spectral alignment.
vbm=float(np.max(b[:,8]));x=np.array(P['distance_inverse_A'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
 'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,2,figsize=(7.0,3.1),gridspec_kw={'width_ratios':[1.35,1]})
fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.88,wspace=.32)
for j in range(nb):
 ax[0].plot(x,(a[:,j]-shift-vbm)*conversion,color='#0072B2',lw=.85,
             label='This work' if j==0 else None)
 ax[0].plot(x,(b[:,j]-vbm)*conversion,color='#D55E00',marker='o',ls='none',
             ms=2.5,mew=.55,mfc='none',markevery=list(range(0,56,2))+[55],label='QE' if j==0 else None)
 color='#7B3294' if j<5 else '#0072B2' if j<9 else '#D55E00'
 ax[1].plot(x,aligned[:,j]*1000,color=color,lw=.75,alpha=.85,
             label='Ga 3d (1–5)' if j==0 else 'Other valence (6–9)' if j==5 else 'Conduction (10–17)' if j==9 else None)
ticks=x[P['ticks']];ticklabels=[r'$\Gamma$' if l=='Gamma' else l for l in P['labels']]
for axes in ax:
 axes.set_xticks(ticks,ticklabels);axes.set_xlim(x[0],x[-1]);axes.set_xlabel('High-symmetry path')
 axes.spines[['top','right']].set_visible(False)
 for t in ticks[1:-1]:axes.axvline(t,color='#BBBBBB',lw=.5,zorder=0)
 axes.axhline(0,color='#999999',lw=.5,zorder=0)
ax[0].set_ylabel(r'Energy relative to QE VBM (eV)')
ax[1].set_ylabel(r'Aligned difference (meV)')
limit=max(.025,float(np.ceil(np.max(abs(aligned))*1000/.025)*.025))
ax[1].set_ylim(-1.15*limit,1.15*limit)
ax[1].set_yticks(np.linspace(-limit,limit,5))
ax[0].legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0,1.14),columnspacing=1.)
ax[1].legend(loc='best',frameon=False,fontsize=7,handlelength=1.2)
for label,axes in zip(['(a)','(b)'],ax):axes.text(-.16,1.04,label,transform=axes.transAxes,fontweight='bold',fontsize=9)
fig.canvas.draw();canvas=fig.get_window_extent();outside=[]
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible() and t.get_text():
    bb=t.get_window_extent(fig.canvas.get_renderer())
    if bb.x0<canvas.x0-.5 or bb.x1>canvas.x1+.5 or bb.y0<canvas.y0-.5 or bb.y1>canvas.y1+.5:outside.append(t.get_text())
assert not outside,('Preview text outside canvas',outside)
for ext in ['pdf','png']:fig.savefig(R/'preview'/('GaAs_bands_preview.'+ext),dpi=350,facecolor='white')
minimum_font=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text())
assert minimum_font>=7
report=dict(complete=True,passed=True,physical_settings=M['physical_settings'],
 environment={name:importlib.metadata.version(name) for name in ['jax','jaxlib','numpy','scipy','matplotlib']},
 QE_version='7.5',QE_fixed_density_hash_unchanged=True,
 points=56,bands_compared=17,bands_solved=40,path=P,
 global_shift_Ha=shift,global_shift_eV=shift*conversion,
 alignment='One mean candidate-QE shift across all56x17 entries; no separate shifts by band/path/group',
 energy_plot_origin='QE path valence maximum (band9); same shift for candidate',
 QE_path_VBM_Ha=vbm,all_bands=stats(list(range(17))),
 valence_including_3d=stats(list(range(9))),valence_excluding_3d=stats(list(range(5,9))),
 conduction=stats(list(range(9,17))),Ga3d=stats(list(range(5))),
 group_definition='Sorted band indices: isolated Ga3d manifold1–5, other valence6–9, conduction10–17; valenceinclusive includes3d.',
 Ga3d_to_other_valence_min_separation_eV=float(np.min(b[:,5]-b[:,4])*conversion),
 maximum_Ritz_residual_Ha=maximum_ritz,
 original_mesh_overlap_checks=overlaps,Gamma_candidate_audit_max_abs_eV=gamma_max,
 repeated_Gamma_candidate_max_abs_eV=repeated_gamma,
 Gamma_QE_original_max_abs_eV=max(v['QE_original_max_abs_eV'] for v in gamma),
 min_empty_buffer_Ha=min(c['buffer_above_highest_compared_Ha'] for c in checks),
 Ritz_and_operator_checks=checks,source_files_sha256=M['source_files_sha256'],
 preview_QA=dict(width_in=7,height_in=3.1,min_font_pt=minimum_font,out_of_canvas=outside),
 limitations='Finite-temperature scalar-relativistic KS bands, noSOC; frozen common-G versus QE shifted-G+k cutoff spheres retained; not experimental/quasiparticle gap')
files=[R/'processed/band_comparison.csv',R/'processed/bands_all40.npz',R/'preview/GaAs_bands_preview.pdf',R/'preview/GaAs_bands_preview.png']
report['output_files_sha256']={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(R/'processed/QA.json').write_text(json.dumps(report,indent=2)+'\n')
summary=f"MAE: {report['all_bands']['MAE_eV']:.8g} eV (56 points × 17 bands, one global shift).\nMaximum aligned deviation: {report['all_bands']['maximum_abs_eV']:.8g} eV.\nGamma consistency vs original candidate audit: {gamma_max:.8g} eV; QE vs original: {report['Gamma_QE_original_max_abs_eV']:.8g} eV.\n"
(R/'processed/terminal_report.txt').write_text(summary)
with (R/'README.md').open('a') as f:
 f.write('\n## Completed quantitative checks\n'+summary+'\nGroup metrics(eV):\n'+json.dumps({n:report[n] for n in ['valence_including_3d','valence_excluding_3d','conduction','Ga3d']},indent=2)+'\nMaximum Ritz residual='+str(maximum_ritz)+'Ha. Full Ritz residuals, source hashes, overlap checks and figureQA in processed/QA.json.\n')
print(summary,end='',flush=True)
