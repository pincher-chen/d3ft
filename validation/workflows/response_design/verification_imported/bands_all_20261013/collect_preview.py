"""Five-system data, overlap QA and7in preview; no manuscript/figure-directory edits."""
import csv,hashlib,json,importlib.metadata
from pathlib import Path
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullLocator

R=Path(__file__).resolve().parents[1];M=json.loads((R/'metadata/protocol.json').read_text())
EV=27.211386245988;systems={};arrays={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def statistics(v):
 v=np.asarray(v)
 return dict(values=int(v.size),MAE_eV=float(np.mean(abs(v))) if v.size else None,
             maximum_abs_eV=float(np.max(abs(v))) if v.size else None,
             RMSE_eV=float(np.sqrt(np.mean(v*v))) if v.size else None)

for mat in ['Al','Na','Si','SiC']:
 d=M['systems'][mat];I=R/'inputs'/mat;O=R/'processed'/mat;P=json.loads((I/'path.json').read_text());n=P['points'];nb=d['bands_compared']
 for f,h in d['source_files_sha256'].items():assert sha(f)==h,('Source changed',f)
 for f,h in d['input_files_sha256'].items():assert sha(R/f)==h,('Input changed',f)
 assert sha(R/'raw'/mat/'QE/tmp/calc.save/charge-density.dat')==d['SCF_density_hash'],('NSCF changed reference density',mat)
 candidate=[];checks=[]
 for i in range(n):
  with np.load(R/'raw'/mat/'candidate'/f'point_{i:03d}.npz') as z:
   candidate.append(z['energies_Ha'].copy());assert np.max(abs(z['k_fractional']-P['fractional_calculation'][i]))<1e-12
  checks.append(json.loads((R/'raw'/mat/'candidate'/f'point_{i:03d}.json').read_text()))
 a=np.array(candidate)
 with np.load(R/'raw'/mat/'QE/bands.npz') as z:b=z['energies_Ha'].copy();qk=z['k_fractional'].copy()
 assert a.shape==b.shape==(n,40) and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
 assert np.max(abs(qk-P['fractional_calculation']))<1e-8
 residual=max(max(c['Ritz_residuals_Ha']) for c in checks);assert residual<=1e-8
 delta=a[:,:nb]-b[:,:nb];shift=float(delta.mean());aligned=(delta-shift)*EV
 with np.load(I/'original_audit_spectrum.npz') as z:ae=z['energies_Ha'].copy();ak=z['k_fractional'].copy()
 with np.load(I/'QE_original_spectrum.npz') as z:qe=z['energies_Ha'].copy();qm=z['k_fractional'].copy()
 overlaps=[]
 for i,p in enumerate(np.array(P['fractional_calculation'])):
  off=p-ak;off-=np.rint(off);idx=int(np.argmin(np.max(abs(off),axis=1)))
  if np.max(abs(off[idx]))<1e-9:
   offq=p-qm;offq-=np.rint(offq);qi=int(np.argmin(np.max(abs(offq),axis=1)));assert np.max(abs(offq[qi]))<1e-8
   mode='identical' if np.max(abs(p-ak[idx]))<1e-9 else 'time-reversed' if np.max(abs(p+ak[idx]))<1e-9 else 'reciprocal-translation-only'
   err=float(np.max(abs(a[i,:nb]-ae[idx,:nb])));qerr=float(np.max(abs(b[i,:nb]-qe[qi,:nb])))
   if mode!='reciprocal-translation-only':assert err<=M['original_audit_tolerance_Ha'],(mat,i,mode,err)
   assert qerr<=M['original_QE_tolerance_Ha'],(mat,i,qerr)
   overlaps.append(dict(point=i,k_fractional=p.tolist(),audit_index=idx,QE_index=qi,
                  representative_match=mode,candidate_audit_max_Ha=err,candidate_audit_max_eV=err*EV,
                  QE_original_max_Ha=qerr,QE_original_max_eV=qerr*EV))
 assert overlaps,('No original grid overlaps',mat)
 with (O/'band_comparison.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['point','band','distance_inverse_A','candidate_Ha','QE_Ha','aligned_delta_eV','label'])
  labels=dict(zip(P['ticks'],P['labels']))
  for i in range(n):
   for j in range(nb):w.writerow([i,j,P['distance_inverse_A'][i],a[i,j],b[i,j],aligned[i,j],labels.get(i,'')])
 np.savez_compressed(O/'bands_all40.npz',candidate_Ha=a,QE_Ha=b,k_fractional=P['fractional_calculation'],distance_inverse_A=P['distance_inverse_A'])
 report=dict(complete=True,passed=True,material=mat,points=n,bands_compared=nb,bands_solved=40,
  physical_settings={k:d[k] for k in ['a_A','cell_A','charge_e','cutoff_Ha','grid','SCF_kmesh','T_Ha','xc','NLCC']},
  source_seed=d['source_seed'],source_state=d['source_state'],source_files_sha256=d['source_files_sha256'],
  one_global_shift_Ha=shift,global_shift_Ha=shift,global_shift_eV=shift*EV,
  alignment='One mean shift per system across all path points/requested bands; no subgroup fit',
  all_bands=statistics(aligned),maximum_Ritz_residual_Ha=residual,
  min_empty_buffer_Ha=min(c['buffer_above_highest_compared_Ha'] for c in checks),
  original_mesh_overlap_checks=overlaps,
  overlap_candidate_audit_max_eV=max(c['candidate_audit_max_eV'] for c in overlaps),
  overlap_QE_original_max_eV=max(c['QE_original_max_eV'] for c in overlaps),
  fixed_QE_density_hash_unchanged=True,Ritz_and_operator_checks=checks,path=P,
  scope='Fixed finite-T KS densities; common-G vs shifted-G+k finite cutoff spheres retained; no path SCF or parameters tuned')
 if mat in ['Al','Na']:
  mu=d['candidate_mu_Ha'];qmu=d['QE_mu_Ha'];charge=float(2/len(ae)*expit((mu-ae)/d['T_Ha']).sum());assert abs(charge-d['charge_e'])<1e-8
  report['chemical_potentials']=dict(candidate_Ha=mu,candidate_aligned_Ha=mu-shift,QE_SCF_Ha=qmu,
     candidate_SCF_FD_charge_e=charge,mu_aligned_minus_QE_eV=(mu-shift-qmu)*EV,
     source='Candidate FD root on complete original SCF audit grid; QE original SCF Fermi energy; neither recomputed on path')
  mask=abs((b[:,:nb]-qmu)*EV)<=1.0
  report['Fermi_window']=dict(criterion='Reference |epsilon_QE-mu_QE|<=1eV; same global alignment',**statistics(aligned[mask]))
  ea=(a[:,:nb]-shift-(mu-shift))*EV;eb=(b[:,:nb]-qmu)*EV
 else:
  nocc=d['charge_e']//2;report['valence']=statistics(aligned[:,:nocc]);report['conduction']=statistics(aligned[:,nocc:]);vbm=float(b[:,nocc-1].max());report['QE_path_VBM_Ha']=vbm;ea=(a[:,:nb]-shift-vbm)*EV;eb=(b[:,:nb]-vbm)*EV
 if mat=='Na':
  oldp=json.loads((I/'old_Na_path.json').read_text());op=np.array(oldp['fractional_calculation']);oldrows=list(csv.DictReader((I/'old_Na_band_comparison.csv').open()));old_a=np.zeros((len(op),nb));old_b=np.zeros_like(old_a)
  for row in oldrows:old_a[int(row['point']),int(row['band'])]=float(row['candidate_Ha']);old_b[int(row['point']),int(row['band'])]=float(row['QE_Ha'])
  oldchecks=[]
  for i,p in enumerate(np.array(P['fractional_calculation'])):
   distances=np.max(abs(op-p),axis=1);oi=int(np.argmin(distances))
   if distances[oi]<1e-9:
    err=float(np.max(abs(a[i,:nb]-old_a[oi,:nb])));qerr=float(np.max(abs(b[i,:nb]-old_b[oi,:nb])))
    assert err<=1e-8 and qerr<=1e-7,('Na FigureS1 overlap mismatch',i,oi,err,qerr)
    oldchecks.append(dict(new_point=i,old_point=oi,candidate_max_eV=err*EV,QE_max_eV=qerr*EV))
  assert oldchecks;report['Na_old_FigureS1_overlap']=dict(checks=oldchecks,candidate_max_eV=max(c['candidate_max_eV'] for c in oldchecks),QE_max_eV=max(c['QE_max_eV'] for c in oldchecks),source_files_sha256=d['old_Na_sources_sha256'])
 report['output_files_sha256']={str(p.relative_to(R)):sha(p) for p in [O/'band_comparison.csv',O/'bands_all40.npz']}
 (O/'QA.json').write_text(json.dumps(report,indent=2)+'\n');systems[mat]=report;arrays[mat]=dict(P=P,ea=ea,eb=eb,aligned=aligned)

# GaAs is retained byte-for-byte, including its original per-system QA.
for src,info in M['GaAs_reuse'].items():assert sha(src)==info['sha256'] and sha(R/info['dest'])==info['sha256']
g=json.loads((R/'processed/GaAs/QA.json').read_text());assert g['passed'];systems['GaAs']=g
with np.load(R/'processed/GaAs/bands_all40.npz') as z:a=z['candidate_Ha'];b=z['QE_Ha']
P=json.loads((R/'inputs/GaAs/path.json').read_text());P['continuous_branches']=[[0,56]];shift=g['global_shift_Ha'];vbm=g['QE_path_VBM_Ha'];arrays['GaAs']=dict(P=P,ea=(a[:,:17]-shift-vbm)*EV,eb=(b[:,:17]-vbm)*EV,aligned=(a[:,:17]-shift-b[:,:17])*EV)

#2x3 preview: lines vs hollow circles, disconnected cubic path stays disconnected.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.8,'legend.fontsize':7,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(7,5.0));fig.subplots_adjust(left=.085,right=.985,bottom=.09,top=.91,wspace=.43,hspace=.40)
order=M['systems_order'];panels=axes.ravel();blue='#0072B2';orange='#D55E00'
for i,mat in enumerate(order):
 ax=panels[i];d=arrays[mat];P=d['P'];x=np.array(P['distance_inverse_A']);nb=d['ea'].shape[1];branches=P['continuous_branches']
 for ib in range(nb):
  for begin,end in branches:
   xx=x[begin:end];marks=sorted(set(list(range(0,len(xx),2))+[len(xx)-1]))
   ax.plot(xx,d['ea'][begin:end,ib],color=blue,lw=.75,label='This work' if i==0 and ib==0 and begin==0 else None)
   ax.plot(xx,d['eb'][begin:end,ib],color=orange,marker='o',mfc='none',mew=.45,ms=1.7,ls='none',markevery=marks,label='QE' if i==0 and ib==0 and begin==0 else None)
 positions=[];labels=[]
 for tick,label in zip(P['ticks'],P['labels']):
  label=r'$\Gamma$' if label=='Gamma' else label
  if positions and abs(x[tick]-positions[-1])<1e-9:labels[-1]+='|'+label
  else:positions.append(x[tick]);labels.append(label)
 ax.set_xticks(positions,labels);ax.set_xlim(x[0],x[-1]);ax.set_title(f'({chr(97+i)}) {mat}',loc='left',fontweight='bold')
 ax.set_ylabel(r'$\epsilon-\mu$ (eV)' if mat in ['Al','Na'] else r'$\epsilon-\mathrm{VBM}_{\mathrm{QE}}$ (eV)')
 for xp in positions[1:-1]:ax.axvline(xp,color='#BBBBBB',lw=.45,zorder=0)
 ax.axhline(0,color='#888888',lw=.45,zorder=0)
 ax.text(.98,.965,f"MAE {systems[mat]['all_bands']['MAE_eV']*1000:.3g} meV",transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
 lo,hi=ax.get_ylim();ax.yaxis.set_major_locator(FixedLocator([v for v in ax.get_yticks() if lo<=v<=hi]))
 ax.spines[['top','right']].set_visible(False)
handles,labels=panels[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False,fontsize=7,handlelength=2)
ax=panels[5];palette=['#0072B2','#D55E00','#009E73','#CC79A7','#7B3294'];rng=np.random.default_rng(0);positive=[];floor=1e-12
for i,mat in enumerate(order):
 data=np.maximum(np.abs(arrays[mat]['aligned'].ravel()),floor);positive.extend(data.tolist());jitter=rng.uniform(-.16,.16,len(data));ax.scatter(i+jitter,data,s=1.6,c=palette[i],alpha=.4,rasterized=False,edgecolors='none')
low=10**np.floor(np.log10(min(positive)));high=10**np.ceil(np.log10(max(positive)));ax.set_yscale('log');ax.set_ylim(low/1.4,high*1.4);ticks=[10.**p for p in range(int(round(np.log10(low))),int(round(np.log10(high)))+1)];ax.set_yticks(ticks);ax.yaxis.set_minor_locator(NullLocator());ax.set_xticks(range(5),order);ax.set_xlim(-.45,4.45);ax.set_title('(f) Aligned absolute errors',loc='left',fontweight='bold');ax.set_ylabel(r'$|\Delta\epsilon|$ (eV)');ax.spines[['top','right']].set_visible(False)
fig.canvas.draw();canvas=fig.get_window_extent();outside=[]
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible() and t.get_text():
  bb=t.get_window_extent(fig.canvas.get_renderer())
  if bb.x0<canvas.x0-.5 or bb.x1>canvas.x1+.5 or bb.y0<canvas.y0-.5 or bb.y1>canvas.y1+.5:outside.append(t.get_text())
assert not outside,('Preview text outside canvas',outside)
for ext in ['pdf','png']:fig.savefig(R/'preview'/('bands_all_preview.'+ext),dpi=350,facecolor='white')
minfont=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text());assert minfont>=7
columns=['system','points','bands','MAE_eV','maximum_abs_eV','RMSE_eV','overlap_candidate_audit_max_eV','overlap_QE_original_max_eV','maximum_Ritz_residual_Ha','min_empty_buffer_Ha']
summary=[]
for mat in order:
 d=systems[mat]
 if mat=='GaAs':
  oc=d['Gamma_candidate_audit_max_abs_eV'];oq=d['Gamma_QE_original_max_abs_eV']
 else:oc=d['overlap_candidate_audit_max_eV'];oq=d['overlap_QE_original_max_eV']
 row=dict(d['all_bands'])
 row.update(system=mat,points=d['points'],bands=d['bands_compared'],overlap_candidate_audit_max_eV=oc,overlap_QE_original_max_eV=oq,maximum_Ritz_residual_Ha=d['maximum_Ritz_residual_Ha'],min_empty_buffer_Ha=d['min_empty_buffer_Ha'])
 summary.append(row)
with (R/'processed/summary.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(summary)
report=dict(complete=True,passed=True,systems=systems,summary=summary,GaAs_byte_preserving_reuse=M['GaAs_reuse'],
 preview_QA=dict(width_in=7,height_in=5,min_font_pt=minfont,titles_and_axis_labels_pt=8,ticks_pt=7,spines_pt=.8,out_of_canvas=outside,absolute_error_display_floor_eV=floor),
 environment={name:importlib.metadata.version(name) for name in ['jax','jaxlib','numpy','scipy','matplotlib']},
 outputs_sha256={str(p.relative_to(R)):sha(p) for p in [R/'processed/summary.csv',R/'preview/bands_all_preview.pdf',R/'preview/bands_all_preview.png']})
(R/'processed/QA.json').write_text(json.dumps(report,indent=2)+'\n')
lines=[f"{d['system']}: MAE={d['MAE_eV']:.8g} eV; max={d['maximum_abs_eV']:.8g} eV; overlap audit/QE={d['overlap_candidate_audit_max_eV']:.4g}/{d['overlap_QE_original_max_eV']:.4g} eV." for d in summary]
(R/'processed/terminal_report.txt').write_text('\n'.join(lines)+'\n')
with (R/'README.md').open('a') as f:f.write('\n## Completed\n'+'\n'.join(lines)+'\nFull per-system residual,emptybuffer,SCF and Na-old-path overlap checks,chemical potentials and hashes are in processed/QA.json.\n')
print('\n'.join(lines),flush=True)
