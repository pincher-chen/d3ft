"""Figure4:five-system posterior bands vs QE; CPU plotting from archived data only.
Run from anywhere:python Figures/Figure_4/script/plot_figure_4.py
No source simulation directories,GPU,or Slurm are accessed.
"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullLocator
HERE=Path(__file__).resolve().parent.parent
DATA=HERE/'data'
EV=27.211386245988
sources=json.loads((DATA/'SOURCES.json').read_text())
for entry in sources:
 path=HERE/entry['file']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],path
QA=json.loads((DATA/'QA.json').read_text());assert QA['passed'] and QA['complete']
systems=QA['systems'];M=json.loads((DATA/'PROTOCOL.json').read_text());arrays={}
for material in M['systems_order']:
 P=json.loads((DATA/material/'path.json').read_text());P.setdefault('continuous_branches',[[0,P['points']]])
 rows=list(csv.DictReader((DATA/material/'band_comparison.csv').open()))
 n=P['points'];nb=systems[material]['bands_compared'];assert len(rows)==n*nb
 a=np.empty((n,nb));b=np.empty_like(a);delta=np.empty_like(a);seen=set()
 for row in rows:
  i,j=int(row['point']),int(row['band']);assert (i,j) not in seen;seen.add((i,j))
  a[i,j]=float(row['candidate_Ha']);b[i,j]=float(row['QE_Ha']);delta[i,j]=float(row['aligned_delta_eV'])
  assert abs(float(row['distance_inverse_A'])-P['distance_inverse_A'][i])<1e-12
 shift=systems[material]['global_shift_Ha']
 assert np.max(abs((a-b-shift)*EV-delta))<1e-11
 assert abs(float(np.mean(abs(delta)))-systems[material]['all_bands']['MAE_eV'])<1e-12
 if material in ['Al','Na']:
  c=systems[material]['chemical_potentials'];ea=(a-shift-c['candidate_aligned_Ha'])*EV;eb=(b-c['QE_SCF_Ha'])*EV
 else:
  vbm=systems[material]['QE_path_VBM_Ha'];ea=(a-shift-vbm)*EV;eb=(b-vbm)*EV
 arrays[material]=dict(P=P,ea=ea,eb=eb,aligned=delta)
#2x3 preview: lines vs hollow circles, disconnected cubic path stays disconnected.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','Arial','Helvetica','DejaVu Sans'],'mathtext.fontset':'dejavusans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.8,'legend.fontsize':7,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'xtick.major.width':.8,'ytick.major.width':.8})
fig,axes=plt.subplots(2,3,figsize=(7,5.0));fig.subplots_adjust(left=.085,right=.985,bottom=.09,top=.91,wspace=.43,hspace=.40)
order=M['systems_order'];panels=axes.ravel();blue='#3b6fb6';orange='#c84f3d'
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
 ax.set_xticks(positions,labels);ax.set_xlim(x[0],x[-1])
 mae_ueV=systems[mat]['all_bands']['MAE_eV']*1e6
 ax.set_title(f'({chr(97+i)}) {mat}, MAE {mae_ueV:.3g} $\\mu$eV',loc='left',fontweight='bold')
 ax.set_ylabel(r'$\epsilon-\mu$ (eV)' if mat in ['Al','Na'] else r'$\epsilon-\mathrm{VBM}_{\mathrm{QE}}$ (eV)')
 for xp in positions[1:-1]:ax.axvline(xp,color='#BBBBBB',lw=.45,zorder=0)
 ax.axhline(0,color='#888888',lw=.45,zorder=0)
 lo,hi=ax.get_ylim();ax.yaxis.set_major_locator(FixedLocator([v for v in ax.get_yticks() if lo<=v<=hi]))

handles,labels=panels[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False,fontsize=7,handlelength=2)
ax=panels[5];palette=['#0072B2','#D55E00','#009E73','#CC79A7','#7B3294'];rng=np.random.default_rng(0);positive=[];floor=1e-12
for i,mat in enumerate(order):
 data=np.maximum(np.abs(arrays[mat]['aligned'].ravel()),floor);positive.extend(data.tolist());jitter=rng.uniform(-.16,.16,len(data));ax.scatter(i+jitter,data,s=1.6,c=palette[i],alpha=.4,rasterized=False,edgecolors='none')
low=10**np.floor(np.log10(min(positive)));high=10**np.ceil(np.log10(max(positive)));ax.set_yscale('log');ax.set_ylim(low/1.4,high*1.4);ticks=[10.**p for p in range(int(round(np.log10(low))),int(round(np.log10(high)))+1)];ax.set_yticks(ticks);ax.yaxis.set_minor_locator(NullLocator());ax.set_xticks(range(5),order);ax.set_xlim(-.45,4.45);ax.set_title('(f) Aligned absolute errors',loc='left',fontweight='bold');ax.set_ylabel(r'$|\Delta\epsilon|$ (eV)')
fig.canvas.draw();canvas=fig.get_window_extent();outside=[]
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible() and t.get_text():
  bb=t.get_window_extent(fig.canvas.get_renderer())
  if bb.x0<canvas.x0-.5 or bb.x1>canvas.x1+.5 or bb.y0<canvas.y0-.5 or bb.y1>canvas.y1+.5:outside.append(t.get_text())
assert not outside,('Preview text outside canvas',outside)
for ext in ['pdf','png']:fig.savefig(HERE/('Figure_4.'+ext),dpi=350,facecolor='white')
minfont=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text());assert minfont>=7

figureqa=dict(figure=4,source_checks_passed=True,width_in=7,height_in=5,
 min_font_pt=minfont,titles_and_axis_labels_pt=8,ticks_pt=7,axis_line_width_pt=.8,
 out_of_canvas=outside,absolute_error_display_floor_eV=floor,
 files={e:hashlib.sha256((HERE/('Figure_4.'+e)).read_bytes()).hexdigest() for e in ['pdf','png']})
(HERE/'Figure_4_QA.json').write_text(json.dumps(figureqa,indent=2)+'\n')
plt.close(fig)
print('Figure4 rendered; archived source hashes,numeric parity and text bounds passed.')
