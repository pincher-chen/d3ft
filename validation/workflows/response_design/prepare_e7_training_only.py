"""CPU scaled Jacobian screen; never open SiC/GaP result directories.

Historical untrained Jacobians inform selection only, not E7 accepted results.
AlP's new NLCC row is conditional. Each LOO fold excludes its validation row
even when choosing the third parameter, so the selector itself is cross-fitted.
"""
from pathlib import Path
import hashlib,json,datetime,sys
import numpy as np
P=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(P/'validation/workflows/phase3'))
from equilibrium_jacobian import derive
A=P/'validation/results/response_design_20261015_e7_preparation'
NAMES=['theta_s','theta_p','theta_d','c:beta','c:gamma','x:kappa','x:mu']
SCALES=np.array([.1,.05,.05,.01,.005,.1,.02])
CORE=['Si','BN','diamond','AlAs'];FORBIDDEN={'SiC','GaP'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def select(data,materials):
 if FORBIDDEN.intersection(materials):raise ValueError('Holdout leakage')
 rows=[];labels=[]
 for mat in materials:
  d=data[mat];n=len(d['observables'])
  for obs in d['observables']:
   scale=(.005 if obs=='a0_A' else .05)*d['targets'][obs]
   rows.append(np.array(d['gradients'][obs])*SCALES/(scale*np.sqrt(n)))
   labels.append([mat,obs])
 j=np.asarray(rows);u,s,_=np.linalg.svd(j[:,[5,6]],full_matrices=False)
 if min(s)<=1e-12*max(s):raise ValueError('Kappa/mu screening block rank deficient')
 q=u[:,:2];candidates=[]
 for index in range(5):
  column=j[:,index];norm=float(np.linalg.norm(column));novelty=float(np.linalg.norm(column-q@(q.T@column)));fraction=novelty/max(norm,1e-30)
  block=j[:,[5,6,index]];norms=np.linalg.norm(block,axis=0);sigma=np.linalg.svd(block/np.maximum(norms,1e-30),compute_uv=False);cond=float(max(sigma)/max(min(sigma),1e-30))
  candidates.append(dict(name=NAMES[index],column_norm=norm,orthogonal_score=novelty,independent_fraction=fraction,normalized_condition=cond,
    eligible=norm>1e-8 and fraction>=.2 and cond<=1000,kind='KB pseudopotential multiplier' if index<3 else 'XC correlation constant'))
 eligible=[x for x in candidates if x['eligible']]
 if not eligible:raise ValueError('No independent third parameter')
 chosen=max(eligible,key=lambda x:(x['orthogonal_score'],x['column_norm']))
 return dict(materials=materials,observables=labels,scaled_jacobian=j.tolist(),candidates=candidates,chosen=chosen,
  parameter_names=['x:kappa','x:mu',chosen['name']],kappa_mu_singular_values=s.tolist(),
  rule='largest perpendicular scaled column norm; fraction>=0.2, normalized condition<=1000',
  weights='fixed 0.5% a and 5% B scales; equal weight per material; not experimental sigma',heldout_results_read=False)

def main():
 A.mkdir(exist_ok=True);manifest=json.loads((P/'validation/results/phase3_20260908/production_manifest_v3.json').read_text())
 targets={'Si':dict(a0_A=5.430941,B_GPa=(165.64+2*63.94)/3),'BN':dict(a0_A=3.6157,B_GPa=(798.4+2*172.4)/3),
  'diamond':dict(a0_A=3.566986,B_GPa=(1079+2*124)/3),'AlAs':dict(a0_A=5.66172,B_GPa=(119.9+2*57.5)/3),'AlP':dict(a0_A=5.4635)}
 data={};cost=[];opened=[]
 for mat in CORE:
  folder=Path(manifest[mat]);files=[folder/n for n in ['summary.json','metadata.json','points.json','mask.npy']];opened.extend(map(str,files))
  s=json.loads(files[0].read_text());m=json.loads(files[1].read_text());pts=json.loads(files[2].read_text())
  if s['parameter_vector']!=[0.]*7:raise ValueError('Untrained PBE center required')
  d=derive(folder);da=np.asarray(d['dlogV_equilibrium_dp'])*s['a0_A']/3
  data[mat]=dict(source=str(folder),source_hashes={str(f):sha(f) for f in files},observables=['a0_A','B_GPa'],targets=targets[mat],
   gradients=dict(a0_A=da.tolist(),B_GPa=d['dB_equilibrium_dp']),
   physical_settings={key:m['arguments'].get(key) for key in ['cutoff','grid','k','epochs','empty','xc_grid_factor','strain_step']},
   pp_hashes=m['pseudopotentials'],scope='Owned historical PBE screen only; not E7 state acceptance')
  selected=[r for r in pts if r['label'].startswith(('eq_','iso_'))];ngpu=len(str(m['gpu']).split(','));seconds=sum(r['seconds'] for r in selected)
  cost.append(dict(material=mat,points=len(selected),active_GPUs=ngpu,point_seconds_sum=seconds,GPU_hours=ngpu*seconds/3600,
   source=str(folder/'points.json'),scope='Historical a0/B subset lower bound; setup excluded'))
 result=dict(time=datetime.datetime.now().isoformat(),training_core=CORE,conditional_material='AlP',heldout=['SiC','GaP'],parameter_scales=SCALES.tolist(),
  core_only=select(data,CORE),training_data=data,historical_costs=cost,historical_core_Ct_lower_bound=sum(x['GPU_hours'] for x in cost),
  opened_material_result_files=opened,heldout_result_files_opened=[],AlP_physical_settings_frozen=False)
 f=P/'validation/results/response_design_20261014_e1_alp/acceptance_20261015/row_k10.json'
 if f.exists():
  row=json.loads(f.read_text());data['AlP']=dict(source=str(f),source_sha256=sha(f),targets=targets['AlP'],observables=['a0_A'],gradients={'a0_A':row['da0_dp']},
   physical_model='New NLCC Al3/P5 K10 preliminary; no substitution of old SG15 Al11/P5',accepted_for_training=False)
  result['conditional_five_material']=select(data,CORE+['AlP']);result['conditional_LOO_folds']=[]
  for validation in CORE+['AlP']:
   train=[x for x in CORE+['AlP'] if x!=validation]
   result['conditional_LOO_folds'].append(dict(fold_held_material=validation,fit_materials=train,selection=select(data,train),validation_used_for_selection=False))
 result['core_four_LOO_fallback']=[dict(fold_held_material=v,fit_materials=[x for x in CORE if x!=v],selection=select(data,[x for x in CORE if x!=v])) for v in CORE]
 if any('/'+v+'/' in x or '_'+v+'_' in x for v in FORBIDDEN for x in opened):raise ValueError('Holdout path accessed')
 (A/'training_only_selection.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print('E7第三参数（四核心筛选）：'+result['core_only']['chosen']['name']+'。')
 print('原设置单轮Ct下界：'+str(result['historical_core_Ct_lower_bound'])+' GPUh，必须缩减。')
 print('SiC/GaP结果文件访问0；AlP仍候选。')
if __name__=='__main__':main()
