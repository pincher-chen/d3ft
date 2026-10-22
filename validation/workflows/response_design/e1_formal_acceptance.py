"""Native E1 gates plus fresh, same-UPF QE checks. Never overwrites old rows."""
from pathlib import Path
import json,sys,datetime,subprocess,hashlib,os,time,xml.etree.ElementTree as ET
import numpy as np
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261014_e1_alp';A=E/'acceptance_20261015'
A.mkdir(exist_ok=False)
sys.path.insert(0,str(P/'validation/workflows/phase3'))
from equilibrium_jacobian import derive
from check_elastic_convergence import compare
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def status(stage,**kw):dump(A/'status.json',dict(time=datetime.datetime.now().isoformat(),stage=stage,jobid=250285,**kw))
try:
 kcheck=compare(E/'runs/AlP_NLCC_k4',E/'runs/AlP_NLCC_k10','k');dump(A/'k_convergence.json',kcheck)
 limits=json.loads((P/'validation/results/response_design_20261014_e0_migration/inputs/AlP_NLCC/protocol.json').read_text())['acceptance']
 rows=[];native=[];jobs=[]
 for k in [4,10]:
  source=E/f'runs/AlP_NLCC_k{k}';meta=json.loads((source/'metadata.json').read_text());s=json.loads((source/'summary.json').read_text());points=json.loads((source/'points.json').read_text())
  eq=[p for p in points if p['label'].startswith('eq_')][-1];derived=derive(source);da=np.array(derived['dlogV_equilibrium_dp'])*s['a0_A']/3
  db=np.array(derived['dB_equilibrium_dp']);aold=np.array(s['da0_dp']);bold=np.array(s['dB_equilibrium_dp'])
  checks=dict(electronic_stationarity=max(p['residual'] for p in points)<=1e-6,
   pressure_root=abs(s['equilibrium_pressure_GPa'])/(3*s['B_GPa'])<1e-5,
   energy_pressure_B_agreement=abs(s['B_GPa']-s['B_stress_GPa'])<.1,
   strain_h_2h_B_stability=abs(s['B_h_GPa']-s['B_2h_GPa'])<.1,
   seven_parameters_finite=bool(da.shape==(7,) and db.shape==(7,) and np.isfinite(np.r_[da,db]).all()),
   a0_method2_vs_second_order=bool(np.all(abs(da-aold)<=.02*np.maximum(abs(da),abs(aold))+1e-7)),
   B_method2_vs_previous=bool(np.all(abs(db-bold)<=.02*np.maximum(abs(db),abs(bold))+1e-3)))
  native.append(dict(k=k,checks=checks,passed=all(checks.values()),pressure_root_normalized=abs(s['equilibrium_pressure_GPa'])/(3*s['B_GPa']),
   source=str(source),source_hashes={n:sha(source/n) for n in ['metadata.json','summary.json','points.json','mask.npy']},derived=derived))
  row=dict(k=k,a0_A=s['a0_A'],B_GPa=s['B_GPa'],da0_dp=da.tolist(),dB_equilibrium_dp=db.tolist(),
   parameter_order=json.loads((E/'protocol.json').read_text())['parameter_order'],method='equilibrium_jacobian method_version=2',
   independently_reoptimized_parameter_FD=False,accepted_for_calibration=False)
  rows.append(row);dump(A/f'row_k{k}.json',row)
  # Parse the final OWN xyz using the same historical JR Bohr convention.
  lines=(source/(eq['label']+'.xyz')).read_text().splitlines();import re
  cell=np.fromstring(re.search('Lattice="([^"]+)"',lines[1]).group(1),sep=' ').reshape(3,3)*1.8897259886
  positions=np.array([[float(x) for x in line.split()[1:4]] for line in lines[2:4]])*1.8897259886
  density=np.load(source/eq['density_file']);charge=float(density.mean()*abs(np.linalg.det(cell)))
  if abs(charge-8)>8e-8:raise RuntimeError('Actual-cell density charge failed')
  for pp,h in meta['pseudopotentials'].items():
   if sha(pp)!=h:raise RuntimeError('Changed PP '+pp)
  root=A/f'qe_k{k}';root.mkdir();(root/'scratch').mkdir()
  pps=list(meta['pseudopotentials']);ppdir=Path(pps[0]).parent
  # Match EXACT representatives and weights, without symmetry reduction/refolding.
  axes=np.arange(k)/k;axes=np.where(axes>.5,axes-1,axes)
  frac=np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),axis=-1).reshape(-1,3)
  inp=f"""&CONTROL
 calculation='scf', prefix='AlP_e1', pseudo_dir='{ppdir}', outdir='{root/'scratch'}', tprnfor=.true., tstress=.true.
/
&SYSTEM
 ibrav=0, nat=2, ntyp=2, ecutwfc=100, ecutrho=400, occupations='fixed', input_dft='PBE', nr1=64, nr2=64, nr3=64, nosym=.true., noinv=.true.
/
&ELECTRONS
 conv_thr=1.0d-12, electron_maxstep=200, mixing_beta=0.3, diagonalization='david'
/
ATOMIC_SPECIES
Al 26.9815385 Al.upf
P 30.973762 P.upf
CELL_PARAMETERS bohr
"""+'\n'.join(' '.join(f'{x:.16f}' for x in r) for r in cell)+'\nATOMIC_POSITIONS bohr\n'+'\n'.join(sym+' '+' '.join(f'{x:.16f}' for x in r) for sym,r in zip(['Al','P'],positions))+f'\nK_POINTS crystal\n{len(frac)}\n'+'\n'.join(' '.join(f'{x:.16f}' for x in r)+f' {1/len(frac):.16f}' for r in frac)+'\n'
  (root/'scf.in').write_text(inp)
  executable='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_box/qe-7.5/bin_cpu_intel/bin/pw.x'
  wrapper=root/'run.sh';wrapper.write_text('#!/bin/bash\nsource /APP/u22/ai_x86/intel/oneapi2023.2/setvars.sh --force > /dev/null 2>&1\nexport OMP_NUM_THREADS=16 MKL_NUM_THREADS=16\nexec timeout 10800 '+executable+' -in '+str(root/'scf.in')+'\n')
  cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=16','bash',str(wrapper)]
  rec=dict(jobid=250285,gpus=[],k=k,command=cmd,start=datetime.datetime.now().isoformat(),status='running',log=str(root/'scf.log'),executable_sha256=sha(executable),pp=meta['pseudopotentials'],cell_bohr=cell.tolist(),positions_bohr=positions.tolist(),k_fractional=frac.tolist(),exact_common_miller_basis=False,
   basis_difference='D3FT fixed |G| sphere; standard QE |G+k| sphere. Same cutoff is not an identical basis. Do not fit an energy shift.',charge_actual_cell=charge)
  dump(root/'launch.json',rec);f=(root/'scf.log').open('xb');proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;dump(root/'launch.json',rec);jobs.append((proc,f,rec,eq,s))
 dump(A/'native_checks.json',native);dump(A/'response_rows.json',rows)
 status('QE_comparisons_running',native_passed=all(n['passed'] for n in native),k_convergence_passed=kcheck['passed'],GPU_hours_QE=0.)
 comparisons=[]
 while any(p.poll() is None for p,*_ in jobs):time.sleep(10)
 for proc,f,rec,eq,s in jobs:
  f.close();k=rec['k'];root=A/f'qe_k{k}';rec.update(end=datetime.datetime.now().isoformat(),returncode=proc.returncode,status='completed' if proc.returncode==0 else 'failed');dump(root/'launch.json',rec)
  xml=root/'scratch/AlP_e1.save/data-file-schema.xml';r=dict(k=k,passed=False)
  if proc.returncode==0 and xml.exists():
   output=ET.parse(xml).getroot().find('output');energy=float(output.findtext('total_energy/etot'));stress=np.fromstring(output.findtext('stress'),sep=' ').reshape(3,3);forces=np.fromstring(output.findtext('forces'),sep=' ').reshape(2,3)
   pressure=float(np.trace(stress)/3*29421.02648438959)
   errors=dict(energy_Ha_cell=abs(energy-eq['energy_Ha']),stress_Ha_Bohr3=float(np.max(abs(stress+np.asarray(eq['stress'])))))
   checks={key:value<=limits[key] for key,value in errors.items()};checks['QE_at_D3FT_pressure_root']=abs(pressure)/(3*s['B_GPa'])<1e-5
   r.update(energy_Ha=energy,pressure_GPa=pressure,forces_Ha_Bohr=forces.tolist(),stress_Ha_Bohr3=stress.tolist(),errors=errors,checks=checks,passed=all(checks.values()),
    QE_lattice_root_independently_optimized=False,linearized_QE_lattice_shift_estimate_A=s['a0_A']*pressure/(3*s['B_GPa']),xml_sha256=sha(xml),unadjusted_comparison=True)
  else:r['failure']='QE task failed or full precision XML missing'
  comparisons.append(r);dump(root/'comparison.json',r)
 parameter_FD=False # No stored orbital checkpoints for seven independent theta +/- re-equilibrations.
 passed=all(n['passed'] for n in native) and kcheck['passed'] and all(r['passed'] for r in comparisons) and parameter_FD
 report=dict(time=datetime.datetime.now().isoformat(),status='完成' if passed else '失败',passed=passed,
  native=native,k_convergence=kcheck,QE=comparisons,independent_seven_parameter_reoptimized_FD=parameter_FD,
  blockers=[] if passed else ['未满足全部门；见逐项检查。K4→K10变化不证明K10本身错误，但无法提供既定收敛证书。','七参数行已独立四阶重组/数值稳定检查，未完成七参数重平衡FD，不能声称该项PASS。'],
  response_rows=rows,old_SG15_rows_replaced=False,GPU_hours_before_QE=json.loads((E/'status.json').read_text()).get('total_GPU_hours'),
  scope='New NLCC Al3/P5 control only. Failed certification retained, not mixed with SG15 Al11/P5. No energy shift, no fitted correction.')
 dump(A/'acceptance.json',report);status('formal_acceptance_finished',passed=passed,decision=report['status'])
 file=P/'paper/response_design_EXPERIMENTS_20261014.md';text=file.read_text();text=text.replace('**进行**：新AlP NLCC K4→K10四卡顺序队列，旧SG15行单独保留','**完成**：新AlP NLCC E1验收通过，旧SG15行保留' if passed else '**失败（正式验收）**：新NLCC K4/K10已算，k收敛/独立参数FD等门未全部通过；新行保留但不采纳，旧SG15行保留');file.write_text(text)
 with file.open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E1正式判定 '+report['status']+'，证据 validation/results/response_design_20261014_e1_alp/acceptance_20261015/acceptance.json；禁止用新NLCC行替换旧SG15，失败如实保留。\n')
 with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E1正式验收：'+report['status']+'；新NLCC行未采纳，旧SG15未改。\n')
 print('E1 正式验收：'+report['status']+'。\n证据：acceptance_20261015/acceptance.json；QE、k收敛及响应行逐项记录。\n旧 SG15 未替换，失败记录保留。',flush=True)
except Exception as exc:status('failed',error=str(exc));raise
