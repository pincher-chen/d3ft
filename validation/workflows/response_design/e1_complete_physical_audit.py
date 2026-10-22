"""Observe fresh force control and final QE XML; no new jobs submitted."""
from pathlib import Path
import json,time,datetime,xml.etree.ElementTree as ET,hashlib
import numpy as np
P=Path(__file__).resolve().parents[3];A=P/'validation/results/response_design_20261014_e1_alp/acceptance_20261015';root=A/'force_checks/AlP_NLCC_k10_force_validation'
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
while not (root/'summary.json').exists():
 if (root/'status.json').exists() and json.loads((root/'status.json').read_text()).get('stage')=='failed':break
 time.sleep(20)
force=dict(passed=False,status='缺力控制态或计算失败')
if (root/'summary.json').exists():
 point=json.loads((root/'summary.json').read_text());f=np.asarray(point['forces']);old=json.loads((P/'validation/results/response_design_20261014_e1_alp/runs/AlP_NLCC_k10/summary.json').read_text())
 checks=dict(force_maximum=bool(abs(f).max()<=1e-6),acoustic_sum_rule=bool(abs(f.sum(axis=0)).max()<=5e-8),electronic_stationarity=point['residual']<=1e-6,
  same_root_geometry=abs(point['a0_A']-old['a0_A'])<=1e-12,same_pressure_root=abs(point['pressure_GPa'])/(3*old['B_GPa'])<1e-5)
 force=dict(passed=all(checks.values()),checks=checks,forces_Ha_Bohr=f.tolist(),maximum_force_Ha_Bohr=float(abs(f).max()),ASR_Ha_Bohr=float(abs(f.sum(axis=0)).max()),source=str(root),summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest())
dump(A/'force_acceptance.json',force)
print('E1 力子步骤：'+('PASS' if force['passed'] else '失败/未验')+'。\n证据：acceptance_20261015/force_acceptance.json。\nk收敛FAIL仍保留，新NLCC行不采纳。',flush=True)
while not (A/'acceptance.json').exists():time.sleep(20)
report=json.loads((A/'acceptance.json').read_text());qes=[]
for k in [4,10]:
 xml=A/f'qe_k{k}/scratch/AlP_e1.save/data-file-schema.xml';record=dict(k=k,converged=False)
 if xml.exists():
  out=ET.parse(xml).getroot().find('output')
  if out is not None:
   text=out.findtext('convergence_info/scf_conv/convergence_achieved');record['convergence_xml_value']=text;record['converged']=text is not None and text.strip().lower()=='true'
 qes.append(record)
audit=dict(time=datetime.datetime.now().isoformat(),status='失败',passed=False,primary=report,force_control=force,QE_convergence=qes,
 scope='Complete additional physical checks; necessary k gate failed and independent seven-parameter re-equilibrium FD unverified. No old SG15 row or paper modified.')
dump(A/'physical_audit.json',audit)
with (P/'paper/response_design_EXPERIMENTS_20261014.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E1补充物理审计完成（formal FAIL）：新K10力控制'+('PASS' if force['passed'] else '未通过')+'，QE XML收敛标记另验；k门失败及七参数独立重平衡FD未验如实记录，旧SG15行不替换。证据E1/acceptance_20261015/physical_audit.json。\n')
with (P/'response_design_to_do_list.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' E1最终物理审计FAIL（k门），力/QE补充证据已保存；未采纳新NLCC行。\n')
print('E1 正式验收及补充审计：FAIL。\n证据：acceptance.json、physical_audit.json、force_acceptance.json。\n旧SG15、论文与图未替换，失败完整保留。',flush=True)
