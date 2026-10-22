"""CPU-only durable publication of audit outcome to experiment records."""
import argparse,datetime as dt,json,os,time
from pathlib import Path
import numpy as np
from e1_three_parameter_queue import read,dump

def main(T):
 c=read(T/'protocol.json');P=Path(c['project'])
 while True:
  s=read(T/'status.json') if (T/'status.json').exists() else {}
  if s.get('stage') in ['complete','failed']:break
  time.sleep(20)
 os.close(os.open(T/'publish_once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY));now=dt.datetime.now().astimezone().isoformat();passed=False;details={}
 if (T/'acceptance.json').exists():
  r=read(T/'acceptance.json');passed=r['passed'];new=r['new_readout'];force=np.asarray(new['forces_Ha_Bohr']);old=read(T/'preliminary_metrics.json');reference=old['trajectory'][0]['energy_Ha'];metrics=old['endpoint'].copy()
  metrics.update(gap_eV=new['gap_eV'],energy_Ha=new['energy_Ha'],energy_change_eV=(new['energy_Ha']-reference)*27.211386245988,energy_change_meV_atom=(new['energy_Ha']-reference)*27.211386245988*1000/len(force),max_single_atom_force_Ha_Bohr=float(np.linalg.norm(force,axis=1).max()),max_single_atom_force_eV_A=float(np.linalg.norm(force,axis=1).max()*51.42208619083232),max_force_component_Ha_Bohr=float(abs(force).max()),max_force_component_eV_A=float(abs(force).max()*51.42208619083232),residual_Ha=new['residual_Ha'],ASR_Ha_Bohr=new['ASR_Ha_Bohr'])
  details=dict(passed=passed,metrics=metrics,checks=r['physical_checks'],numerical=r['numerical_acceptance'],audit_GPU_hours=r['audit_GPU_hours'],structural_equilibrium_claim=False)
  dump(T/'independent_endpoint_metrics.json',details)
  text=f"E4独立端点审计 {'PASS' if passed else 'FAIL'}：gap={new['gap_eV']:.9f}eV，max单原子力={metrics['max_single_atom_force_eV_A']:.6f}eV/Å，ΔE={metrics['energy_change_meV_atom']:.6f}meV/atom；原目标/物理/谱门不变，非离子平衡；审计另用{r['audit_GPU_hours']:.6f}GPUh。"
 else:text='E4独立端点审计未完成/失败：'+s.get('error','unknown')+'；原三步达标轨迹保留，不能记独立PASS。'
 with (P/'paper/response_design_E4_ENDPOINT_AUDIT_20261016.md').open('a') as f:f.write('\n## '+now+' 独立审计结论\n\n'+text+'\n\n原始证书：`'+str(T/'acceptance.json')+'`，详细指标：`'+str(T/'independent_endpoint_metrics.json')+'`；未通过时检查原生/谱/重占据逐项门，不放宽。\n')
 for p in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md']:
  with p.open('a') as f:f.write('\n- '+now+' '+text+'\n')
 p=P/'paper/response_design_EXPERIMENTS_20261014.md';lines=p.read_text().splitlines()
 for j,line in enumerate(lines):
  if line.startswith('| E4 |'):
   cols=line.split('|');cols[-2]=' **Γ54原设计达标；独立端点审计'+('PASS' if passed else 'FAIL/未完成')+'**：'+text+' 原六应变对照详见response_design_E4_ENDPOINT_AUDIT_20261016.md ';lines[j]='|'.join(cols)
 p.write_text('\n'.join(lines)+'\n');dump(T/'publication_status.json',dict(time=now,completed=True,passed=passed,details=details,protected_manuscript_modified=False))
 print('E4独立审计 '+('PASS' if passed else 'FAIL/未完成')+'。\n物理指标与验收结论已落盘，非平衡结构。\n实验清单和todolist已更新，论文/图未改。',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
