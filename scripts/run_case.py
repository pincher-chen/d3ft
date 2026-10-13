#!/usr/bin/env python3
"""Replay the unchanged two-stage protocol with independent 8+3 audits."""
from pathlib import Path
import argparse,json,os,shutil,sys,subprocess
from runtime import ROOT,DEP,run,verify

def plan(case,seed):
 return dict(case=case,seed=seed,prefix_steps=1000,first_corrected_steps=3000,audits=[4000,8000,12000,16000,20000],temperature_initial_Ha=.1,temperature_Ha=.01,occupation_coordinate_temperature_Ha=.1,delta_Ha=.001,orbital_lr=.1,prefix_occupation_lr=1.,corrected_occupation_lr=10.,early_stop='first strict 8+3 pass',source_selection='Original PH prefix; corrected stable metric after annealing; NLCC kernel for Na only')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=['Al','Si','SiC','SiC_distorted','GaAs','Na']);p.add_argument('--seed',type=int);p.add_argument('--output',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 c=json.loads((ROOT/'examples/cases.json').read_text())[a.case];seed=c['seed'] if a.seed is None else a.seed
 for v in [c['prefix_source'],c['corrected_source']]:verify(v)
 protocol=plan(a.case,seed)
 if a.dry_run:print(json.dumps(dict(protocol=protocol,physical=c),indent=2));return
 if a.output is None:p.error('--output is required (new directory)')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 shutil.copytree(ROOT/'reproducibility/inputs',out/'inputs');shutil.copytree(ROOT/'reproducibility/metadata',out/'metadata')
 # Reference registry is relocated without rewriting original data or fitted shifts.
 refs=json.loads((out/'inputs/QE_references.json').read_text())
 for k,d in refs.items():d['folder']=str(ROOT/'reproducibility'/d['folder'])
 (out/'inputs/QE_references.json').write_text(json.dumps(refs,indent=2)+'\n')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'results').mkdir()
 driver=ROOT/'reproducibility/scripts/timing_driver.py';audit=ROOT/'reproducibility/scripts/audit_low_spectrum_streamed.py'
 common=['--material',c['material'],'--orbital','postkp','--occupation','fd','--lr','.1','--lr-schedule','constant','--seed',str(seed),'--grid',str(c['grid']),'--empty-bands',str(c['empty_bands']),'--occupation-temperature','.1','--entropy-eps','1e-12','--metric-orbitals','1','--metric-occupations','1','--skip-warmup','--endpoint-check','--structure',str(out/'inputs/structures'/c['structure'])]
 def optimize(version,start,end,previous):
  tag=f'{a.case}_s{seed}_segment{start}_to{end}'
  cmd=common+['--epochs',str(end-start),'--tag',tag,'--occupation-lr','1' if start==0 else '10','--initial-temperature','.1' if start==0 else '.01','--temperature','.01','--anneal-steps','1000' if start==0 else '0']
  if previous:cmd+=['--restart-from',str(previous/'final_state.npz')]
  before=set((out/'results').iterdir());run(version,driver,cmd,out)
  added=set((out/'results').iterdir())-before
  if len(added)!=1:raise RuntimeError('Unexpected solver output count')
  return added.pop()
 previous=optimize(c['prefix_source'],0,1000,None);start=1000
 records=[]
 for end in protocol['audits']:
  state=optimize(c['corrected_source'],start,end,previous)
  before=set((out/'results').iterdir());run(c['corrected_source'],audit,['--state',state,'--bands',c['audit_bands'],'--tag',f'audit{end}'],out)
  added=set((out/'results').iterdir())-before
  if len(added)!=1:raise RuntimeError('Unexpected audit output count')
  audit_dir=added.pop()
  report=out/f'acceptance_{end}.json'
  run(c['corrected_source'],ROOT/'scripts/check_endpoint.py',[state,audit_dir,c['reference'],report],out)
  d=json.loads(report.read_text());records.append({'step':end,'passed':d['passed'],'report':str(report)})
  (out/'status.json').write_text(json.dumps({'status':'passed' if d['passed'] else ('censored' if end==20000 else 'running'),'audits':records},indent=2)+'\n')
  if d['passed']:break
  previous=state;start=end
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
