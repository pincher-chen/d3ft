#!/usr/bin/env python3
"""Run relocated archived QE SCF inputs; no shell interpolation or scheduler action."""
from pathlib import Path
import argparse,re,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=['Al','Si','SiC','SiC_distorted','GaAs','Na']);p.add_argument('--pw',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--mpi',nargs=argparse.REMAINDER,default=[]);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
s=ROOT/'reproducibility/references'/a.case/'original_inputs/scf.in';text=s.read_text();out=a.output.resolve()
for key,value in [('pseudo_dir',str(ROOT/'reproducibility/inputs/pseudopotentials')),('outdir',str(out/'scratch'))]:
 pattern=rf"(?im)(\b{key}\s*=\s*)(['\"])[^'\"]*\2";text,n=re.subn(pattern,lambda m:m[1]+"'"+value+"'",text);assert n==1,(key,n)
if a.dry_run:print(text);raise SystemExit(0)
out.mkdir(parents=True,exist_ok=False);(out/'scratch').mkdir();(out/'scf.in').write_text(text)
cmd=[*a.mpi,str(a.pw.resolve()),'-in',str(out/'scf.in')]
(out/'command.json').write_text(json.dumps(cmd)+'\n')
with (out/'scf.out').open('w') as log:subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
