#!/usr/bin/env python3
"""Verify distributed response-code, case, seed and figure-data hashes."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'provenance/RESPONSE_DESIGN_MANIFEST.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
j=json.loads(manifest.read_text());errors=[]
for item in j['files']:
    p=ROOT/item['path']
    if not p.is_file() or sha(p)!=item['sha256']:errors.append(item['path'])
for i in range(2,6):
    d=ROOT/f'Figures/response_design/Figure_{i}/data'
    for row in json.loads((d/'SOURCES.json').read_text())['files']:
        if sha(d/row['path'])!=row['sha256']:errors.append(str(d/row['path']))
fig1=ROOT/'Figures/response_design/Figure_1'
if sorted(p.name for p in fig1.iterdir())!=['Figure_1.pdf','README.md']:errors.append('Figure1 must contain only its authoritative PDF and README')
if errors:
    print(json.dumps({'passed':False,'errors':errors},indent=2));raise SystemExit(1)
print(json.dumps({'passed':True,'verified_files':len(j['files']),'figure_data_hashes_verified':True,
                  'Figure1':'hand-edited PDF only','numerical_scope':'Integrity check; not a new DFT calculation'},indent=2))
