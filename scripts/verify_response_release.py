#!/usr/bin/env python3
"""Verify the current branch inventory, functional figure index and source hashes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
j=json.loads((ROOT/'provenance/RESPONSE_DESIGN_MANIFEST.json').read_text());errors=[]
for item in j['files']:
    p=ROOT/item['path']
    if not p.is_file() or sha(p)!=item['sha256']:errors.append(item['path'])
packages=json.loads((ROOT/'Figures/catalogue.json').read_text())['collections']
actual={p.name for p in (ROOT/'Figures').iterdir() if p.is_dir()}
expected={p['directory'] for p in packages}
if actual!=expected:errors.append('Figure catalogue/directory mismatch')
data_count=0
for item in packages:
    folder=ROOT/'Figures'/item['directory']
    for name in ['README.md',item['pdf']]:
        if not (folder/name).is_file():errors.append(str(folder/name))
    if item['manual']:
        if {p.name for p in folder.iterdir()}!={'README.md',item['pdf']}:errors.append('Manual architecture must contain only PDF and README')
        continue
    if not (folder/item['script']).is_file():errors.append(str(folder/item['script']))
    sources=folder/'data/SOURCES.json'
    if not sources.exists():continue
    doc=json.loads(sources.read_text())
    rows=doc['files'] if isinstance(doc,dict) else doc
    for row in rows:
        if not isinstance(row,dict) or not row.get('sha256'):continue
        if 'file' in row:target=folder/row['file']
        elif 'path' in row:target=folder/'data'/row['path']
        else:continue
        if not target.is_file() or sha(target)!=row['sha256']:errors.append(str(target))
        data_count+=1
    local_manifest=folder/'MANIFEST.json'
    if local_manifest.exists():
        for name,digest in json.loads(local_manifest.read_text())['files'].items():
            if sha(folder/name)!=digest:errors.append(str(folder/name))
if errors:
    print(json.dumps({'passed':False,'errors':errors},indent=2));raise SystemExit(1)
print(json.dumps({'passed':True,'verified_files':len(j['files']),'figure_packages':len(packages),
                  'figure_data_hashes_verified':data_count,'architecture':'hand-edited PDF only',
                  'numerical_scope':'Integrity check; not a new DFT calculation'},indent=2))
