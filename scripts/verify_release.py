#!/usr/bin/env python3
"""Check content hashes, solver identities, reference inputs and figure packages."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from runtime import verify
m=json.loads((ROOT/'provenance/RELEASE_MANIFEST.json').read_text());errors=[]
for rel,h in m['files'].items():
 p=ROOT/rel
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:errors.append(rel)
for name in ['current','paper_frozen','ph_prefix','ph_prefix_nlcc']:
 try:verify(name)
 except Exception as e:errors.append(str(e))
for line in (ROOT/'provenance/jrystal_dependency.sha256').read_text().splitlines():
 digest,rel=line.split('  ',1)
 p=ROOT/'third_party/jrystal_dependency'/rel
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append('Dependency changed: '+rel)
for folder in sorted((ROOT/'Figures').glob('Figure_*')):
 if not folder.is_dir():continue
 for item in ['README.md','data','script',folder.name+'.pdf',folder.name+'.png']:
  if not (folder/item).exists():errors.append(str(folder/item))
 if not list((folder/'script').glob('plot_figure_*.py')):errors.append(str(folder)+' missing plot script')
f=ROOT/'Figures/Figure_4'
for rel,digest in json.loads((f/'MANIFEST.json').read_text())['files'].items():
 if hashlib.sha256((f/rel).read_bytes()).hexdigest()!=digest:errors.append('Figure4 input changed: '+rel)
refs=json.loads((ROOT/'reproducibility/inputs/QE_references.json').read_text())
for case,ref in refs.items():
 for n,h in ref['files'].items():
  if hashlib.sha256((ROOT/'reproducibility'/ref['folder']/n).read_bytes()).hexdigest()!=h:errors.append(case+'/'+n)
for p in ROOT.rglob('*'):
 if '.git' in p.parts or not p.is_file():continue
 if p.stat().st_size>95_000_000:errors.append('Large Git file: '+str(p))
 if p.name.startswith('id_ed25519') or p.suffix in ['.key','.pem']:errors.append('Credential filename: '+str(p))
print(json.dumps(dict(files_checked=len(m['files']),errors=errors,passed=not errors),indent=2))
raise SystemExit(bool(errors))
