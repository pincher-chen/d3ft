#!/usr/bin/env python3
"""Replot response-design numerical panels using function-named packages."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,default=ROOT/'runs/response_figures')
p.add_argument('--band-python',default=sys.executable,help='Python with the recorded Sumo/pymatgen plotting environment')
a=p.parse_args()
packages=json.loads((ROOT/'Figures/catalogue.json').read_text())['collections']
for item in packages:
    if item['collection']!='response-design' or item['manual'] or item['directory']=='Figure_mgsin2_structures':continue
    folder=ROOT/'Figures'/item['directory']
    python=a.band_python if item['directory']=='Figure_bands_and_transfer' else sys.executable
    subprocess.run([python,str(folder/item['script']),'--verify-data',
                    '--output-dir',str(a.output_dir.resolve()/item['directory'])],check=True)
print('The hand-edited architecture PDF is not regenerated: Figures/Figure_response_architecture/Figure_response_architecture.pdf')
