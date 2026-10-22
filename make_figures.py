#!/usr/bin/env python3
"""Render function-named packages into separate output directories; never run DFT."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--collection',choices=['response-design','ground-state','all'],default='response-design')
p.add_argument('--output-dir',type=Path,default=ROOT/'runs/figures')
p.add_argument('--band-python',default=sys.executable)
a=p.parse_args();output=a.output_dir.resolve()
if a.collection in ['response-design','all']:
    subprocess.run([sys.executable,str(ROOT/'scripts/response_figures.py'),
                    '--output-dir',str(output),'--band-python',a.band_python],check=True)
if a.collection in ['ground-state','all']:
    # Earlier scripts write beside their data: copy each package first to preserve inputs/renders.
    packages=json.loads((ROOT/'Figures/catalogue.json').read_text())['collections']
    for item in packages:
        if item['collection']!='ground-state':continue
        dest=output/item['directory'];shutil.copytree(ROOT/'Figures'/item['directory'],dest)
        subprocess.run([sys.executable,str(dest/item['script'])],check=True)
