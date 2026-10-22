#!/usr/bin/env python3
"""Replot numeric figures; Figure1 is an authoritative hand-edited PDF."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'runs/response_figures')
p.add_argument('--band-python',default=sys.executable,help='Separate Python with the recorded Sumo/pymatgen plotting environment')
a=p.parse_args()
for i in range(2,6):
    subprocess.run([a.band_python if i==4 else sys.executable,str(ROOT/f'Figures/response_design/Figure_{i}/script/plot_figure_{i}.py'),
                    '--verify-data','--output-dir',str(a.output_dir.resolve()/f'Figure_{i}')],check=True)
print('Figure1 is not regenerated; use Figures/response_design/Figure_1/Figure_1.pdf')
