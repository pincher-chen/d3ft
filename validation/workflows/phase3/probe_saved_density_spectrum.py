"""Small CPU end-to-end saved-density spectrum check, not a material benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from ml_collections import ConfigDict
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch
from residual_report import fused_report

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
meta=json.loads((a.source/'metadata.json').read_text())
cfg=ConfigDict(meta['config']);cfg.epoch=3000;cfg.convergence_condition=0.
cfg.grid_sizes=16;cfg.cutoff_energy=5;cfg.k_grid_sizes=1;cfg.d3ft.parallel.k_pool=False
cfg.d3ft.xc_grid_factor=2
cfg.crystal_file_path_path=str((a.source/'sample.xyz').resolve())
with plain_fft_patch():
    out=calc(cfg)
    report=fused_report(out)
    residual=float(report['variational_residual'])
    if residual>1e-6:raise ValueError(f'CPU probe is not stationary: {residual}')
    np.save(a.out/'mask.npy',out.freq_mask)
    np.save(a.out/'sample_density.npy',np.asarray(out.density_grid))
    shutil.copyfile(a.source/'sample.xyz',a.out/'sample.xyz')
    meta['config']=cfg.to_dict()
    meta['scope']='Small CPU integration probe; not a converged material result'
    (a.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    point=dict(label='sample',residual=residual,density_file='sample_density.npy',
        density_sha256=hashlib.sha256((a.out/'sample_density.npy').read_bytes()).hexdigest())
    (a.out/'points.json').write_text(json.dumps([point],indent=2))
    print('SAVED stationary probe',residual,flush=True)
del out
jax.clear_caches()
subprocess.run([sys.executable,str(Path(__file__).with_name('check_occupations.py')),
    str(a.out),'--out',str(a.out/'spectrum')],check=True)
