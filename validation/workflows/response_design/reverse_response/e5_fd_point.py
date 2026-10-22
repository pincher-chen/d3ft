"""Generic owned geometry continuation with explicit, frozen radial program."""
import argparse,json,pickle,sys,gc
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from ml_collections import ConfigDict
from gamma_response import sha,dump
from bounded_xc import memory_bounded_xc
from fixed_radial_program import fixed_radial_program
P=Path(__file__).resolve().parents[4];sys.path.insert(0,str(P/'validation/workflows/response_design'))
import generic_point
from radial_quadrature import local_simpson_patch
from d3ft.parallel import plain_fft_patch
ap=argparse.ArgumentParser();ap.add_argument('--radial-parent',required=True,type=Path);a,remaining=ap.parse_known_args()
source=a.radial_parent;meta=json.loads((source/'metadata.json').read_text())
if sha(source/'state.pkl')!=meta['state_sha256']:raise ValueError('Changed owned parent')
with (source/'state.pkl').open('rb') as f:state=pickle.load(f)
c=ConfigDict(meta['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.d3ft.parallel.k_pool=False;c.crystal_file_path_path=str(source/'input.xyz')
with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc():
 parent=generic_point.calc(c,init_params=state,freq_mask=np.load(source/'mask.npy'))
 if parent.core_density_grid is not None:raise ValueError('Non-NLCC pilot only')
 grid=float(parent.grid_kmax);bk=float(parent.beta_kmax);tables=parent.beta_tables
 del parent;jax.clear_caches();gc.collect()
 original_calc=generic_point.calc
 def calc_with_frozen_grid(*args,**kw):
  out=original_calc(*args,**kw);out.grid_kmax=grid;return out
 generic_point.calc=calc_with_frozen_grid
 with fixed_radial_program(grid,bk,tables):
  sys.argv=[str(generic_point.__file__),*remaining];generic_point.main()
 output=Path(remaining[remaining.index('--output')+1]);table_path=output/'frozen_beta_tables.pkl'
 with table_path.open('xb') as f:pickle.dump(tables,f,protocol=5)
 m=json.loads((output/'metadata.json').read_text());m['frozen_radial_program']=dict(parent=str(source),
  parent_metadata_sha256=sha(source/'metadata.json'),grid_kmax=grid,beta_kmax=bk,
  tables_file=table_path.name,tables_sha256=sha(table_path),wrapper_sha256=sha(__file__),
  scope='Same Miller/FFT and same radial nodes/KB tables as analytical KMeshResponse, geometry only changes')
 dump(output/'metadata.json',m)
 __import__('shutil').copyfile(__file__,output/'frozen_program_wrapper_snapshot.py')
