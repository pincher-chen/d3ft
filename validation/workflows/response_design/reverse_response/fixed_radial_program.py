"""Owned frozen-table geometry program for E5 finite differences/readout."""
from contextlib import contextmanager
import pickle
from pathlib import Path
from gamma_response import sha
@contextmanager
def fixed_radial_program(grid_kmax,beta_kmax,tables):
 from d3ft import pseudopotential as pp
 oldlocal,oldk,oldt=pp.potential_local_reciprocal,pp.kmax_for_grid,pp.build_tables
 def local(*args,**kw):kw['kmax']=grid_kmax;return oldlocal(*args,**kw)
 pp.potential_local_reciprocal=local;pp.kmax_for_grid=lambda *args,**kw:beta_kmax
 pp.build_tables=lambda *args,**kw:tables
 try:yield
 finally:pp.potential_local_reciprocal=oldlocal;pp.kmax_for_grid=oldk;pp.build_tables=oldt
@contextmanager
def source_radial_program(meta,source):
 info=meta.get('frozen_radial_program')
 if info is None:yield None;return
 path=Path(source)/info['tables_file']
 if sha(path)!=info['tables_sha256']:raise ValueError('Frozen owned beta table changed')
 with path.open('rb') as f:tables=pickle.load(f)
 with fixed_radial_program(info['grid_kmax'],info['beta_kmax'],tables):yield info
