import argparse,pathlib,json,pickle,numpy as np,jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from d3ft.calc.ground_state import calc
from d3ft.calc.xc_tunable import xc_energy_tunable
from d3ft import pseudopotential as pp
from jrystal._src import energy
from jrystal._src.grid import g_vectors
from jrystal.calc.opt_utils import get_ewald_coulomb_repulsion
from radial_quadrature import local_simpson_patch
from d3ft.parallel import plain_fft_patch
ap=argparse.ArgumentParser();ap.add_argument('source',type=pathlib.Path);ap.add_argument('output',type=pathlib.Path);a=ap.parse_args();m=json.loads((a.source/'metadata.json').read_text());c=ConfigDict(m['config']);c.epoch=1;c.optimizer_args.learning_rate=0.;c.d3ft.optim.lr_schedule='constant';c.crystal_file_path_path=str(a.source/'input.xyz')
with (a.source/'state.pkl').open('rb') as f:st=pickle.load(f)
with local_simpson_patch(),plain_fft_patch(k_pool=False):
 out=calc(c,init_params=st,freq_mask=np.load(a.source/'mask.npy'));g=g_vectors(out.crystal.cell_vectors,np.asarray(out.freq_mask.shape));v=out.crystal.vol;k=jnp.asarray(out.k_frac)@jnp.asarray(out.crystal.reciprocal_vectors);coef=out.make_coeff(out.params_pw);occ=out.get_occupation(out.params_occ,out.final_eps);rho=out.density_grid;rec=jnp.fft.fftn(rho,axes=(-3,-2,-1));vl=pp.potential_local_reciprocal(out.crystal.positions,g,out.pseudopot.r_grid,out.pseudopot.local_potential_grid,out.pseudopot.local_potential_charge,v,kmax=out.grid_kmax);gk=pp.gk_vectors(g,k,out.freq_mask);b=pp.beta_grid(out.pseudopot.r_grid,out.pseudopot.nonlocal_beta_grid,out.pseudopot.nonlocal_angular_momentum,gk,out.beta_kmax,tables=out.beta_tables);pn=pp.projectors_reciprocal(out.crystal.positions,gk,out.pseudopot.nonlocal_angular_momentum,b)
 values=dict(kinetic=energy.kinetic(g,k,coef,occ),hartree=energy.hartree(rec,g,v),local=pp.energy_local(rec,vl,v),nonlocal_=pp.energy_nonlocal(coef,pn,v,occ,d_matrix=out.nl_d_matrix,freq_mask=out.freq_mask),xc=xc_energy_tunable(rho,g,v,c.xc,core_density_grid=out.core_density_grid,xc_params=c.d3ft.xc_params),ewald=get_ewald_coulomb_repulsion(c,charges=np.asarray(out.pseudopot.valence_charges,float)))
 values={k:float(x) for k,x in values.items()};values['total']=sum(values.values());values['core_charge']=None if out.core_density_grid is None else float(out.core_density_grid.mean()*v);values['core_density_min']=None if out.core_density_grid is None else float(out.core_density_grid.min());a.output.write_text(json.dumps(values,indent=2)+'\n');print(json.dumps(values),flush=True)
