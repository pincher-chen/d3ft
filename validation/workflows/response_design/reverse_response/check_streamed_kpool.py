"""CPU values and all VJPs: streamed pool versus original eight-k program."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from streamed_kpool import streamed_density,streamed_kinetic
from d3ft.parallel.kpool import _local_occupied_density,_local_kinetic
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
if jax.default_backend()!='cpu':raise ValueError('CPU only')
rng=np.random.default_rng(20261015);shape=(6,6,6);nk=8;nb=8
c=jnp.asarray(rng.normal(size=(1,nk,nb,*shape))+1j*rng.normal(size=(1,nk,nb,*shape)))/np.sqrt(np.prod(shape))
o=jnp.asarray(rng.uniform(.1,.5,size=(1,nk,nb)));g=jnp.asarray(rng.normal(size=(*shape,3)));k=jnp.asarray(rng.normal(size=(nk,3)));v=jnp.asarray(137.);potential=jnp.asarray(rng.normal(size=shape))
def make(density,kinetic):
    def evaluate(c,v,g,k):
        rho=jax.vmap(density,in_axes=(0,None,0),axis_name='k')(c[None,...],v,o[None,...])[0]
        energy=jax.vmap(kinetic,in_axes=(None,0,0,0),axis_name='k')(g,k[None,...],c[None,...],o[None,...])[0]
        return jnp.sum(rho[0]*potential)+energy,rho
    return jax.jit(evaluate)
old,new=make(_local_occupied_density,_local_kinetic),make(streamed_density,streamed_kinetic)
(ov,orr),(nv,nrr)=old(c,v,g,k),new(c,v,g,k)
og=jax.grad(lambda *xs:old(*xs)[0],argnums=(0,1,2,3))(c,v,g,k)
ng=jax.grad(lambda *xs:new(*xs)[0],argnums=(0,1,2,3))(c,v,g,k)
errors={'scalar':float(abs(ov-nv)),'density':float(jnp.max(abs(orr-nrr)))}
for name,x,y in zip(('coeff_VJP','volume_VJP','reciprocal_grid_VJP','k_vector_VJP'),og,ng):errors[name]=float(jnp.max(abs(x-y)))
passed=all(e<1e-10 for e in errors.values())
(a.output/'acceptance.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),passed=passed,errors=errors,nk=nk,band_block=4,scope='CPU discrete energy/density and complex coefficient/volume/g/k VJP equivalence; actual Mg GPU memory separate'),indent=2)+'\n')
print(json.dumps({'passed':passed,'errors':errors}),flush=True)
if not passed:raise RuntimeError('Streamed k-pool equivalence failed')
