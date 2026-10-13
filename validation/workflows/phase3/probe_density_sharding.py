import json
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from d3ft.parallel import kpool_mesh,kpool_sharding,plain_fft_patch
from d3ft.optim.stiefel import stiefel_coeff
from jrystal._src import pw
mesh=kpool_mesh(); nk=8; nb=2; g=8
mask=np.zeros((g,g,g),bool); mask[:2,:2,:2]=True
rng=np.random.default_rng(723)
x=rng.normal(size=(1,nk,8,nb))+1j*rng.normal(size=(1,nk,8,nb))
occ=np.ones((1,nk,nb))/nk
s=kpool_sharding(mesh,4); rep=NamedSharding(mesh,P())
x=jax.device_put(x,s); occ=jax.device_put(occ,rep)
with plain_fft_patch(k_pool=True):
 def dense(p):return jax.lax.with_sharding_constraint(stiefel_coeff(p,mask),kpool_sharding(mesh,6))
 def density(p,o):return pw.density_grid(dense(p),123.,o)
 eager=jax.block_until_ready(density(x,occ))
 compiled=jax.jit(density,in_shardings=(s,rep),out_shardings=rep).lower(x,occ).compile()
 fused=jax.block_until_ready(compiled(x,occ))
 np.testing.assert_allclose(eager,fused,atol=1e-12,rtol=1e-12)
 zeros=jnp.zeros((1,nk,nb,g,g,g),dtype=x.dtype)
 expanded=jax.block_until_ready(dense(x))
 result=dict(devices=[str(d) for d in jax.devices()],zero_sharding=str(zeros.sharding),expanded_sharding=str(expanded.sharding),expanded_local_shapes=[list(a.data.shape) for a in expanded.addressable_shards],max_density_error=float(np.max(np.abs(np.asarray(eager)-np.asarray(fused)))),compiled_memory=str(compiled.memory_analysis()),scope='Synthetic final-density path; not Ge OOM root-cause proof or physical acceptance')
 print(json.dumps(result,indent=2))
