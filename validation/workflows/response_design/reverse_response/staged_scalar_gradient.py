"""Combine the unchanged non-XC pullback with staged, exact mixed XC AD."""
import gc
import time
import jax
from staged_xc_geometry import StagedXCGeometry


def staged_scalar_gradient(response,nonxc):
    nonxc_gradient=jax.jit(jax.grad(nonxc,argnums=0))

    def gradient(delta,weight_g,weight_r,lam):
        started=time.monotonic()
        print('RESPONSE_STAGE staged_geometry_nonxc',flush=True)
        ordinary=nonxc_gradient(delta,weight_g,weight_r,lam)
        jax.block_until_ready(ordinary)
        # Do not retain fine-grid fields while the electronic PCG kernel runs.
        jax.clear_caches();gc.collect()
        print('RESPONSE_STAGE staged_geometry_prepare_xc',flush=True)
        staged=StagedXCGeometry(response.rho,response.g,response.volume,
                                response.geometry_inverse0,response.out.config.xc,
                                response.xc_params,response.xc_factor)
        print('RESPONSE_STAGE staged_geometry_contract_xc',flush=True)
        xc=staged.gradient(delta,weight_r)
        result=ordinary+xc;jax.block_until_ready(result)
        del staged,xc,ordinary
        jax.clear_caches();gc.collect()
        print('RESPONSE_STAGE staged_geometry_complete seconds='+str(time.monotonic()-started),flush=True)
        return result

    return gradient
