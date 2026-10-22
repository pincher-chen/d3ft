"""Real-inner-product projected PCG for complex orbital response.

The electronic operator must be positive on the physical horizontal subspace;
a nonpositive curvature or nonconverged residual is a hard failure.
"""
import time
import os
import jax
import jax.numpy as jnp
import numpy as np


def _elementwise_inner(a,b):
    # Avoid vdot's flatten/dot dispatch: JAX 0.6.2 may inspect an invalid
    # UnspecifiedValue layout for outputs of a sharded orbital operator.
    return jnp.real(jnp.sum(jnp.conj(a)*b))


_single_device_inner=jax.jit(_elementwise_inner)
inner_probe=None


def real_inner(a,b):
    if isinstance(a,jax.core.Tracer) or isinstance(b,jax.core.Tracer):
        return _elementwise_inner(a,b)
    if os.environ.get('D3FT_RESPONSE_SINGLE_DEVICE')=='1':
        # Only enabled by the one-visible-device streamed response entry.
        value=_single_device_inner(a,b)
        if inner_probe is not None:
            inner_probe(np.asarray(jax.device_get(a)),
                        np.asarray(jax.device_get(b)),value)
        return value
    # Do not inspect Array.format or Array.sharding, and do not feed the
    # malformed GPU-layout array to another compiled call. Host materialisation
    # is an intentional conservative bridge; the fresh buffers have ordinary
    # single-device layouts. This costs transfers and is not a performance claim.
    ah=np.asarray(jax.device_get(a));bh=np.asarray(jax.device_get(b))
    device=jax.local_devices()[0]
    aa=jax.device_put(ah,device);bb=jax.device_put(bh,device)
    value=_single_device_inner(aa,bb)
    if inner_probe is not None:inner_probe(ah,bh,value)
    return value


def real_norm(a):return jnp.sqrt(real_inner(a,a))

def pcg(action,b,project,precondition,relative_tolerance=1e-8,absolute_tolerance=1e-11,
        maxiter=200,callback=None):
    if not 0<relative_tolerance<=1e-6 or absolute_tolerance<=0 or maxiter<1:
        raise ValueError('Invalid response tolerance or iteration limit')
    x=jnp.zeros_like(b);r=project(b);norm0=float(jnp.sqrt(real_inner(r,r)))
    threshold=max(absolute_tolerance,relative_tolerance*norm0)
    if norm0<=threshold:return x,dict(converged=True,iterations=0,residual=norm0,relative_residual=0.)
    z=precondition(r);p=z;rz=float(real_inner(r,z));start=time.monotonic();history=[]
    for iteration in range(1,maxiter+1):
        ap=action(p);curvature=float(real_inner(p,ap))
        if not np.isfinite(curvature) or curvature<=0 or not np.isfinite(rz) or rz<=0:
            raise RuntimeError(f'Nonpositive projected curvature at iteration {iteration}: {curvature}; cannot certify response')
        alpha=rz/curvature;x=project(x+alpha*p);r=project(r-alpha*ap)
        norm=float(jnp.sqrt(real_inner(r,r)))
        record=dict(iteration=iteration,residual=norm,relative_residual=norm/norm0,seconds=time.monotonic()-start)
        history.append(record)
        if callback:callback(record)
        if not np.isfinite(norm):raise RuntimeError('Nonfinite projected residual')
        if norm<=threshold:
            # Recurrence residual alone can drift. Recompute the actual residual.
            true=float(real_norm(project(b-action(x))))
            if true<=threshold:
                return x,dict(converged=True,iterations=iteration,residual=true,
                              relative_residual=true/norm0,threshold=threshold,history=history)
            r=project(b-action(x));z=precondition(r);p=z;rz=float(real_inner(r,z));continue
        z=precondition(r);new=float(real_inner(r,z));p=project(z+(new/rz)*p);rz=new
    true=float(real_norm(project(b-action(x))))
    raise RuntimeError(f'Projected PCG failed after {maxiter} iterations; true residual {true}, required {threshold}')
