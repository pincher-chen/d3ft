"""Fuse the existing stationarity calculation to avoid eager full-grid tapes."""
import os
import jax
import jax.numpy as jnp
from functools import partial
from d3ft.calc.implicit import _kwargs_from
from d3ft.calc.stress import free_energy_at_strain, stress_tensor, pressure, variational_residual, ewald_translation_indices
from d3ft.optim.stiefel import project_tangent

def report(out,compare=False):
    kw=_kwargs_from(out)
    sigma=stress_tensor(**kw)
    kw['ewald_indices']=ewald_translation_indices(out.crystal.cell_vectors,out.config.ewald_args['ewald_cutoff'])
    pw=kw.pop('params_pw'); occ=kw.pop('params_occ')
    def residual(p,o):
        gp,go=jax.grad(lambda x,y:free_energy_at_strain(jnp.zeros((3,3)),x,y,**kw),argnums=(0,1))(p,o)
        gp=jnp.conj(gp)
        if out.config.d3ft.optim.enabled: gp=project_tangent(p,gp)
        rp=jnp.max(jnp.abs(gp))
        leaves=jax.tree_util.tree_leaves(go)
        ro=jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in leaves])) if leaves else jnp.asarray(0.)
        return rp,ro
    # The eager stress above has already validated these identical spline
    # inputs. interpax's Python boolean checks cannot run inside a JIT trace.
    # Scope the unchecked constructors to this process and restore immediately.
    from d3ft.pseudopotential import local, nlcc
    original_local,original_nlcc=local.CubicSpline,nlcc.CubicSpline
    try:
        local.CubicSpline=partial(original_local,check=False)
        nlcc.CubicSpline=partial(original_nlcc,check=False)
        rp,ro=map(float,jax.jit(residual)(pw,occ))
    finally:
        local.CubicSpline, nlcc.CubicSpline=original_local,original_nlcc
    result=dict(stress=sigma,pressure_GPa=pressure(sigma),variational_residual=max(rp,ro),residual_parts=dict(pw=rp,occ=ro))
    if compare:
        old=float(variational_residual(params_pw=pw,params_occ=occ,**kw))
        result['eager_residual']=old
        if abs(old-result['variational_residual'])>1e-10*max(1.,old):
            raise AssertionError('Fused/eager stationarity mismatch')
    return result


def fused_report(out,compare=False):
    """Share one reverse pass for stress and electronic stationarity.

    Validate spline inputs in an eager forward energy evaluation first. For a
    k-pool calculation with ``D3FT_FUSE_FINAL_DENSITY=1``, the ground-state
    calculation has already constructed the same radial splines with checking
    enabled. Repeating that forward eagerly would materialise the complete
    spin/k/band/FFT coefficient tensor on GPU 0 before its sharding constraint.
    In that mode the validation is therefore not repeated and the report is
    compiled with the state arrays' existing shardings.
    """
    import numpy as np
    kw=_kwargs_from(out)
    kw['ewald_indices']=ewald_translation_indices(out.crystal.cell_vectors,out.config.ewald_args['ewald_cutoff'])
    pw=kw.pop('params_pw');occ=kw.pop('params_occ')
    volume=float(abs(np.linalg.det(np.asarray(out.crystal.cell_vectors))))
    sharded_report=(os.environ.get('D3FT_FUSE_FINAL_DENSITY')=='1'
                    and hasattr(pw,'sharding') and hasattr(pw.sharding,'mesh'))

    if not sharded_report:
        jax.block_until_ready(free_energy_at_strain(
            jnp.zeros((3,3)),params_pw=pw,params_occ=occ,**kw))

    def calculate(p,o):
        gs,gp,go=jax.grad(lambda s,x,y:free_energy_at_strain(s,x,y,**kw),argnums=(0,1,2))(
            jnp.zeros((3,3)),p,o)
        gp=jnp.conj(gp)
        if out.config.d3ft.optim.enabled: gp=project_tangent(p,gp)
        leaves=jax.tree_util.tree_leaves(go)
        ro=jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in leaves])) if leaves else jnp.asarray(0.)
        return (gs+gs.T)/(2*volume),jnp.max(jnp.abs(gp)),ro

    from d3ft.pseudopotential import local,nlcc
    original_local,original_nlcc=local.CubicSpline,nlcc.CubicSpline
    try:
        local.CubicSpline=partial(original_local,check=False)
        nlcc.CubicSpline=partial(original_nlcc,check=False)
        if sharded_report:
            from jax.sharding import NamedSharding, PartitionSpec
            replicated=NamedSharding(pw.sharding.mesh,PartitionSpec())
            occ_layout=jax.tree_util.tree_map(
                lambda x:x.sharding if hasattr(x,'sharding') else replicated,occ)
            compiled=jax.jit(calculate,
                in_shardings=(pw.sharding,occ_layout),
                out_shardings=(replicated,replicated,replicated))
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE report_gradient_begin',flush=True)
            sigma,rp,ro=compiled(pw,occ)
        else:
            sigma,rp,ro=jax.jit(calculate)(pw,occ)
        rp,ro=float(rp),float(ro)
        if os.environ.get('D3FT_TRACE_STAGES')=='1':
            print('D3FT_STAGE report_gradient_complete',flush=True)
    finally:
        local.CubicSpline,nlcc.CubicSpline=original_local,original_nlcc
    result=dict(stress=sigma,pressure_GPa=pressure(sigma),variational_residual=max(rp,ro),
                residual_parts=dict(pw=rp,occ=ro))
    if compare:
        old=report(out)
        np.testing.assert_allclose(sigma,old['stress'],atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(result['variational_residual'],old['variational_residual'],atol=1e-12,rtol=1e-10)
        result['comparison']=dict(max_stress_difference=float(np.abs(np.asarray(sigma)-np.asarray(old['stress'])).max()),
                                  residual_difference=abs(result['variational_residual']-old['variational_residual']))
    return result
