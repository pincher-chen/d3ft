"""Exploratory positive block metric; no universality or new-Hessian claim."""
import jax.numpy as jnp


def adjoint(x):
    return jnp.swapaxes(jnp.conj(x),-1,-2)


def positive_qr(x):
    q,r=jnp.linalg.qr(x,mode='reduced')
    diagonal=jnp.diagonal(r,axis1=-2,axis2=-1)
    phase=diagonal/jnp.maximum(abs(diagonal),1e-30)
    return q*phase[...,None,:]


def rotation_weight(f):
    cap=2/f.shape[1]
    electron=f[...,None,:]+f[...,:,None]
    holes=cap-f
    hole=holes[...,None,:]+holes[...,:,None]
    return jnp.maximum(electron*hole/(2*cap),1e-14)


def direction(x,f,ad_orbital,ad_levels,kinetic,tau,normalize_orbitals,normalize_occupations):
    g=jnp.conj(ad_orbital)
    xg=adjoint(x)@g
    tangent=g-x@((xg+adjoint(xg))/2)
    a=adjoint(x)@tangent
    a=(a-adjoint(a))/2
    horizontal=tangent-x@a
    if normalize_orbitals:
        horizontal=horizontal/jnp.maximum(f[...,None,:],1e-14)
        # Smooth interior particle-hole symmetric positive rotation weight.
        # The numerical floor is retained at exactly empty/full boundaries.
        a=a/rotation_weight(f)
    if kinetic is not None:horizontal=kinetic*horizontal
    horizontal=horizontal-x@(adjoint(x)@horizontal)
    dx=-(horizontal+x@a)
    dz=-ad_levels
    if normalize_occupations:
        cap=2/f.shape[1]
        weights=f*(1-f/cap)/tau
        dz=dz/jnp.maximum(weights,1e-14)
    # Removing a common level shift does not alter occupations or the directional derivative.
    dz=dz-jnp.mean(dz)
    slope=jnp.real(jnp.vdot(g,dx))+jnp.vdot(ad_levels,dz)
    return dx,dz,jnp.real(slope)
