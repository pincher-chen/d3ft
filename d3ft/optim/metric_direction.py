"""Exploratory positive block metric; no universality or new-Hessian claim."""
import jax.numpy as jnp


def adjoint(x):
    return jnp.swapaxes(jnp.conj(x),-1,-2)


def positive_qr(x):
    q,r=jnp.linalg.qr(x,mode='reduced')
    diagonal=jnp.diagonal(r,axis1=-2,axis2=-1)
    phase=diagonal/jnp.maximum(abs(diagonal),1e-30)
    return q*phase[...,None,:]


def rotation_weight(f,xg):
    cap=2/f.shape[1]
    electron=f[...,None,:]+f[...,:,None]
    holes=cap-f
    hole=holes[...,None,:]+holes[...,:,None]
    # Positive, symmetric energy-gap component from diagonal AD contractions.
    # Avoids assembling a complete represented H; this is a diagnostic extension.
    # G=2HXf: for positive occupations, dividing by the actual f recovers
    # the Rayleigh energy even when f is far below1e-14. Clamping f here
    # suppresses the empty-orbital energy and underestimates rotation stiffness.
    # For exactly zero f no diagonal energy is identifiable from AD; retain
    # the old zero fallback and the positive metric floor, without inf/NaN.
    positive=f>0
    denominator=2*jnp.where(positive,f,1.0)
    levels=jnp.where(positive,jnp.real(jnp.diagonal(xg,axis1=-2,axis2=-1))/denominator,0.0)
    difference=levels[...,None,:]-levels[...,:,None]
    gap=jnp.sqrt(difference*difference+0.001**2)
    return jnp.maximum(electron*hole*gap/(2*cap),1e-14)


def direction(x,f,ad_orbital,ad_levels,kinetic,tau,normalize_orbitals,normalize_occupations):
    g=jnp.conj(ad_orbital)
    xg=adjoint(x)@g
    tangent=g-x@((xg+adjoint(xg))/2)
    a=adjoint(x)@tangent
    a=(a-adjoint(a))/2
    # Algebraically identical projection avoids subtracting two large
    # occupied-column mixtures in an almost empty orbital column.
    horizontal=g-x@xg
    # For the KS energy, g=2 H X f and X^*g=2 h f.
    # skew(X^*g)=h_ij(f_j-f_i). Evaluate it via the Hermitian
    # contraction and a bounded occupation ratio. Equal-occupation
    # gauge rotations vanish exactly, including the diagonal phases.
    pair_sum=f[...,None,:]+f[...,:,None]
    pair_difference=f[...,None,:]-f[...,:,None]
    ratio=pair_difference/jnp.where(pair_sum>0,pair_sum,1.0)
    a=((xg+adjoint(xg))/2)*ratio
    if normalize_orbitals:
        horizontal=horizontal/jnp.maximum(f[...,None,:],1e-14)
        # Smooth interior particle-hole symmetric positive rotation weight.
        # The numerical floor is retained at exactly empty/full boundaries.
        a=a/rotation_weight(f,xg)
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
