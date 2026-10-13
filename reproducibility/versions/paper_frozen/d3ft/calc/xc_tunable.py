"""The exchange-correlation energy with the functional's own parameters exposed.

``jax_xc`` carries libxc's parameter struct as a namedtuple of JAX arrays --
PBE exchange is ``params(kappa=0.804, mu=0.21951497, lambda_=0.0)`` and PBE
correlation is ``params(BB=1.0, beta=0.06672455, gamma=0.03109069)`` -- and
``jrystal._src.xc`` builds it internally from the functional's name. That makes
those numbers unreachable from outside, which is a pity: kappa and mu are the two
constants a fitted GGA would actually move, and they are already differentiable.

This module rebuilds the unpolarised GGA path with the struct passed in rather
than constructed inside, so ``jax.grad`` can reach the parameters. It is a
re-expression of :func:`jrystal._src.energy.xc_energy` composed with
:func:`jrystal._src.xc.xc_density`, restricted to the case this code runs in
(unpolarised, ``kohn_sham=False``, GGA or LDA); ``test_20_xc_tunable`` pins it
against the stock path, because a re-implementation that quietly disagrees with
the original would poison every energy computed through it.

Why bother, rather than fitting the nonlocal strength: a sensitivity screen is
cheap here. One reverse-mode pass returns the derivative with respect to every
parameter at once, so asking "which of these knobs actually moves C44" costs the
same as asking about one of them.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from jax_xc.utils import get_p
from jrystal._src.utils import safe_real
from jrystal._src.xc import XC_UNPOLARIZED, get_xc_functional, sigma_r_fn

__all__ = ["xc_energy_tunable", "default_xc_params", "PBE_KEYS"]

#: The PBE constants worth moving, per functional name.
PBE_KEYS = {
    "gga_x_pbe": ("kappa", "mu"),
    "gga_c_pbe": ("beta", "gamma"),
}


def default_xc_params(xc_type: str) -> Dict[str, Dict[str, float]]:
    """The stock parameter values, as a nested dict ready to be perturbed.

    Args:
        xc_type: the ``+``-joined functional string, e.g. ``"gga_x_pbe+gga_c_pbe"``.

    Returns:
        ``{functional_name: {param_name: value}}`` for the names in
        :data:`PBE_KEYS` that appear in ``xc_type``.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in xc_type.split("+"):
        keys = PBE_KEYS.get(name)
        if not keys:
            continue
        p = get_p(name, XC_UNPOLARIZED)
        out[name] = {k: float(getattr(p.params, k)) for k in keys}
    return out


def _exc_density(xc_type: str, rho, sigma, xc_params):
    """Energy density summed over the constituent functionals."""
    flat_rho = rho.reshape(1, -1)[0]
    flat_sigma = sigma.reshape(1, -1)[0]
    total = None
    for name in xc_type.split("+"):
        p = get_p(name, XC_UNPOLARIZED)
        override = (xc_params or {}).get(name)
        if override:
            p = p._replace(params=p.params._replace(**override))
        f = get_xc_functional(name, polarized=False)
        e = jax.vmap(f, (None, 0, 0), 0)(p, flat_rho, flat_sigma)
        total = e if total is None else total + e
    return total.reshape(rho.shape[1:])


def _refine_density_grid(density, g_vectors, factor):
    """Fourier-interpolate a band-limited density, retaining its mean.

    Split even-grid Nyquist coefficients between both signs along every axis.
    Only the density is enlarged; the k-point/band wavefunction FFT stays on
    its original grid. Reciprocal vectors retain their cell derivatives.
    """
    spectrum = jnp.fft.fftn(density, axes=(-3, -2, -1))
    original_shape = density.shape[-3:]
    for axis in range(spectrum.ndim - 3, spectrum.ndim):
        n = spectrum.shape[axis]
        m = n * factor
        take = lambda lo, hi: jax.lax.slice_in_dim(spectrum, lo, hi, axis=axis)
        split = (n + 1) // 2
        zero_shape = list(spectrum.shape)
        if n % 2 == 0:
            zero_shape[axis] = m - n - 1
            nyquist = take(n // 2, n // 2 + 1) / 2
            pieces = [take(0, split), nyquist, jnp.zeros(zero_shape, spectrum.dtype),
                      nyquist, take(split + 1, n)]
        else:
            zero_shape[axis] = m - n
            pieces = [take(0, split), jnp.zeros(zero_shape, spectrum.dtype), take(split, n)]
        spectrum = jnp.concatenate(pieces, axis=axis)
    refined = jnp.real(jnp.fft.ifftn(spectrum, axes=(-3, -2, -1))) * factor**3
    frequencies = [jnp.fft.fftfreq(n * factor) * (n * factor) for n in original_shape]
    reciprocal_steps = (g_vectors[1, 0, 0], g_vectors[0, 1, 0], g_vectors[0, 0, 1])
    fine_g = (frequencies[0][:, None, None, None] * reciprocal_steps[0]
              + frequencies[1][None, :, None, None] * reciprocal_steps[1]
              + frequencies[2][None, None, :, None] * reciprocal_steps[2])
    return refined, fine_g


def xc_energy_tunable(
    density_grid,
    g_vector_grid,
    vol,
    xc_type: str,
    core_density_grid=None,
    xc_params: Optional[Dict[str, Dict[str, Any]]] = None,
    grid_factor: int = 1,
):
    """``E_xc`` with an optional override of the functional's constants.

    Args:
        density_grid: real-space density, shape ``[spin, x, y, z]``; this code
            path assumes one spin channel.
        g_vector_grid: reciprocal lattice vectors on the grid.
        vol: cell volume.
        xc_type: ``+``-joined functional string. GGA only -- an LDA string has no
            gradient term and would need the other branch of the stock code.
        core_density_grid: NLCC pseudo-core density, added to the valence density
            before the functional sees it, exactly as the stock path does.
        xc_params: ``{functional: {param: value}}``. The values may be tracers,
            which is the point. ``None`` reproduces the stock energy.
        grid_factor: integer density-grid refinement for XC quadrature only.
            The wavefunction grid must already resolve the density. This
            optional path currently excludes NLCC, which needs its own direct
            fine-grid reconstruction. Factor one preserves the original path.

    Returns:
        The exchange-correlation energy, a scalar.

    Raises:
        ValueError: for a polarised density or a non-GGA functional, neither of
            which this re-expression covers.
    """
    if density_grid.shape[0] != 1:
        raise ValueError(
            "xc_energy_tunable covers the unpolarised path only; the stock "
            "jrystal implementation handles the spin-polarised scaling relations."
        )
    if "gga" not in xc_type:
        raise ValueError(f"xc_energy_tunable covers GGA only, got {xc_type!r}")
    if int(grid_factor) != grid_factor or grid_factor < 1:
        raise ValueError('XC grid_factor must be a positive integer')
    if grid_factor > 1:
        if core_density_grid is not None:
            raise ValueError('XC grid refinement requires direct fine-grid NLCC reconstruction; valence-only supported')
        if min(density_grid.shape[-3:]) < 3:
            raise ValueError('XC grid refinement requires at least three samples per axis')
        density_grid, g_vector_grid = _refine_density_grid(density_grid, g_vector_grid, int(grid_factor))

    num_grid = jnp.prod(jnp.array(density_grid.shape[-3:]))
    if core_density_grid is not None:
        rho_xc = density_grid + core_density_grid[None, ...] / density_grid.shape[0]
    else:
        rho_xc = density_grid

    sigma = sigma_r_fn(rho_xc, g_vector_grid)
    exc = _exc_density(xc_type, rho_xc, sigma, xc_params)
    return safe_real(jnp.sum(exc * rho_xc)) * vol / num_grid
