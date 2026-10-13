"""Matrix-free Kohn-Sham Hamiltonian action for D3FT-Lanczos.

Two operators are provided:

* :func:`dense_hamiltonian_matvec` — assembles the explicit ``[band, band]``
  Hamiltonian (cost O(I^2) to build); used only for validation/timing baselines.

* :func:`make_ks_hvp` — the **matrix-free** Kohn-Sham Hamiltonian action in the
  plane-wave (G) space at one k-point. It never forms the Hamiltonian matrix; each
  application costs O(n_g log n_g) via FFTs (plus the nonlocal projector contraction),
  which is the basis for D3FT-Lanczos's O(k_iter * dim) band extraction.

The operator action is derived to match jrystal's matrix-element conventions
(matching :mod:`d3ft.pseudopotential`):

  (H_kin  phi)_G = T_G * phi_G                       (T = 1/2 |G+k|^2)
  (H_loc  phi)_G = FFT[ v_eff(r) * IFFT(phi) ]_G     (v_eff local in real space)
  (H_nl   phi)_G = (1/vol) sum_bp conj(beta_bp,G) <beta_bp | phi>

so that <c_i | H | c_j> reproduces ``hamiltonian_matrix[i, j]`` for orthonormal
band coefficients ``c``.
"""
from __future__ import annotations

from typing import Callable, Tuple

import os
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float


def _xc_potential_autodiff(rho_xc, g_vector_grid, xc_type, dim):
  """Exact KS XC potential v_xc(r) = d(sum_r exc(rho) rho)/d rho(r) via autodiff.

  ``rho_xc`` may be ``[x,y,z]`` or ``[spin,x,y,z]``; the returned potential has
  shape ``[x,y,z]`` (spin-restricted) matching the scalar v_eff used by H_op.
  Computed with ``jax.grad`` of the SAME energy-density functional the SCF uses
  (``jrystal._src.xc.xc_density(rho, ..., kohn_sham=False)`` returns exc), so the
  band-readout Hamiltonian is the exact gradient of the converged SCF energy.
  """
  from jrystal._src import xc as _xc

  rho_real = jnp.real(rho_xc)

  def exc_total(rho):
    exc = _xc.xc_density(rho, g_vector_grid, False, xc_type)
    return jnp.real(jnp.sum(exc * rho))

  v = jax.grad(exc_total)(rho_real)            # same shape as rho_real
  if v.ndim == dim + 1:                         # has spin axis -> reduce
    v = v[0]
  return v


def dense_hamiltonian_matvec(
    coeff_k: Complex[Array, "band x y z"],
    density_grid: Float[Array, "x y z"],
    potential_loc_rec: Float[Array, "x y z"],
    potential_nl_psi: Complex[Array, "beta phi x y z"],
    g_vector_grid: Float[Array, "x y z 3"],
    kpt: Float[Array, "3"],
    vol: float,
    xc: str,
    nl_d_matrix: Float[Array, "beta beta"] = None,
    freq_mask: Float[Array, "x y z"] = None,
) -> Tuple[Callable, Complex[Array, "band band"]]:
    """Return ``(H_op, H)``: the explicit ``[band, band]`` Hamiltonian.

    Assembled from D3FT's own pseudopotential terms, so this is a self-consistency
    check on :func:`make_ks_hvp` rather than a comparison against another package.
    It used to call ``jrystal.pseudopotential.normcons.hamiltonian_matrix``; that
    made the check compare two different conventions once D3FT fixed the
    Kleinman-Bylander conjugation, which a loose tolerance could hide.
    """
    from jrystal._src import braket, kinetic, pw
    from jrystal._src import potential as jr_potential

    from .. import pseudopotential as d3pp

    coeff = coeff_k[None, None, ...]
    dim = g_vector_grid.shape[-1]
    wave_grid = pw.wave_grid(coeff, vol)

    h_nl = d3pp.hamiltonian_nonlocal(
        coeff, potential_nl_psi[None, ...], vol, d_matrix=nl_d_matrix,
        freq_mask=freq_mask,
    )
    h_loc = d3pp.hamiltonian_local(wave_grid, potential_loc_rec, vol)

    rho_rec = jnp.fft.fftn(density_grid, axes=range(-dim, 0))
    v_har = jnp.fft.ifftn(
        jr_potential.hartree_reciprocal(rho_rec, g_vector_grid, kohn_sham=True),
        axes=range(-dim, 0),
    )
    v_xc = jr_potential.xc_density(
        density_grid, g_vector_grid, kohn_sham=True, xc_type=xc
    )
    h_s = braket.expectation(
        wave_grid, v_har + v_xc, vol, diagonal=False, mode="real"
    )
    h_kin = braket.expectation(
        coeff, kinetic.kinetic_operator(g_vector_grid, kpt[None, :]),
        vol, diagonal=False, mode="kinetic",
    )
    H = (h_nl + h_loc + h_s + h_kin)[0, 0]

    def H_op(v):
        return H @ v

    return H_op, H


def make_ks_hvp(
    density_grid: Float[Array, "x y z"],
    potential_loc_rec: Complex[Array, "x y z"],
    potential_nl_psi: Complex[Array, "beta phi *grid"],
    g_vector_grid: Float[Array, "x y z 3"],
    kpt: Float[Array, "3"],
    vol: float,
    freq_mask: Float[Array, "x y z"],
    xc: str = "lda_x",
    core_density_grid: Float[Array, "x y z"] = None,
    nl_d_matrix: Float[Array, "beta beta"] = None,
    nl_sign: Float[Array, "beta"] = None,
    potential_loc_real: Float[Array, "x y z"] = None,
) -> Tuple[Callable, int]:
    """Build a matrix-free KS Hamiltonian action on the masked G-subspace.

    Returns ``(H_op, n_g)`` where ``H_op`` maps a length-``n_g`` complex vector
    (coefficients on the active G-vectors) to ``H |phi>`` restricted to the same
    active set, and ``n_g`` is the number of active plane waves.

    ``nl_d_matrix`` is the block-diagonal Kleinman-Bylander coupling matrix
    (from :func:`d3ft.pseudopotential.concat_d_matrix`). When ``None``, falls back to the
    deprecated ``nl_sign`` path or a positive-definite |proj|^2 sum.
    """
    from jrystal._src import kinetic, potential

    mask = np.asarray(freq_mask).astype(bool)
    grid_shape = mask.shape
    flat_mask = mask.reshape(-1)
    idx = np.where(flat_mask)[0]
    n_g = int(idx.size)
    N = float(np.prod(grid_shape))
    dim = kpt.shape[-1]

    # Kinetic operator T_G on the full grid (real, [x,y,z]).
    T_grid = kinetic.kinetic_operator(g_vector_grid, kpt[None, :])[0]  # [x,y,z]
    T_flat = jnp.asarray(T_grid).reshape(-1)[idx]                       # [n_g]

    # Fixed local effective potential v_eff(r) (real space, [x,y,z]).
    rho_rec = jnp.fft.fftn(density_grid, axes=range(-dim, 0))
    v_har = potential.hartree_reciprocal(rho_rec, g_vector_grid, kohn_sham=True)
    v_har = jnp.fft.ifftn(v_har, axes=range(-dim, 0))
    # NLCC: the XC potential is evaluated at rho_valence + rho_core; the Hartree
    # (and all other) terms keep using the valence density only.
    if core_density_grid is not None:
        if density_grid.ndim == 4:
            n_spin = density_grid.shape[0]
            rho_xc = density_grid + core_density_grid[None, ...] / n_spin
        else:
            rho_xc = density_grid + core_density_grid
    else:
        rho_xc = density_grid
    # KS XC potential v_xc(r) = delta E_xc / delta rho(r). We compute it as the
    # AUTODIFF functional derivative of the exact XC energy functional that the
    # D3FT/jrystal SCF minimises (E_xc = (vol/N) sum_r exc(rho) * rho), rather
    # than jrystal's hand-coded ``vxc_gga``. The hand-coded GGA potential uses an
    # explicit FFT-Laplacian (IFFT[-|G|^2 rho_G]) plus a 1-D-FFT gradient helper,
    # which (a) is not the exact derivative of the SCF energy and (b) blows up on
    # the sharp NLCC pseudo-core charge, spuriously raising the s-like Gamma
    # conduction state by ~2 eV and pushing CBM/VBM off Gamma. The autodiff
    # derivative is exactly consistent with the SCF energy and stays smooth on the
    # sharp core, so the band extrema land at the correct k-points.
    v_xc = _xc_potential_autodiff(rho_xc, g_vector_grid, xc, dim)
    v_loc = (
        jnp.asarray(potential_loc_real)
        if potential_loc_real is not None
        else jnp.fft.ifftn(potential_loc_rec, axes=range(-dim, 0))
    )
    v_eff = (v_har + v_xc + v_loc)  # complex [x,y,z]; imaginary part ~0

    # Nonlocal projectors, as [beta, phi, <grid>] for this k-point. Two storage
    # forms reach here: the cutoff sphere, [beta, phi, n_g], which is what
    # d3ft.pseudopotential builds and what the SCF and the band readout use; and
    # the full FFT box, [beta, phi, x, y, z], which the QE parity probes and
    # jrystal's own normcons path still produce. Both end up as the same
    # [beta, phi, n_g] -- the box form simply has its inactive G-vectors dropped
    # here instead of never being computed.
    nl = jnp.asarray(potential_nl_psi)
    nb, nphi = nl.shape[0], nl.shape[1]
    if nl.ndim == 3:
        nl_flat = nl                                            # already masked
    else:
        nl_flat = nl.reshape(nb, nphi, -1)[:, :, jnp.asarray(idx)]
    if nl_flat.shape[-1] != n_g:
        raise ValueError(
            f"projectors span {nl_flat.shape[-1]} G-vectors but freq_mask "
            f"selects {n_g}; they must come from the same mask."
        )
    nl_bp = nl_flat.reshape(nb * nphi, n_g)                     # [(beta*phi), n_g]

    nl_d = jnp.asarray(nl_d_matrix) if nl_d_matrix is not None else None
    # Deprecated sign path (only used when nl_d_matrix is absent).
    if nl_d is None and nl_sign is not None:
        sign_bp = jnp.repeat(jnp.asarray(nl_sign), nphi)
    else:
        sign_bp = None

    def scatter(phi_masked):
        full = jnp.zeros((int(N),), dtype=phi_masked.dtype)
        full = full.at[jnp.asarray(idx)].set(phi_masked)
        return full.reshape(grid_shape)

    def gather(full_grid):
        return full_grid.reshape(-1)[jnp.asarray(idx)]

    def H_op(phi_masked: Complex[Array, "n_g"]) -> Complex[Array, "n_g"]:
        # Kinetic (diagonal in G).
        out = T_flat * phi_masked

        # Local: FFT[ v_eff(r) * IFFT(phi_full) ], restricted to masked G.
        phi_full = scatter(phi_masked)
        psi_r = jnp.fft.ifftn(phi_full, axes=range(-dim, 0))
        vpsi_g = jnp.fft.fftn(v_eff * psi_r, axes=range(-dim, 0))
        out = out + gather(vpsi_g)

        # Nonlocal Kleinman-Bylander term H_nl = (1/vol) sum_{ij,m} conj(beta_i)
        # D_ij beta_j. Conjugation convention matches the dense
        # hamiltonian_nonlocal / legacy matrix-free path (no conj on the inner
        # projection, conj on the output scatter) so test_06 stays exact.
        if nl_d is not None:
            # H_nl = sum_ij |beta_i> D_ij <beta_j| means the INNER projection is
            # <beta_j|phi> = sum_G conj(beta_j(G)) phi(G) and the outer scatter
            # uses beta_i(G) unconjugated. The legacy order below has them
            # swapped, which is equivalent to replacing every projector by its
            # conjugate: beta(G) = 4pi(-i)^l Y_lm beta_r e^{-i(G+k).R} becomes
            # 4pi(+i)^l Y_lm beta_r e^{+i(G+k).R}, i.e. the projectors of an atom
            # at R are placed at -R. Harmless for an atom at the origin, but for
            # the zincblende anion at a/4(1,1,1) (2R is not a lattice vector) the
            # projectors land on the empty tetrahedral site, so the deep state
            # bound by that atom's local potential never sees the KB repulsion.
            if os.environ.get("HNL_CONJ_FIX", "1") == "1":
                inner = jnp.einsum("bmg,g->bm", jnp.conj(nl_flat), phi_masked)
                coupled = nl_d @ inner
                out = out + jnp.einsum("bmg,bm->g", nl_flat, coupled) / vol
            else:
                inner = jnp.einsum("bmg,g->bm", nl_flat, phi_masked)
                coupled = nl_d @ inner
                out = out + jnp.einsum("bmg,bm->g", jnp.conj(nl_flat), coupled) / vol
        else:
            proj = nl_bp @ phi_masked
            if sign_bp is not None:
                proj = sign_bp * proj
            out = out + (jnp.conj(nl_bp).T @ proj) / vol

        return out

    return H_op, n_g
