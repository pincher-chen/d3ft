"""Stress tensor by direct differentiation of the free energy.

D3FT minimises the Mermin free energy ``F = E - T S``, so the stress is

.. math::
    \\sigma_{\\alpha\\beta} = \\frac{1}{\\Omega}
        \\left.\\frac{\\partial F}{\\partial \\varepsilon_{\\alpha\\beta}}\\right|_{\\text{params}}

evaluated at the converged variational parameters. Differentiating ``E`` instead
would be wrong at finite smearing: the occupations are variational degrees of
freedom, and only at the minimum *of F* does ``dF/d(params) = 0`` let the total
derivative collapse to the partial one. The entropy itself depends only on
``params_occ`` and so contributes nothing to the cell derivative, but it is kept
explicit here so the objective being differentiated is unambiguous.

Strain is applied as ``a_i -> (I + eps) a_i`` with atomic positions carried along
affinely, i.e. fractional coordinates held fixed.

**No Pulay term.** The plane-wave basis is indexed by a fixed frequency mask on
integer Miller indices, not by a fixed energy cutoff, so the basis deforms with
the cell and the number of plane waves does not change under strain. That is what
makes the partial derivative at fixed coefficients the whole answer. It also
means the stress inherits the basis-set error of that fixed mask; converge the
cutoff before trusting absolute numbers.
"""
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jrystal._src import energy, entropy as jr_entropy, pw
from jrystal._src.ewald import ewald_coulomb_repulsion
from jrystal._src.grid import _vector_grid, g_vectors

from .. import pseudopotential as d3pp

__all__ = ["free_energy_at_strain", "stress_tensor", "pressure",
           "stress_from_ground_state", "stress_report",
           "ewald_translation_indices", "variational_residual"]

_HA_BOHR3_TO_GPA = 29421.015697  # Hartree/Bohr^3 -> GPa


def ewald_translation_indices(cell_vectors, cutoff) -> np.ndarray:
    """Integer lattice-image indices for the Ewald real-space sum.

    ``jrystal._src.grid.translation_vectors`` decides how many images to keep by
    reading the cell, which makes it unusable under a strain derivative. How many
    images to sum is a convergence parameter, not something to differentiate, so
    the indices are frozen at the nominal cell here and only the vectors
    ``indices @ cell`` follow the strain.
    """
    cell = np.asarray(cell_vectors)
    dim = cell.shape[0]
    n = int(np.ceil(cutoff / np.linalg.norm(np.sum(cell, axis=0)) ** 2))
    unit = np.eye(dim)
    grid = np.reshape(np.asarray(_vector_grid(unit, [n] * dim)), [-1, dim])
    return grid


def _strained(cell_vectors, positions, strain):
    """Apply ``a_i -> (I + eps) a_i`` and carry the atoms along affinely."""
    f = jnp.eye(3) + strain
    return cell_vectors @ f.T, positions @ f.T


def free_energy_at_strain(
    strain,
    params_pw,
    params_occ,
    *,
    config,
    crystal,
    freq_mask,
    make_coeff: Callable,
    get_occupation: Callable,
    pseudopot=None,
    beta_tables=None,
    kmax: float = None,
    grid_kmax: float = None,
    k_frac=None,
    temperature: float = 0.0,
    eps: float = None,
    core_charge: bool = True,
    ewald_indices=None,
    displacement=None,
    theta_nonlocal=0.0,
    theta_nonlocal_l=None,
    xc_params=None,
):
    """Mermin free energy of a strained cell at fixed variational parameters.

    Args:
      strain: symmetric ``[3, 3]`` strain tensor; the stress is its gradient at 0.
      params_pw, params_occ: converged parameters (from ``GroundStateOutput``).
      make_coeff: ``params_pw -> [spin, kpt, n_g, band]``. Must depend on the
        fixed frequency mask only, never on the cell.
      get_occupation: ``(params_occ, eps) -> occupations``.
      beta_tables: output of :func:`d3ft.pseudopotential.build_tables`, tabulated
        once at ``kmax`` so the transform is not redone at every strain.
      k_frac: k-points in fractional coordinates; they must be re-expressed in
        Cartesian at the strained cell, which is why fractional is what gets stored.
      temperature: smearing temperature the run converged at.
    """
    if eps is None:
        raise ValueError(
            "eps must be the Sinkhorn epsilon the run ended at "
            "(GroundStateOutput.final_eps). It is annealed during optimisation and "
            "sets the occupations, so reusing config.eps -- a different quantity, "
            "the entropy floor -- would evaluate the stress at the wrong occupations."
        )
    grid_sizes = np.asarray(freq_mask.shape)

    cell, positions = _strained(
        jnp.asarray(crystal.cell_vectors), jnp.asarray(crystal.positions), strain
    )
    # Cartesian displacement of the nuclei at fixed cell, for forces. Applied
    # after the strain so the two derivatives stay independent: d/d(strain) at
    # zero displacement is the stress, d/d(displacement) at zero strain is minus
    # the force, and neither picks up the other.
    if displacement is not None:
        positions = positions + jnp.asarray(displacement)
    vol = jnp.abs(jnp.linalg.det(cell))
    g_vec = g_vectors(cell, grid_sizes)
    recip = 2 * jnp.pi * jnp.linalg.inv(cell).T
    k_vec = jnp.asarray(k_frac) @ recip

    coeff = make_coeff(params_pw)
    occ = get_occupation(params_occ, eps)
    density = pw.density_grid(coeff, vol, occ)
    # ``density_grid_reciprocal`` recomputes ``density_grid`` from ``coeff``.
    # The real-space density is already available, so transform it directly;
    # this is algebraically identical and avoids a second full wavefunction FFT
    # and reverse-mode tape in stress/stationarity derivatives.
    density_rec = jnp.fft.fftn(density, axes=range(-3, 0))

    e_kin = energy.kinetic(g_vec, k_vec, coeff, occ)
    e_har = energy.hartree(density_rec, g_vec, vol)

    core_density_grid = None
    if pseudopot is not None:
        pot_loc = d3pp.potential_local_reciprocal(
            positions, g_vec, pseudopot.r_grid, pseudopot.local_potential_grid,
            pseudopot.local_potential_charge, vol, kmax=grid_kmax,
        )
        # Cutoff sphere, not the box; see d3ft.pseudopotential.nonlocal_. The
        # mask is on Miller indices and does not follow the cell, so the same
        # G-vectors are selected at every strain -- which is the same fixed
        # basis the "no Pulay term" argument at the top of this module rests on.
        gk_vec = d3pp.gk_vectors(g_vec, k_vec, freq_mask)
        beta_gk = d3pp.beta_grid(
            pseudopot.r_grid, pseudopot.nonlocal_beta_grid,
            pseudopot.nonlocal_angular_momentum, gk_vec, kmax, tables=beta_tables,
        )
        pot_nl = d3pp.projectors_reciprocal(
            positions, gk_vec, pseudopot.nonlocal_angular_momentum, beta_gk,
        )
        # theta_nonlocal is a learnable-parameter stand-in: a scalar multiplier
        # on the Kleinman-Bylander D matrix, i.e. the nonlocal pseudopotential
        # strength. It exists so d3ft.calc.implicit can differentiate through it
        # at fixed (converged) params via Hellmann-Feynman -- no re-solving --
        # and is the mechanism, not a physical fit target on its own. Zero
        # reproduces the unperturbed pseudopotential exactly.
        nl_d = d3pp.scaled_d_matrix(
            pseudopot, theta_nonlocal=theta_nonlocal,
            theta_nonlocal_l=theta_nonlocal_l,
        )

        e_ext = (
            d3pp.energy_local(density_rec, pot_loc, vol=vol)
            + d3pp.energy_nonlocal(coeff, pot_nl, vol=vol, occupation=occ,
                                   d_matrix=nl_d, freq_mask=freq_mask)
        )
        cores = getattr(pseudopot, "core_charge_grid", None)
        if core_charge and cores is not None and any(c is not None for c in cores):
            core_density_grid = d3pp.core_charge_real(
                positions, g_vec,
                [r for r, c in zip(pseudopot.r_grid, cores) if c is not None],
                [c for c in cores if c is not None],
                vol, kmax=grid_kmax,
            )
        charges = jnp.asarray(pseudopot.valence_charges, dtype=float)
    else:
        e_ext = energy.external(density_rec, crystal.positions, crystal.charges,
                                g_vec, vol)
        charges = jnp.asarray(crystal.charges, dtype=float)

    xc_grid_factor = config.d3ft.get('xc_grid_factor', 1)
    if xc_params is None and xc_grid_factor == 1:
        e_xc = energy.xc_energy(density, g_vec, vol, config.xc, kohn_sham=False,
                                core_density_grid=core_density_grid)
    else:
        # Only when someone asks for tunable constants; the default path stays on
        # jrystal's implementation so nothing already measured can shift.
        from .xc_tunable import xc_energy_tunable
        e_xc = xc_energy_tunable(density, g_vec, vol, config.xc,
                                 core_density_grid=core_density_grid,
                                 xc_params=xc_params, grid_factor=xc_grid_factor)

    # Ion-ion. The image set is fixed by integer indices; only the vectors they
    # map to follow the strain.
    if ewald_indices is None:
        ewald_indices = ewald_translation_indices(
            crystal.cell_vectors, config.ewald_args["ewald_cutoff"]
        )
    ewald_grid = jnp.asarray(ewald_indices) @ cell
    e_ewald = ewald_coulomb_repulsion(
        positions, charges, g_vec, vol,
        ewald_eta=config.ewald_args["ewald_eta"], ewald_grid=ewald_grid,
    )

    total = e_kin + e_har + e_ext + e_xc + e_ewald
    if temperature:
        # Zero derivative with respect to strain (the occupations are parameters),
        # but kept so the differentiated object is F and not E.
        total = total - temperature * jr_entropy.fermi_dirac(
            get_occupation(params_occ, eps), eps=eps
        )
    return total


def stress_tensor(**kwargs):
    """``sigma_ab = (1/Omega) dF/d(eps_ab)`` at zero strain, in Hartree/Bohr^3.

    Takes the same keyword arguments as :func:`free_energy_at_strain` minus
    ``strain``. Returns a symmetrised ``[3, 3]`` array; a positive diagonal means
    the cell is under tension.
    """
    crystal = kwargs["crystal"]
    vol = float(jnp.abs(jnp.linalg.det(jnp.asarray(crystal.cell_vectors))))
    grad = jax.grad(lambda s: free_energy_at_strain(s, **kwargs))(jnp.zeros((3, 3)))
    return 0.5 * (grad + grad.T) / vol


def variational_residual(return_parts: bool = False, **kwargs):
    """Residual of the stationarity condition behind the Hellmann-Feynman step.

    :func:`stress_tensor` assumes ``dF/d(params) = 0`` so the total derivative with
    respect to strain collapses to the partial one. This measures how far from that
    the state actually is.

    The wavefunction parameters live ON the Stiefel manifold -- ``stiefel_coeff``
    deliberately does not re-orthonormalise, because ``retract_qr`` keeps them
    there. At a minimum the *Riemannian* gradient vanishes while the Euclidean one
    generally does not: it retains a component along the manifold normal, which is
    pure gauge and moves nothing physical. Measuring the raw ``dF/d(params_pw)``
    therefore reports a large residual no matter how well converged the run is.
    The projection onto the tangent space is what has to be checked.

    Args:
      return_parts: also return the separate wavefunction and occupation residuals.

    Returns:
      ``max`` residual, or ``(total, {"pw": ..., "occ": ...})``.
    """
    kw = dict(kwargs)
    pw_params = kw.pop("params_pw")
    occ_params = kw.pop("params_occ")
    config = kw["config"]

    def f(pw_p, occ_p):
        return free_energy_at_strain(jnp.zeros((3, 3)), pw_p, occ_p, **kw)

    g_pw, g_occ = jax.grad(f, argnums=(0, 1))(pw_params, occ_params)

    # JAX returns the *conjugate* Wirtinger gradient for a real-valued function of
    # a complex input, so the Euclidean gradient is its conjugate. The magnitude is
    # unchanged, which is why this went unnoticed -- but project_tangent is not a
    # magnitude, and projecting the unconjugated vector keeps a component the
    # optimiser never moves along. Measured on AlP at the converged state: the
    # unconjugated projection reports 2.6e-3 and walking along it does not lower
    # the energy at all (dE/predicted = 0.000), while the conjugated one reports
    # 5.5e-5 and reproduces the first-order slope to 0.999. The residual floor that
    # 8x more optimiser effort could not move was this, not a real one.
    # d3ft.optim.riemannian_adam conjugates before projecting for the same reason.
    g_pw = jnp.conj(g_pw)

    if config.d3ft.optim.enabled:
        from ..optim.stiefel import project_tangent
        g_pw = project_tangent(pw_params, g_pw)

    r_pw = float(jnp.max(jnp.abs(g_pw)))
    occ_leaves = jax.tree_util.tree_leaves(g_occ)
    r_occ = float(max(jnp.max(jnp.abs(l)) for l in occ_leaves)) if occ_leaves else 0.0

    total = max(r_pw, r_occ)
    return (total, {"pw": r_pw, "occ": r_occ}) if return_parts else total


def pressure(sigma) -> float:
    """Hydrostatic pressure ``-tr(sigma)/3`` in GPa."""
    return float(-jnp.trace(sigma) / 3.0) * _HA_BOHR3_TO_GPA


def stress_from_ground_state(out, **overrides):
    """``sigma`` from a converged :class:`~d3ft.calc.ground_state.GroundStateOutput`.

    Uses the parameters the optimiser converged to, which is what makes the
    Hellmann-Feynman step legitimate: ``dF/d(params) = 0`` holds at the minimum of
    F, so the total derivative with respect to strain is the partial one.
    """
    kwargs = dict(
        params_pw=out.params_pw,
        params_occ=out.params_occ,
        config=out.config,
        crystal=out.crystal,
        freq_mask=out.freq_mask,
        make_coeff=out.make_coeff,
        get_occupation=out.get_occupation,
        pseudopot=out.pseudopot,
        beta_tables=out.beta_tables,
        kmax=out.beta_kmax,
        grid_kmax=out.grid_kmax,
        k_frac=out.k_frac,
        temperature=out.temperature,
        eps=out.final_eps,
        # theta_nonlocal is applied to the KB D matrix inside ground_state,
        # not stored on pseudopot, so it has to be carried here or this
        # evaluates a different Hamiltonian than the state was converged
        # with. Omitting it made C44 appear to depend on theta when the
        # energy says it does not; see d3ft.calc.stationary.
        theta_nonlocal=float(out.config.d3ft.get("theta_nonlocal", 0.0)),
        theta_nonlocal_l=out.config.d3ft.get("theta_nonlocal_l", None),
        xc_params=out.config.d3ft.get("xc_params", None),
    )
    kwargs.update(overrides)
    return stress_tensor(**kwargs)


def stress_report(out, **overrides) -> Dict[str, Any]:
    """``sigma``, the pressure, and the variational residual that qualifies them."""
    kwargs = dict(
        params_pw=out.params_pw, params_occ=out.params_occ, config=out.config,
        crystal=out.crystal, freq_mask=out.freq_mask, make_coeff=out.make_coeff,
        get_occupation=out.get_occupation, pseudopot=out.pseudopot,
        beta_tables=out.beta_tables, kmax=out.beta_kmax, grid_kmax=out.grid_kmax,
        k_frac=out.k_frac, temperature=out.temperature, eps=out.final_eps,
        # See stress_from_ground_state.
        theta_nonlocal=float(out.config.d3ft.get("theta_nonlocal", 0.0)),
        theta_nonlocal_l=out.config.d3ft.get("theta_nonlocal_l", None),
        xc_params=out.config.d3ft.get("xc_params", None),
    )
    kwargs.update(overrides)
    sigma = stress_tensor(**kwargs)
    return {
        "stress": sigma,
        "pressure_GPa": pressure(sigma),
        **dict(zip(("variational_residual", "residual_parts"),
                   variational_residual(return_parts=True, **kwargs))),
    }
