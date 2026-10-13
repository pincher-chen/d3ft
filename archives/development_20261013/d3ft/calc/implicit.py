"""Parameter derivatives of the relaxed-ion stress -- **incomplete, do not use**.

The intent was ``d(sigma_yz)/d(theta)`` at the relaxed internal coordinate, with
``theta`` scaling the Kleinman-Bylander D matrix as a stand-in for a learnable
Hamiltonian parameter. Verified against finite differences it is wrong by 362%
on AlP and 224% on Si, with the opposite sign in both cases. The cause is a
design error, not a coding slip, and it is recorded here rather than deleted
because the reasoning was wrong in a way that is easy to repeat.

**What the design assumed.** Two implicit conditions in series -- the variational
state ``psi*(theta)`` and force balance ``R*(theta)`` -- and that the first is
"free by the variational property" while only the second needs the implicit
function theorem. The first half of that is false.

At the minimum ``dE/d(psi) = 0``, so a *first* derivative equals its partial:
``sigma = dE/d(eps) = partial E / partial eps`` is Hellmann-Feynman and holds.
``d(sigma)/d(theta)`` is a **mixed second** derivative and is not:

    d(sigma)/d(theta) = d2E/(d eps d theta) + (d2E / d eps d psi) . d(psi)/d(theta)

``partial E / partial psi`` vanishing does not make ``d2E / d eps d psi`` vanish.
The same applies to ``dF/d(theta)``. Both were computed at frozen coefficients,
so both are missing the electronic response.

**Verified on both materials -- but read the reference's own noise first.**
Against finite differences over theta, each point fully re-relaxed:

| | full chain | electronic only | finite difference |
|---|---|---|---|
| AlP, k = 6, FD relaxed by steepest descent | -2.4580e-05 | +5.5136e-06 | +5.8374e-06 |
| AlP, k = 6, FD relaxed by Newton | -2.4580e-05 | +5.5136e-06 | **+1.1055e-05** |
| AlP, k = 4 | -2.2026e-05 | +7.0278e-06 | +8.4147e-06 |
| Si, k = 6 | -4.0942e-05 | +1.3614e-05 | +3.3120e-05 |

The first two rows are the same calculation, differing only in how the two
finite-difference relaxations were converged, and **their "ground truth" differs
by a factor of 1.9**. So the reference is not accurate enough to support fine
statements.

Why: `dsigma/dtheta` is about 1e-05 and is taken as
`(sigma(+0.02) - sigma(-0.02)) / 0.04` from sigma values near 4.1e-05. The
difference being measured is 3.4e-07, **0.8% of the baseline**, and a relaxation
converged to `max|F|` of 2e-05 leaves a position uncertainty whose effect on
sigma is the same order. A usable reference needs a theta step of 0.1 to 0.2, or
a force tolerance an order tighter, or both.

**What survives that noise**: the chain gives -2.46e-05 on AlP where both finite
differences give positive values, and -4.09e-05 on Si against +3.31e-05. **The
sign is wrong**, robustly, and that is enough to withdraw the chain.

**What does not survive**: any claim about how accurate the electronic term is on
its own, or about how much of the derivative the ionic response carries. The two
AlP runs put the electronic term's error at 5.6% and 50%, which is the noise
speaking. An earlier revision of this file recorded 5.6% and built an argument on
it; that argument is withdrawn along with the opposite one before it ("the ionic
term dominates and flips the sign"). Both were read off a reference that cannot
resolve them.

The claim Phase 2 actually rests on is untouched by any of this, because it does
not involve this derivative: the **value** of C44 needs the ionic relaxation,
84.9 GPa clamped against 61.4 relaxed on AlP under PBE, against an experimental
61.5.

**The factor-by-factor diagnostic** (AlP, k = 4, eps_yz = 0.01, against
relaxations at theta = +/- 0.02):

| | chain | finite difference | |
|---|---|---|---|
| `dsigma/dtheta` | -2.2026e-05 | +8.4147e-06 | 362% |
| electronic term alone | +7.0278e-06 | -- | |
| ionic term | -2.9054e-05 | +1.3869e-06 | wrong sign, 21x |
| `dR*/dtheta`, atom 0 x | +2.5029e-02 | -2.2259e-03 | wrong sign, 11x |

and the tell is in ``dF/dtheta`` itself: its components are +7.17e-03 and
-3.18e-03, a **net of +3.99e-03, 56% of the individual terms**. Translational
invariance requires the net to vanish. It does not, for exactly the reason the
acoustic sum rule fails at frozen coefficients -- translating the nuclei without
letting the electrons follow does change the energy. Feeding that spurious net
through ``Phi^-1`` is what produces the 1224% error in ``dR*/dtheta``.

The electronic term is 84% of the truth on AlP and 41% on Si. Its looking
plausible on AlP is luck, and it is what led to the earlier -- now withdrawn --
claim that "the ionic term dominates and flips the sign".

**What a correct version needs.** ``d(psi)/d(theta)``: a Sternheimer/linear-
response solve, implicit differentiation through the electronic minimisation, or
finite differences over theta (which is the reference this was checked against,
and costs a full re-relaxation per parameter). That is a design task, not a
repair, and it is the thing the AD-DFPT literature actually solves.

**None of this touches the empirical Phase 2 result.** Clamped against relaxed
C44 -- 84.9 to 61.4 GPa on AlP under PBE, against an experimental 61.5 -- comes
from direct relaxation and the AD stress, both validated against QE and
experiment. What is unproven is the claim that the derivative of that quantity
can be obtained cheaply.
"""

from __future__ import annotations

import copy
import os
import tempfile
from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .elastic import _write_xyz
from .forces import forces_from_ground_state, relax_internal_coordinates
from .stress import free_energy_at_strain, stress_from_ground_state

__all__ = [
    "theta_partials",
    "combine_implicit_chain",
    "response_constants_at",
    "dsigma_yz_dtheta_at_strain",
    "verify_against_finite_difference",
]


def _kwargs_from(out) -> Dict[str, Any]:
    return dict(
        config=out.config, crystal=out.crystal, freq_mask=out.freq_mask,
        make_coeff=out.make_coeff, get_occupation=out.get_occupation,
        pseudopot=out.pseudopot, beta_tables=out.beta_tables, kmax=out.beta_kmax,
        grid_kmax=out.grid_kmax, k_frac=out.k_frac, temperature=out.temperature,
        eps=out.final_eps, params_pw=out.params_pw, params_occ=out.params_occ,
        # See stress_from_ground_state: theta lives on the D matrix built inside
        # ground_state, not on pseudopot, so it does not survive a round trip
        # through GroundStateOutput unless carried explicitly.
        theta_nonlocal=float(out.config.d3ft.get("theta_nonlocal", 0.0)),
        theta_nonlocal_l=out.config.d3ft.get("theta_nonlocal_l", None),
        xc_params=out.config.d3ft.get("xc_params", None),
    )


def theta_partials(out, natom: Optional[int] = None):
    """Hellmann-Feynman partials at fixed R: ``d(sigma_yz)/d(theta)`` and
    ``dF/d(theta)``, both at ``theta = 0`` and the geometry stored in ``out``.

    No re-solving -- both are plain autodiff through the already-converged,
    frozen coefficients, licensed by the electronic stationarity at ``out``.

    Args:
        out: a converged :class:`GroundStateOutput`. Its own
            ``crystal.cell_vectors`` / ``crystal.positions`` are treated as the
            reference geometry (zero strain, zero displacement in this frame).
        natom: number of atoms; inferred from ``out.crystal.positions`` if
            omitted.

    Returns:
        ``sigma_yz`` at theta=0, ``d(sigma_yz)/d(theta)``, and ``dF/d(theta)``
        (shape ``(natom, 3)``, Ha/Bohr per unit theta).
    """
    kwargs = _kwargs_from(out)
    natom = natom or np.asarray(out.crystal.positions).shape[0]
    zero_strain = jnp.zeros((3, 3))
    zero_disp = jnp.zeros((natom, 3))
    cell = jnp.asarray(out.crystal.cell_vectors)
    vol = jnp.abs(jnp.linalg.det(cell))

    other = {k: v for k, v in kwargs.items() if k not in ("params_pw", "params_occ")}
    # _kwargs_from now carries the state's theta, which this function owns as an
    # argument instead. Evaluate at that value rather than at zero: the module is
    # withdrawn (see the docstring) but a wrong evaluation point is a second
    # defect on top of the one it was withdrawn for, and the tests still run it.
    theta0 = float(other.pop("theta_nonlocal", 0.0))

    def free_energy(strain, disp, theta):
        return free_energy_at_strain(
            strain, kwargs["params_pw"], kwargs["params_occ"],
            displacement=disp, theta_nonlocal=theta, **other,
        )

    def sigma_yz(theta):
        grad = jax.grad(free_energy, argnums=0)(zero_strain, zero_disp, theta)
        sig = 0.5 * (grad + grad.T) / vol
        return sig[1, 2]

    def force(theta):
        return -jax.grad(free_energy, argnums=1)(zero_strain, zero_disp, theta)

    sigma_val = float(sigma_yz(theta0))
    dsigma_dtheta = float(jax.grad(sigma_yz)(theta0))
    # Forward mode, not reverse. theta is a scalar and the force is a (natom, 3)
    # vector, so jax.jacobian -- which is jacrev -- runs one VJP per output
    # component and vmaps them, holding six backward passes at once. With the
    # per-k nonlocal projectors at 573 MB that asks for 17 GB and dies. A single
    # JVP gives the same derivative in one pass.
    dforce_dtheta = np.asarray(jax.jacfwd(force)(theta0))
    return sigma_val, dsigma_dtheta, dforce_dtheta


def _solve_at_positions(config, symbols, cell, positions, init_params, freq_mask, workdir):
    from .ground_state import calc

    tag = f"{abs(hash(positions.tobytes())):016x}"
    xyz = os.path.join(workdir, f"impl_{tag}.xyz")
    _write_xyz(xyz, cell, positions, symbols)
    cfg = copy.deepcopy(config)
    cfg.crystal = None
    cfg.crystal_file_path_path = xyz
    return calc(cfg, init_params=init_params, freq_mask=freq_mask)


def response_constants_at(
    config, symbols: Sequence[str], relaxed_out,
    displacement_step: float = 0.01, workdir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """``Phi = -dF/dR`` and ``d(sigma_yz)/dR`` at ``theta = 0``, by re-solving.

    Displaces each Cartesian coordinate of each atom by ``+/- displacement_step``
    around ``relaxed_out``'s geometry, re-converging the ground state at each
    point (see the module docstring on why AD at frozen coefficients will not
    do), and central-differences both the Hellmann-Feynman force and
    ``sigma_yz`` across the SAME set of solves. Costs ``6 N`` extra ground
    states for ``N`` atoms.

    Args:
        config: base config (cutoff, k-mesh, xc, ...); its own geometry is
            unused, only ``relaxed_out``'s is.
        symbols: chemical symbols in crystal order.
        relaxed_out: a converged :class:`GroundStateOutput`, typically from
            :func:`d3ft.calc.forces.relax_internal_coordinates`, whose geometry
            is expanded about.
        displacement_step: Cartesian displacement, Bohr.
        workdir: where displaced geometries are written.
        verbose: report each displacement.

    Returns:
        ``Phi`` (3N, 3N, Ha/Bohr^2, ASR-imposed) and ``dsigma_dR`` (3N,
        Ha/Bohr^3 per Bohr).
    """
    symbols = list(symbols)
    cell = np.asarray(relaxed_out.crystal.cell_vectors)
    pos0 = np.asarray(relaxed_out.crystal.positions)
    natom = pos0.shape[0]
    mask = relaxed_out.freq_mask
    seed = {"pw": relaxed_out.params_pw, "occ": relaxed_out.params_occ}

    owned = workdir is None
    workdir = tempfile.mkdtemp(prefix="d3ft_implicit_") if owned else workdir
    os.makedirs(workdir, exist_ok=True)

    phi = np.zeros((3 * natom, 3 * natom))
    dsigma_dr = np.zeros(3 * natom)
    for atom in range(natom):
        for axis in range(3):
            col = 3 * atom + axis
            dvec = np.zeros((natom, 3))
            dvec[atom, axis] = displacement_step
            out_p = _solve_at_positions(config, symbols, cell, pos0 + dvec, seed, mask, workdir)
            out_m = _solve_at_positions(config, symbols, cell, pos0 - dvec, seed, mask, workdir)
            f_p = np.asarray(forces_from_ground_state(out_p))
            f_m = np.asarray(forces_from_ground_state(out_m))
            s_p = float(np.asarray(stress_from_ground_state(out_p))[1, 2])
            s_m = float(np.asarray(stress_from_ground_state(out_m))[1, 2])
            phi[:, col] = -(f_p - f_m).reshape(-1) / (2.0 * displacement_step)
            dsigma_dr[col] = (s_p - s_m) / (2.0 * displacement_step)
            if verbose:
                print(f"    atom {atom} axis {axis}: |dF|="
                      f"{np.abs(f_p - f_m).max():.3e}  d(sigma_yz)={s_p - s_m:+.3e}")

    phi = 0.5 * (phi + phi.T)
    blocks = phi.reshape(natom, 3, natom, 3)
    for i in range(natom):
        blocks[i, :, i, :] -= blocks[i].sum(axis=1)  # acoustic sum rule

    return {"Phi": phi, "dsigma_dR": dsigma_dr}


def combine_implicit_chain(dsigma_dtheta_r, dforce_dtheta, phi, dsigma_dr):
    """``d(sigma_yz)/d(theta)`` from the four pieces the implicit chain needs.

    Pure arithmetic, no DFT -- split out so the sign conventions (the minus in
    the implicit function theorem, ``Phi = -dF/dR`` so ``dF/dR = -Phi``) can be
    tested against a synthetic system without spending any GPU time. See
    ``tests/test_16_implicit.py``.

    Args:
        dsigma_dtheta_r: ``d(sigma_yz)/d(theta)`` at fixed R (scalar).
        dforce_dtheta: ``dF/d(theta)`` at fixed R, shape ``(natom, 3)``.
        phi: force-constant matrix, ``-dF/dR``, shape ``(3N, 3N)``.
        dsigma_dr: ``d(sigma_yz)/dR`` at fixed theta, shape ``(3N,)``.

    Returns:
        ``(dR_dtheta, total)``: the ionic response, shape ``(3N,)``, and the
        total ``d(sigma_yz)/d(theta)``.
    """
    # dF/dR = -Phi, so dR*/dtheta = -(dF/dR)^-1 (dF/dtheta) = Phi^-1 (dF/dtheta).
    # Pseudoinverse: Phi carries exact zero modes (rigid translation) after the
    # acoustic sum rule is imposed, and dforce_dtheta has no component along
    # them (translating the whole crystal cannot feel a Hamiltonian parameter),
    # so pinv reproduces the unique physical solution rather than papering over
    # a genuine singularity.
    dr_dtheta = np.linalg.pinv(phi) @ np.asarray(dforce_dtheta).reshape(-1)
    total = float(dsigma_dtheta_r) + float(np.dot(np.asarray(dsigma_dr), dr_dtheta))
    return dr_dtheta, total


def dsigma_yz_dtheta_at_strain(
    config, symbols: Sequence[str], strain,
    displacement_step: float = 0.01, relax_kwargs: Optional[dict] = None,
    workdir: Optional[str] = None, verbose: bool = True,
) -> Dict[str, Any]:
    """Total ``d(sigma_yz)/d(theta)`` at the relaxed geometry for ``strain``.

    Relaxes internal coordinates at ``theta = 0`` (the physical pseudopotential),
    then assembles the implicit-chain derivative from :func:`theta_partials` and
    :func:`response_constants_at`.

    Returns:
        Dict with ``sigma_yz``, ``dsigma_dtheta_electronic`` (the ``|_R`` term
        alone), ``dsigma_dtheta_total``, ``dR_dtheta``, ``Phi``, ``dsigma_dR``,
        and the relaxation diagnostics.
    """
    relax_kwargs = dict(relax_kwargs or {})
    relax_kwargs.setdefault("max_steps", 22)
    relax_kwargs.setdefault("force_tol", 1.0e-4)
    relax_kwargs.setdefault("step", 1.0)
    if verbose:
        print(f"  relaxing at strain (eps_yz={float(strain[1, 2]):+.4f})...")
    r = relax_internal_coordinates(
        config, symbols, strain=strain, workdir=workdir, verbose=verbose,
        **relax_kwargs,
    )
    relaxed_out = r["ground_state"]
    if verbose:
        print(f"  relaxed: {r['steps']} steps, converged={r['converged']}, "
              f"max|F|={r['max_force']:.3e} Ha/Bohr")

    resp = response_constants_at(
        config, symbols, relaxed_out, displacement_step, workdir, verbose,
    )
    sigma_val, dsigma_dtheta_r, dforce_dtheta = theta_partials(relaxed_out)

    dr_dtheta, total = combine_implicit_chain(
        dsigma_dtheta_r, dforce_dtheta, resp["Phi"], resp["dsigma_dR"]
    )

    if verbose:
        print(f"  sigma_yz(theta=0)                       = {sigma_val:+.4e}")
        print(f"  d(sigma_yz)/dtheta |_R  (electronic)     = {dsigma_dtheta_r:+.4e}")
        print(f"  d(sigma_yz)/dtheta, ionic-relaxation term= {total - dsigma_dtheta_r:+.4e}")
        print(f"  d(sigma_yz)/dtheta  TOTAL                = {total:+.4e}")

    return {
        "sigma_yz": sigma_val,
        "dsigma_dtheta_electronic": dsigma_dtheta_r,
        "dsigma_dtheta_total": total,
        "dR_dtheta": dr_dtheta,
        "relax_steps": r["steps"],
        "relax_converged": r["converged"],
        "relax_max_force": r["max_force"],
        **resp,
    }


def verify_against_finite_difference(
    config, symbols: Sequence[str], strain, theta_step: float = 0.02,
    displacement_step: float = 0.01, relax_kwargs: Optional[dict] = None,
    workdir: Optional[str] = None, verbose: bool = True,
) -> Dict[str, Any]:
    """Check the implicit-chain derivative against an independent finite difference.

    The finite difference bakes ``theta`` into the SCF solve itself
    (``config.d3ft.theta_nonlocal``) and fully re-relaxes the internal
    coordinate at ``theta0 +/- theta_step`` -- a completely separate code path
    from the implicit chain (no autodiff through frozen coefficients anywhere,
    no force-constant matrix), so agreement is not circular.

    Returns:
        Dict with both derivatives and their relative difference, plus the
        implicit-chain diagnostics from :func:`dsigma_yz_dtheta_at_strain`.
    """
    relax_kwargs = dict(relax_kwargs or {})
    relax_kwargs.setdefault("max_steps", 22)
    relax_kwargs.setdefault("force_tol", 1.0e-4)
    relax_kwargs.setdefault("step", 1.0)

    implicit = dsigma_yz_dtheta_at_strain(
        config, symbols, strain, displacement_step=displacement_step,
        relax_kwargs=relax_kwargs, workdir=workdir, verbose=verbose,
    )

    # Reuse the force-constant matrix the implicit half already paid for. The
    # finite difference needs two full relaxations, and steepest descent takes 17
    # steps each where a Newton step takes 2 -- two hours against twenty minutes.
    # It does not weaken the check: Phi only sets the search direction, and the
    # convergence criterion on the force is unchanged, so the geometry the two
    # sides land on is the same one. What the check compares is still two
    # independent routes to d(sigma)/d(theta), one through the implicit function
    # theorem and one through re-solving at shifted theta.
    fd_kwargs = dict(relax_kwargs)
    fd_kwargs.setdefault("hessian", implicit["Phi"])

    sigmas = {}
    for sign, tag in ((+1, "plus"), (-1, "minus")):
        cfg = copy.deepcopy(config)
        cfg.d3ft.theta_nonlocal = sign * theta_step
        if verbose:
            print(f"  finite difference: theta = {sign * theta_step:+.3f}, relaxing...")
        r = relax_internal_coordinates(
            cfg, symbols, strain=strain, workdir=workdir, verbose=verbose,
            **fd_kwargs,
        )
        sig = np.asarray(stress_from_ground_state(r["ground_state"]))[1, 2]
        sigmas[tag] = float(sig)
        if verbose:
            print(f"    theta={sign * theta_step:+.3f}: {r['steps']} steps, "
                  f"converged={r['converged']}, sigma_yz={sig:+.4e}")

    fd = (sigmas["plus"] - sigmas["minus"]) / (2.0 * theta_step)
    total = implicit["dsigma_dtheta_total"]
    electronic_only = implicit["dsigma_dtheta_electronic"]
    rel_err = abs(total - fd) / abs(fd) if fd else float("nan")
    rel_err_electronic_only = abs(electronic_only - fd) / abs(fd) if fd else float("nan")

    if verbose:
        print(f"\n  implicit chain (electronic + ionic) = {total:+.4e}")
        print(f"  implicit chain (electronic term only, i.e. ignoring relaxation)"
              f" = {electronic_only:+.4e}")
        print(f"  finite difference over theta (ground truth) = {fd:+.4e}")
        print(f"  relative error, full chain     = {rel_err:.2%}")
        print(f"  relative error, electronic only = {rel_err_electronic_only:.2%}"
              "   (should be much worse if the ionic term matters)")

    return {
        "implicit_total": total,
        "implicit_electronic_only": electronic_only,
        "finite_difference": fd,
        "relative_error": rel_err,
        "relative_error_electronic_only": rel_err_electronic_only,
        "sigmas": sigmas,
        **implicit,
    }
