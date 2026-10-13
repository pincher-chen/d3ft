"""Hellmann-Feynman forces, and relaxation of the internal coordinates.

``F_I = -dF/dR_I`` at fixed cell. The same argument that licenses the stress
licenses this: the optimiser converges to ``dF/d(psi) = 0``, so the total
derivative with respect to a nuclear coordinate equals the partial one and no
response of the wavefunctions has to be computed.

**There is no Pulay force here**, and for a sharper reason than in the stress.
Pulay forces arise when the basis functions are attached to the atoms, so that
moving an atom moves the variational space; the correction is the part of
``dF/dR`` that acts through the basis. A plane-wave basis is attached to the
cell, not to the atoms, so displacing a nucleus at fixed cell leaves the
variational space untouched and the correction is identically zero. This is one
of the standard arguments for plane waves and it is worth stating precisely,
because the corresponding claim for the *stress* is weaker -- there the basis
does deform, and the reason there is no Pulay term is the separate one that the
Miller-index set is held fixed.

The acoustic sum rule ``sum_I F_I = 0`` follows from translational invariance and
is reported by :func:`force_report`, but read it carefully: **it tests the state
and the implementation together, not the implementation alone.** Translating the
nuclei at *fixed* plane-wave coefficients does not translate the electron
density, so the energy genuinely changes and the sum only vanishes as the state
approaches the variational minimum. A nonzero sum therefore bounds the error in
the forces without attributing it; pair it with
:func:`~d3ft.calc.stress.variational_residual` to tell an unconverged state from
a missing term. What it does test unambiguously is a *rigid translation of the
whole system*, which is how the KB projector conjugation bug was caught: 9.947 eV
before the fix, 3.1e-13 eV after.
"""

from __future__ import annotations

from functools import partial
import os
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .stress import free_energy_at_strain, ewald_translation_indices

_HA_BOHR_TO_EV_ANG = 51.42208619083232
"""Ha/Bohr to eV/Angstrom, the unit QE prints forces in."""


def forces(**kwargs):
    """``-dF/dR`` for every atom, in Ha/Bohr, shape (natom, 3).

    Takes the same keyword arguments as
    :func:`~d3ft.calc.stress.free_energy_at_strain`, minus ``strain`` and
    ``displacement``, which this owns.
    """
    kwargs.pop("strain", None)
    kwargs.pop("displacement", None)
    crystal = kwargs["crystal"]
    natom = np.asarray(crystal.positions).shape[0]
    zero_strain = jnp.zeros((3, 3))

    def energy(displacement):
        return free_energy_at_strain(
            zero_strain, kwargs["params_pw"], kwargs["params_occ"],
            displacement=displacement,
            **{k: v for k, v in kwargs.items()
               if k not in ("params_pw", "params_occ")},
        )

    return -jax.grad(energy)(jnp.zeros((natom, 3)))


def forces_from_ground_state(out, **overrides):
    """Forces from a converged :class:`GroundStateOutput`."""
    kwargs = dict(
        params_pw=out.params_pw, params_occ=out.params_occ, config=out.config,
        crystal=out.crystal, freq_mask=out.freq_mask, make_coeff=out.make_coeff,
        get_occupation=out.get_occupation, pseudopot=out.pseudopot,
        beta_tables=out.beta_tables, kmax=out.beta_kmax, grid_kmax=out.grid_kmax,
        k_frac=out.k_frac, temperature=out.temperature, eps=out.final_eps,
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
    pw_state=kwargs['params_pw']
    sharded=(os.environ.get('D3FT_FUSE_FINAL_DENSITY')=='1'
             and hasattr(pw_state,'sharding')
             and hasattr(pw_state.sharding,'mesh'))
    if sharded:
        pw_state=kwargs.pop('params_pw');occ_state=kwargs.pop('params_occ')
        kwargs['ewald_indices']=ewald_translation_indices(
            out.crystal.cell_vectors,out.config.ewald_args['ewald_cutoff'])
        from jax.sharding import NamedSharding, PartitionSpec
        replicated=NamedSharding(pw_state.sharding.mesh,PartitionSpec())
        occ_layout=jax.tree_util.tree_map(
            lambda x:x.sharding if hasattr(x,'sharding') else replicated,occ_state)
        # These identical radial tables were checked while constructing the
        # ground state. Disabling repeated interpax checks makes the force
        # traceable without an eager full-coefficient validation pass.
        from d3ft.pseudopotential import local,nlcc
        original_local,original_nlcc=local.CubicSpline,nlcc.CubicSpline
        try:
            local.CubicSpline=partial(original_local,check=False)
            nlcc.CubicSpline=partial(original_nlcc,check=False)
            calculate=jax.jit(
                lambda p,o:forces(params_pw=p,params_occ=o,**kwargs),
                in_shardings=(pw_state.sharding,occ_layout),
                out_shardings=replicated)
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE force_gradient_begin',flush=True)
            result=calculate(pw_state,occ_state)
            jax.block_until_ready(result)
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE force_gradient_complete',flush=True)
            return result
        finally:
            local.CubicSpline,nlcc.CubicSpline=original_local,original_nlcc
    return forces(**kwargs)


def force_report(out, **overrides) -> Dict[str, Any]:
    """Forces with the acoustic sum rule residual that qualifies them.

    A nonzero net force is unphysical, so ``|sum_I F_I|`` bounds the error in the
    individual forces. It does not say where the error comes from: the sum
    vanishes only at the variational minimum, so an unconverged state produces
    one just as a missing term would. See the module docstring.
    """
    f = np.asarray(forces_from_ground_state(out, **overrides))
    net = np.abs(f.sum(axis=0)).max()
    largest = np.abs(f).max()
    return {
        "forces_Ha_Bohr": f,
        "forces_eV_Ang": f * _HA_BOHR_TO_EV_ANG,
        "acoustic_sum_rule": float(net),
        "max_force": float(largest),
        "relative_asr": float(net / largest) if largest > 0 else 0.0,
    }


def force_amplification(coeff_a, force_a, coeff_b, force_b, n_occ):
    """How hard this system's forces magnify a difference between two solutions.

    Run the same ground state twice from different random seeds, then hand this
    both converged coefficient arrays and both force arrays. It returns

        amplification = |dF_max| / mean principal angle between the subspaces

    which is the only quantity found so far that separates a system whose forces
    can be trusted from one whose forces cannot.

    Measured on two seeds at 3000 steps, cutoff 50 Ha, grid 32, k = 6:

        AlP    subspace angle 4.05e-05 rad   |dF| 5.49e-06   amplification 0.135
        GaAs   subspace angle 1.85e-05 rad   |dF| 1.28e-03   amplification 69.2

    Note which way that runs. **AlP's two solutions are further apart than
    GaAs's** -- more than twice the mean principal angle -- and its forces still
    agree 233x better. GaAs does not converge worse; its forces respond ~500x
    more strongly to the same difference, because ``dV_nl/dR`` on a deep Ga 3d
    projector has that much larger a norm.

    This is why the variational residual is the wrong thing to look at: it reads
    1.7e-05 for AlP and 3.6e-05 for GaAs, the same order, and orders the two
    backwards. Reading it was what first led to dismissing convergence as the
    problem, on a system whose forces turned out to be unusable.

    Args:
      coeff_a, coeff_b: ``make_coeff(params_pw)`` from the two runs, shaped
        ``[spin, kpt, band, x, y, z]``.
      force_a, force_b: the corresponding force arrays, ``[atom, 3]``.
      n_occ: number of occupied bands. It must be **all** of them -- the bands
        within the occupied manifold have no defined order, so comparing a
        subset measures gauge and reports large angles that mean nothing. Count
        it from the valence charges: 16 electrons in AlP gives 8, 18 in GaAs
        gives 9.

    **Resolution floor: about 2e-08 rad.** The angles come from ``arccos`` of
    the overlap's singular values, and ``arccos`` is ill-conditioned at 1:
    ``arccos(1 - eps) = sqrt(2 eps) = 2.1e-08`` in double precision, so two
    identical subspaces measure as ~1e-08 apart rather than 0. The measured
    angles here (1.8e-05 to 4.1e-05) sit three decades above that, but the floor
    does bound what this can verify -- getting GaAs's force to 1% would need the
    subspace converged to ~3e-07 rad, only one decade of headroom. Below that,
    switch to a sine-form principal angle before trusting the number.

    Returns:
      ``{"amplification", "angle_mean", "angle_max", "dforce_max"}``.
    """
    a = np.asarray(coeff_a)[0][:, :n_occ]
    b = np.asarray(coeff_b)[0][:, :n_occ]
    a = a.reshape(a.shape[0], n_occ, -1)
    b = b.reshape(b.shape[0], n_occ, -1)
    angles = []
    for ik in range(a.shape[0]):
        sv = np.linalg.svd(a[ik] @ b[ik].conj().T, compute_uv=False)
        angles.append(np.arccos(np.clip(sv, -1.0, 1.0)))
    angles = np.asarray(angles)
    angle_mean = float(angles.mean())
    dforce = float(np.abs(np.asarray(force_a) - np.asarray(force_b)).max())
    return {
        "amplification": dforce / angle_mean if angle_mean > 0 else float("inf"),
        "angle_mean": angle_mean,
        "angle_max": float(angles.max()),
        "dforce_max": dforce,
    }


def _check_acoustic_sum(f, symbols, tol: float = 0.05):
    """Refuse to relax on forces that do not sum to zero.

    ``sum_I F_I = 0`` follows from translational invariance. As the module
    docstring says, a nonzero sum bounds the error in the forces without
    attributing it -- it can be an unconverged state as easily as a missing
    term. Either way a relaxation driven by such forces is not finding the
    energy minimum, so it should stop rather than return a geometry.

    Measured on a displaced zincblende cell at uniform settings (SG15 PBE,
    cutoff 50 Ha, grid 32, k = 6), ``|sum F| / max|F|``:

        AlP  (Al 11 with 2s2p semicore, P 5)      1.6e-02
        AlAs (Al 11, As 5)                        1.0e-02
        GaP  (Ga 13 with 3d, P 5)                 2.2e-01
        GaAs (Ga 13 with 3d, As 5)                5.4e-01

    Only the two systems carrying an occupied d shell break, and the sharper
    statement of the same failure is a point-group violation. With one atom
    displaced along x, C2x -- ``(x, y, z) -> (x, -y, -z)`` -- is an *exact*
    symmetry of the cell: it fixes the displaced atom and moves the other by a
    lattice vector. Every force must therefore lie along x. Transverse over
    longitudinal came out 7e-04 (AlP) and 3e-03 (AlAs) against 0.23 (GaP) and
    1.06 (GaAs).

    **On a d-valence system this force is not a reproducible quantity, and the
    cause is not known.** Four random seeds at identical settings, ``F_x`` on
    the cation over the Quantum ESPRESSO reference:

        AlP    0.9860  0.9873  0.9880  0.9851     spread 0.003
        GaAs   0.4625  0.8056  0.5198 -0.2382     spread 1.044
        GaP    1.0002  0.6372  1.4944  0.7673     spread 0.857

    The spread exceeds the quantity itself, and GaAs seed 3 has the wrong sign.
    Meanwhile the total energy over those same seeds is reproducible to 1e-06 Ha
    in every system alike. Any single agreement is luck: GaP seed 0 matches QE
    to 0.018% and seed 1 is 36% off.

    Tightening convergence makes it worse, not better. Over 3000 / 9000 / 27000
    steps on GaAs the variational residual falls monotonically -- 3.62e-05,
    1.57e-05, 6.62e-06 -- while ``F_x``/QE goes 0.46, -0.53, +2.18 and the
    transverse component grows from 9.1e-04 to 4.4e-03. The energy *rises* over
    the same sequence, by 5.1e-09 Ha, which a variational descent should not do.

    Ruled out, each by measurement rather than argument:

    * The FFT grid. ``grid_sizes`` 32/40/48/64 gives an acoustic sum rule
      violation of 0.995/0.470/0.514/0.718 on GaAs -- scatter, not a trend --
      while the energy is flat to 1e-07 Ha above 40 and the *anion* force is
      flat to 0.05%. (The grid was separately found to be aliasing at 32; that
      is a real defect, fixed in ``_check_density_grid``, but not this one.)
    * The analytic force implementation. Against QE on byte-identical
      pseudopotentials the AlP and AlAs forces agree to 1.4%, and the d-valence
      ones scatter around the reference rather than sitting off it.
    * A frozen-wavefunction Hellmann-Feynman error. That would shrink with the
      residual; it grows.
    * Gauge freedom in the degenerate Ga 3d manifold. The occupations are
      exactly uniform -- 9 occupied bands at 2/216 to every digit, 4 empty bands
      at exactly zero -- so the density and every energy term are strictly
      invariant under any unitary rotation of the occupied subspace, at any
      displacement. Gauge cannot enter.
    * A differing converged subspace. The seed-0 and seed-1 occupied subspaces
      differ by principal angles of 1.8e-05 rad on average (1.8e-03 max), which
      accounts for the 3.7e-08 Ha energy difference at second order but has no
      predictive relation to the force: the residual and the force move in
      opposite directions across the step scan above.

    Until this is understood, treat a force on any system with an occupied d
    shell as unusable, and with it every quantity built on the internal
    relaxation those forces drive -- C44 above all. AlP and AlAs are unaffected.

    **Do not diagnose this by comparing the analytic force to a finite
    difference of the energy.** Both sides are computed from the same converged
    state, so they agree while both are wrong: on GaAs the analytic force sits
    54% below Quantum ESPRESSO on a byte-identical pseudopotential, yet its gap
    to d3ft's own finite difference is only 10%. The same comparison at
    mismatched settings had earlier reported 112%, and neither number attributes
    anything. **Nor is a single comparison against an external reference enough**
    -- it has to be repeated over seeds, or it reports the luck of one draw.
    """
    f = np.asarray(f)
    fmax = float(np.abs(f).max())
    if fmax <= 0.0:
        return
    ratio = float(np.abs(f.sum(axis=0)).max()) / fmax
    if ratio > tol:
        raise RuntimeError(
            f"forces violate the acoustic sum rule: |sum F| is {ratio:.1%} of "
            f"max|F| (tolerance {tol:.0%}) for {list(symbols)}. Relaxing on "
            f"these would return a geometry that is not a force minimum. "
            f"Check the FFT grid first -- an aliased density breaks the "
            f"crystal's own point group and is the known cause here; "
            f"_check_density_grid reports the ratio. Then "
            f"d3ft.calc.stress.variational_residual, to tell an unconverged "
            f"state from a wrong force. Do not use a finite difference of the "
            f"energy as the check: it reads the same grid as the analytic "
            f"force and agrees with it while both are wrong."
        )


def relax_internal_coordinates(
    config,
    symbols,
    strain=None,
    max_steps: int = 20,
    force_tol: float = 1.0e-5,
    step: float = 0.5,
    hessian=None,
    freq_mask=None,
    workdir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Move the atoms to zero force at fixed cell, re-solving at each geometry.

    This is what separates a **clamped-ion** elastic constant from a relaxed one.
    In zincblende the two sublattices shift relative to one another under a shear
    -- the Kleinman internal-strain parameter -- and holding them fixed makes C44
    substantially too stiff: this code gives 89.8 GPa clamped against an
    experimental 61.5. Comparing to QE at matched settings does not expose that,
    since a QE `scf` run with fixed `ATOMIC_POSITIONS` clamps them too and agrees
    to 0.1%. Only relaxing does.

    Pass ``hessian`` to take Newton steps instead of steepest-descent ones. This
    is close to free: the force-constant matrix is needed anyway for the implicit
    chain and for phonons, its 12 displaced solves are independent so they
    parallelise across GPUs, and it is reusable across strains and parameter
    values while steepest descent has to be paid again at every one. Measured on
    AlP, steepest descent contracts the force by 0.787 per step and needs 15-17
    steps to reach 1e-4 Ha/Bohr; a Newton step solves the harmonic problem
    exactly, so what is left is only anharmonicity.

    That matters because the relaxation is the one part of this workload that
    cannot be data-parallel -- each step needs the geometry the last one
    produced. Cutting 17 sequential ground states to 3 is worth more than
    parallelising the parts that were already embarrassingly parallel.

    Steepest descent with a fixed step is the fallback: the whole point is to reach
    zero force, and the geometry has one free internal parameter by symmetry, so
    there is nothing here that warrants a quasi-Newton method and its extra
    failure modes.

    Args:
        config: base config at the reference geometry.
        symbols: chemical symbols in crystal order.
        strain: optional (3, 3) deformation to apply before relaxing.
        max_steps: cap on relaxation steps.
        force_tol: convergence threshold on the largest force, Ha/Bohr.
        step: displacement per unit force, Bohr^2/Ha.
        workdir: where deformed geometries are written.
        verbose: print the force at each step.

    Returns:
        Dict with the relaxed ``positions``, the final ``forces``, the number of
        ``steps`` taken, whether it ``converged``, and the last
        :class:`GroundStateOutput`.
    """
    import copy
    import os
    import tempfile

    from jrystal.calc.opt_utils import create_crystal

    from .elastic import _write_xyz
    from .ground_state import calc
    from .stress import _strained

    symbols = list(symbols)
    crystal = create_crystal(config)
    cell, positions = _strained(
        np.asarray(crystal.cell_vectors), np.asarray(crystal.positions),
        np.zeros((3, 3)) if strain is None else np.asarray(strain),
    )
    cell = np.asarray(cell)
    positions = np.asarray(positions)

    owned = workdir is None
    workdir = tempfile.mkdtemp(prefix="d3ft_relax_") if owned else workdir
    os.makedirs(workdir, exist_ok=True)

    seed = None
    # A caller sweeping strain has to pass the reference mask in, or each strain
    # derives its own from its own deformed lattice and the energies stop being
    # comparable. Measured on AlP at cutoff 20: |G| <= cutoff gives 1243 plane
    # waves at |eps_yz| <= 0.01 but 1227 at 0.02, and the 16 that drop out shift
    # the energy by about 0.03 mRy -- 3% of the first difference the stress is
    # being checked against. elastic.py and hessian.py already pin it this way;
    # this function had no way to accept one.
    mask = freq_mask
    out = None
    history = []
    converged = False
    for step_index in range(max_steps):
        xyz = os.path.join(workdir, f"relax_{step_index:03d}.xyz")
        _write_xyz(xyz, cell, positions, symbols)
        cfg = copy.deepcopy(config)
        cfg.crystal = None
        cfg.crystal_file_path_path = xyz
        out = calc(cfg, init_params=seed, freq_mask=mask)
        # The basis has to stay fixed across the relaxation for the same reason
        # it does across a strain sweep: otherwise the energy being minimised
        # changes underneath the geometry.
        mask = out.freq_mask if mask is None else mask
        seed = {"pw": out.params_pw, "occ": out.params_occ}

        f = np.asarray(forces_from_ground_state(out))
        # Validate the physical forces BEFORE subtracting their mean. Checking
        # after that projection makes the acoustic-sum guard pass identically.
        raw_fmax = float(np.abs(f).max())
        if raw_fmax > force_tol:
            _check_acoustic_sum(f, symbols)
        # Project out the net force. It should already be zero; removing it stops
        # any residual from translating the whole crystal instead of relaxing it.
        f = f - f.mean(axis=0)
        fmax = raw_fmax
        history.append(fmax)
        if verbose:
            print(f"    step {step_index:2d}: max|F| = {fmax:.3e} Ha/Bohr "
                  f"({fmax * _HA_BOHR_TO_EV_ANG:.4f} eV/A)")
        if fmax < force_tol:
            converged = True
            break
        if step_index + 1 == max_steps:
            # Return the geometry actually evaluated, not an unevaluated step.
            break
        if hessian is None:
            positions = positions + step * f
        else:
            # dR = Phi^-1 F, with Phi = -dF/dR. pinv rather than solve: Phi has
            # exact zero modes from rigid translation once the acoustic sum rule
            # is imposed, and f has already had its net component projected out,
            # so the null space carries nothing and pinv returns the unique
            # physical displacement.
            delta = np.linalg.pinv(np.asarray(hessian)) @ f.reshape(-1)
            positions = positions + delta.reshape(positions.shape)

    return {
        "positions": positions,
        "cell": cell,
        "forces": f,
        "freq_mask": mask,
        "max_force": history[-1] if history else None,
        "history": history,
        "steps": len(history),
        "converged": converged,
        "ground_state": out,
    }
