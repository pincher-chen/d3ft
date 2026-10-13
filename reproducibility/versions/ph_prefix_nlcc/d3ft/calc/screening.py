"""Which parameters actually move the number you care about.

Fitting a parameter that does not couple to the target wastes the fit: the
gradient is zero, the Newton step divides by nothing, and a loss curve trained on
it descends normally while learning a direction that changes no observable. That
happened here -- a uniform scaling of the Kleinman-Bylander D matrix was fitted to
C44 twice, converging cleanly both times, before the sensitivity turned out to be
zero to within noise. See :mod:`d3ft.calc.stationary` for the measurements.

Screening is where the stationary-point derivative pays best. One reverse-mode
pass returns the derivative with respect to **every** parameter at once, so the
cost does not grow with the number of candidates -- and screening is by
construction the step with the most candidates, since choosing among them is the
point. Seven parameters cost three ground states here; a finite-difference screen
of the same seven would cost 3 x (2 x 7 + 1) = 45.

Two things the report must carry, both learned the hard way:

*Scale.* Raw derivatives are not comparable across parameters with different
units and magnitudes. PBE's ``kappa`` is 0.804 and its ``beta`` is 0.0667, so
``dC44/d(kappa)`` and ``dC44/d(beta)`` differ by an order of magnitude before any
physics enters. Multiplying by the range the parameter would plausibly move over
is what makes the column sortable.

*A noise floor.* The C44 sensitivities come from a second difference of
``dF/d(theta)`` over shear strain, and that estimator carries about +/-3.8 GPa at
``delta = 0.02``. Without the floor printed next to the values, a 3 GPa entry
reads as a small coupling rather than as nothing. The same estimator is fine for
the bulk modulus, where the signal is 25x larger; whether it is usable depends on
the signal, not on the method.
"""
from __future__ import annotations

import copy
from functools import partial
import os
from typing import Any, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .elastic import _HA_BOHR3_TO_GPA, ground_state_at_strain
from .forces import relax_internal_coordinates
from .implicit import _kwargs_from
from .stress import free_energy_at_strain, ewald_translation_indices
from .xc_tunable import default_xc_params

__all__ = ["parameter_names", "sensitivity", "screen", "screen_report"]


def parameter_names(xc_base: Dict[str, Dict[str, float]],
                    n_channels: int = 3) -> List[str]:
    """Names for the combined parameter vector, in the order it is packed."""
    names = [f"theta_l{i}" for i in range(n_channels)]
    for func in sorted(xc_base):
        # 'gga_x_pbe' -> 'x', 'gga_c_pbe' -> 'c'; the trailing token is 'pbe'
        # for both, which made the two functionals indistinguishable by name and
        # let a caller line a scale vector up with the wrong parameters.
        kind = func.split("_")[1]
        names += [f"{kind}:{k}" for k in sorted(xc_base[func])]
    return names


def sensitivity(out, xc_base: Dict[str, Dict[str, float]],
                n_channels: int = 3) -> np.ndarray:
    """``dF/dp`` for the whole parameter vector, from one reverse-mode pass.

    The vector is offsets from the current values, packed as ``n_channels``
    per-angular-momentum KB strengths followed by the exchange-correlation
    constants in the order :func:`parameter_names` gives.

    Args:
        out: a converged, force-relaxed :class:`GroundStateOutput`. The same
            stationarity licence as :func:`d3ft.calc.stationary.dfree_energy_dtheta`
            applies -- zero force and a variational minimum.
        xc_base: the unperturbed functional constants, from
            :func:`d3ft.calc.xc_tunable.default_xc_params`.
        n_channels: number of angular-momentum channels.

    Returns:
        ``numpy`` array of shape ``(n_channels + n_xc,)``, Hartree per unit
        parameter.
    """
    kwargs = _kwargs_from(out)
    kwargs['ewald_indices']=ewald_translation_indices(
        out.crystal.cell_vectors,out.config.ewald_args['ewald_cutoff'])
    other = {k: v for k, v in kwargs.items()
             if k not in ("params_pw", "params_occ")}
    # These four are what the closure below owns. Leaving them in `other` would
    # pass each one twice; taking `xc_base` from the module defaults rather than
    # from the state would differentiate a different functional than the one the
    # state was solved with -- the same mismatch that made theta's derivative
    # silently wrong before it was guarded.
    for k in ("theta_nonlocal", "theta_nonlocal_l", "xc_params"):
        other.pop(k, None)
    natom = np.asarray(out.crystal.positions).shape[0]
    theta_base = kwargs.get("theta_nonlocal_l")
    if theta_base is None:
        theta_base = np.full(n_channels, kwargs.get("theta_nonlocal", 0.0))
    theta_base = jnp.asarray(theta_base)
    if theta_base.shape != (n_channels,):
        raise ValueError("n_channels must match the current theta_nonlocal_l vector")
    funcs = sorted(xc_base)
    keys = {f: sorted(xc_base[f]) for f in funcs}
    n_xc = sum(len(keys[f]) for f in funcs)

    def energy_at(p,o,v):
        xp, i = {}, n_channels
        for f in funcs:
            xp[f] = {k: xc_base[f][k] + v[i + j] for j, k in enumerate(keys[f])}
            i += len(keys[f])
        return free_energy_at_strain(
            jnp.zeros((3, 3)), p, o,
            displacement=jnp.zeros((natom, 3)),
            theta_nonlocal_l=theta_base + v[:n_channels], xc_params=xp, **other,
        )
    pw_state=kwargs['params_pw'];occ_state=kwargs['params_occ']
    zero=jnp.zeros(n_channels+n_xc)
    sharded=(os.environ.get('D3FT_FUSE_FINAL_DENSITY')=='1'
             and hasattr(pw_state,'sharding')
             and hasattr(pw_state.sharding,'mesh'))
    if sharded:
        from jax.sharding import NamedSharding,PartitionSpec
        replicated=NamedSharding(pw_state.sharding.mesh,PartitionSpec())
        occ_layout=jax.tree_util.tree_map(
            lambda x:x.sharding if hasattr(x,'sharding') else replicated,occ_state)
        from d3ft.pseudopotential import local,nlcc
        original_local,original_nlcc=local.CubicSpline,nlcc.CubicSpline
        try:
            local.CubicSpline=partial(original_local,check=False)
            nlcc.CubicSpline=partial(original_nlcc,check=False)
            calculate=jax.jit(
                lambda p,o,v:jax.grad(energy_at,argnums=2)(p,o,v),
                in_shardings=(pw_state.sharding,occ_layout,replicated),
                out_shardings=replicated)
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE sensitivity_gradient_begin',flush=True)
            result=calculate(pw_state,occ_state,zero)
            jax.block_until_ready(result)
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE sensitivity_gradient_complete',flush=True)
            return np.asarray(result)
        finally:
            local.CubicSpline,nlcc.CubicSpline=original_local,original_nlcc
    return np.asarray(jax.grad(lambda v:energy_at(pw_state,occ_state,v))(zero))


def screen(config, symbols: Sequence[str], mode: str = "isotropic",
           delta: float = 0.02, n_channels: int = 3,
           hessian=None, workdir: Optional[str] = None,
           verbose: bool = True) -> Dict[str, Any]:
    """Sensitivities of an elastic observable to every candidate parameter.

    Three strain points, one reverse pass each.

    ``"isotropic"`` gives the bulk modulus and the pressure and needs no
    relaxation -- cubic symmetry pins the internal coordinate -- so it is the
    cheap screen to run first. ``"tetragonal"`` gives ``C11 - C12`` under a
    volume-conserving strain and needs no relaxation either, for the same kind
    of reason: both sublattices stay on S4 axes, which forbids any displacement.
    ``"shear"`` gives C44 and is the only one that moves the atoms, so it is
    also the only one exposed to a defective force; pass ``hessian`` or it falls
    back to steepest descent and takes 17 steps instead of 3.

    That split matters for which systems can contribute what. A force that
    cannot be trusted -- see ``d3ft.calc.forces`` on d-valence systems -- costs
    only C44. a0, B, B', C11 and C12 all come from energies and stresses and are
    unaffected.

    Args:
        config: base config at the reference geometry.
        symbols: chemical symbols in crystal order.
        mode: ``"isotropic"`` or ``"shear"``.
        delta: strain amplitude. 0.02 rather than 0.01 for the shear screen: the
            signal grows as delta^2 while the estimator noise does not.
        n_channels: angular-momentum channels in the pseudopotential.
        hessian: force-constant matrix, for Newton steps in the shear mode.
        workdir: scratch directory.
        verbose: print each strain point as it lands.

    Returns:
        Dict with ``names``, ``d_observable`` (per parameter), ``observable``
        (the constant itself, as a control), ``mode`` and ``delta``.

    Raises:
        ValueError: for an unknown mode.
    """
    if mode not in ("isotropic", "tetragonal", "shear"):
        raise ValueError(
            f"mode must be 'isotropic', 'tetragonal' or 'shear', got {mode!r}"
        )

    # The constants this run is actually using, so the derivatives are taken at
    # the current point rather than at PBE's published values.
    xc_base = config.d3ft.get("xc_params", None) or default_xc_params(config.xc)
    xc_base = {f: dict(p) for f, p in xc_base.items()}
    names = parameter_names(xc_base, n_channels)
    mask, energies, grads = None, {}, {}

    for d in (0.0, -delta, +delta):
        if mode == "isotropic":
            out = ground_state_at_strain(config, np.eye(3) * d, list(symbols),
                                         workdir=workdir, freq_mask=mask)
        elif mode == "tetragonal":
            # Volume-conserving tetragonal strain, and like the isotropic one it
            # needs no relaxation: under it both sublattices stay on S4 axes, and
            # S4 sends a displacement to minus itself, so symmetry pins the
            # internal coordinate exactly as cubic symmetry does at zero strain.
            # Only the shear branch below moves the atoms.
            #
            # eps_zz is the exact (1+d)^-2 - 1 rather than the linearised -2d.
            # The two differ by 3 d^2, which at delta = 0.02 is 1.2e-3 of strain
            # -- second order in exactly the quantity whose second derivative is
            # being measured, so linearising here biases the answer it returns.
            out = ground_state_at_strain(
                config, np.diag([d, d, (1.0 + d) ** -2 - 1.0]), list(symbols),
                workdir=workdir, freq_mask=mask,
            )
        else:
            strain = np.zeros((3, 3))
            strain[1, 2] = strain[2, 1] = d
            r = relax_internal_coordinates(
                config, list(symbols), strain=strain, workdir=workdir,
                verbose=False, hessian=hessian, freq_mask=mask,
                max_steps=22, force_tol=1.0e-4,
            )
            if not r["converged"]:
                # A relaxation that runs out of steps returns its last geometry
                # like any other, and C44 comes out of it with nothing to show
                # that the atoms never reached zero force. The step count is set
                # by the quality of `hessian`: GaAs at cutoff 50 Ha on a force
                # constant matrix whose ASR residual is 1.17e-1 (against 2e-3 for
                # AlP and Si) needed roughly 11 Newton steps per strain instead
                # of 3, which is close enough to the cap of 22 that hitting it is
                # a live possibility rather than a theoretical one.
                raise RuntimeError(
                    f"internal relaxation did not converge at strain {d:+.4f}: "
                    f"max|F| = {r['max_force']:.3e} Ha/Bohr after {r['steps']} "
                    f"steps (tolerance 1.0e-4). C44 from this geometry would be "
                    f"wrong with nothing to show for it. Improve the force "
                    f"constant matrix passed as `hessian` -- its acoustic sum "
                    f"rule residual is the thing that sets the step count."
                )
            out = r["ground_state"]
        if mask is None:
            mask = out.freq_mask
        energies[d] = float(out.total_energy)
        grads[d] = sensitivity(out, xc_base, n_channels)
        if d == 0.0:
            # The reference volume, captured inside the loop. Reading it from
            # whatever `out` the loop leaves behind picks up the +delta cell and
            # scales every constant by (1+delta)^3 -- 6% at delta = 0.02. This
            # module inherited that bug from the script it was promoted from, and
            # it was caught by a control value rather than by review.
            vol = float(abs(np.linalg.det(np.asarray(out.crystal.cell_vectors))))
        if verbose:
            print(f"  strain {d:+.4f}: dF/dp = "
                  f"{np.array2string(grads[d], precision=6)}", flush=True)

    second_e = energies[+delta] - 2 * energies[0.0] + energies[-delta]
    second_g = grads[+delta] - 2 * grads[0.0] + grads[-delta]

    if mode == "isotropic":
        # V = V0 (1+d)^3, so dV = 3 V0 dd and B = d2E/dd2 / (9 V0).
        conv = _HA_BOHR3_TO_GPA / (delta ** 2 * 9.0 * vol)
        first_e = energies[+delta] - energies[-delta]
        first_g = grads[+delta] - grads[-delta]
        p_conv = -_HA_BOHR3_TO_GPA / (2.0 * delta * 3.0 * vol)
        extra = {"pressure": first_e * p_conv,
                 "d_pressure": first_g * p_conv}
    elif mode == "tetragonal":
        # dE = 3 (C11 - C12) V0 d^2 under this strain, so the curvature is
        # 6 (C11 - C12) V0 and the observable returned is C11 - C12.
        #
        # Three modes, three different denominators -- 9 V0, 6 V0, and a quarter
        # -- and they are not interchangeable. Substituting the isotropic one
        # into the shear branch has already happened here once and cost a factor
        # of 9/4; see the comment below. Combine this with the isotropic B
        # through :func:`cubic_constants`, which is where that arithmetic lives
        # so it is written down once and tested.
        conv = _HA_BOHR3_TO_GPA / (delta ** 2 * 6.0 * vol)
        extra = {}
    else:
        # Voigt: eps_4 = 2 eps_yz, so the curvature in eps_yz is divided by four.
        # This is NOT the 1/(9 V0) of the isotropic branch; using that one by
        # mistake scales everything by 9/4.
        conv = 0.25 * _HA_BOHR3_TO_GPA / (delta ** 2 * vol)
        extra = {}

    return {"names": names, "mode": mode, "delta": delta,
            "observable": second_e * conv, "d_observable": second_g * conv,
            **extra}


def screen_report(result: Dict[str, Any], scales: Optional[Dict[str, float]] = None,
                  noise: Optional[float] = None,
                  control: Optional[Dict[str, float]] = None,
                  expect_observable: Optional[float] = None) -> str:
    """Render a screen, sorted by scaled magnitude, with the floor alongside.

    Args:
        result: what :func:`screen` returned.
        scales: per-parameter range to multiply by, making the column comparable.
            Defaults to 0.1 for the KB channels and the parameter's own value for
            the functional constants -- i.e. "what if this moved by 100%".
        noise: the estimator's standard deviation in GPa. Entries below it are
            reported as indistinguishable from zero rather than as small.
        control: expected values for any parameters that were measured
            independently, so a broken run announces itself.
        expect_observable: an independently known value of the constant itself.
            Pass it whenever one exists. The first version of this module scaled
            every result by ``(1+delta)^3`` because it read the volume from the
            strained cell, and nothing in the numbers looked wrong -- the ratios
            between parameters were right, only the overall scale was off. A
            control on the observable is what caught it, so the report asks for
            one rather than leaving it to the caller to remember.

    Returns:
        A multi-line string.
    """
    names = result["names"]
    d = np.asarray(result["d_observable"])
    label = {"shear": "C44", "tetragonal": "C11-C12"}.get(result["mode"], "B")
    obs = float(result["observable"])
    head = (f"{label} = {obs:.3f} GPa "
            f"(mode={result['mode']}, delta={result['delta']})")
    if expect_observable is not None:
        rel = abs(obs - expect_observable) / abs(expect_observable)
        head += (f"   against {expect_observable:.3f}: "
                 f"{'ok' if rel < 0.01 else f'MISMATCH {rel:.1%}'}")
    lines = [head]
    if expect_observable is None:
        lines.append("  (no control value given -- pass expect_observable when "
                     "one is known)")

    sc = dict(scales or {})
    for n in names:
        sc.setdefault(n, 0.1)

    rows = sorted(zip(names, d), key=lambda r: -abs(r[1] * sc[r[0]]))
    lines.append(f"  {'parameter':>14} {'d' + label:>10} {'x range':>10}  verdict")
    for n, v in rows:
        if noise is None:
            verdict = ""
        elif abs(v) < noise:
            verdict = f"null ({abs(v)/noise:.1f} sigma)"
        elif abs(v) < 2.5 * noise:
            verdict = f"marginal ({abs(v)/noise:.1f} sigma), widen delta"
        else:
            verdict = f"couples ({abs(v)/noise:.1f} sigma)"
        lines.append(f"  {n:>14} {v:>10.3f} {v*sc[n]:>10.3f}  {verdict}")

    if control:
        lines.append("  control:")
        for n, want in control.items():
            got = float(d[names.index(n)])
            ok = "ok" if abs(got - want) < max(0.02 * abs(want), 1e-9) else "MISMATCH"
            lines.append(f"    {n}: {got:+.3f} against {want:+.3f}  {ok}")
    if noise is not None:
        lines.append(f"  noise floor {noise:.2f} GPa")
    return "\n".join(lines)


def cubic_constants(bulk_modulus, c11_minus_c12, c44=None):
    """Assemble C11 and C12 from the two screens that measure them.

    The two are independent measurements of independent combinations:
    ``screen(mode="isotropic")`` gives ``B = (C11 + 2 C12) / 3`` and
    ``screen(mode="tetragonal")`` gives ``C11 - C12``. Solving the pair,

        C11 = B + (2/3) (C11 - C12)
        C12 = B - (1/3) (C11 - C12)

    This lives in one place, and is tested, because the arithmetic connecting a
    curvature to an elastic constant is where this module has gone wrong before:
    the shear branch once used the isotropic ``1 / (9 V0)`` in place of the Voigt
    quarter and scaled C44 by 9/4 with nothing to show for it. The three strain
    modes carry three different denominators -- ``9 V0``, ``6 V0`` and a quarter
    -- and none of them is interchangeable with another.

    The Born criteria for a cubic crystal come back alongside, but be clear on
    what they can do. For any positive ``B`` and positive ``C11 - C12`` they are
    satisfied identically -- ``C11 + 2 C12 = 3B > 0`` by construction, and
    ``C11 > |C12|`` reduces to ``2B > -(C11 - C12) / 3`` -- so **they cannot
    catch a wrong factor**. Feeding the 9/4 error above gives C11 = 251,
    C12 = 21.5 and a perfectly stable crystal. What they do catch is a sign: a
    negative ``C11 - C12`` from a slipped sign or a fit that has wandered fails
    ``C11 > |C12|``, and a negative ``B`` fails both. Scale errors have to be
    caught by the factors themselves being written down and tested, which is
    what ``tests/test_25_tetragonal_strain.py`` does.

    Args:
      bulk_modulus: B in GPa, from the isotropic screen.
      c11_minus_c12: C11 - C12 in GPa, from the tetragonal screen.
      c44: C44 in GPa from the shear screen. Optional -- it is the one constant
        that needs a relaxation, so it is also the one unavailable on a system
        whose forces cannot be trusted (see :mod:`d3ft.calc.forces`).

    Returns:
      ``{"C11", "C12", "C44", "B", "shear_modulus", "anisotropy", "stable",
      "violations"}``. ``shear_modulus`` is the tetragonal shear
      ``C' = (C11 - C12) / 2``; ``anisotropy`` is Zener's ``2 C44 / (C11 - C12)``,
      1 for an elastically isotropic crystal, and ``None`` without ``c44``.
    """
    b = float(bulk_modulus)
    d = float(c11_minus_c12)
    c11 = b + 2.0 * d / 3.0
    c12 = b - d / 3.0

    violations = []
    # Born stability for cubic symmetry.
    if not c11 > abs(c12):
        violations.append(f"C11 > |C12| fails: {c11:.2f} vs {abs(c12):.2f}")
    if not c11 + 2.0 * c12 > 0.0:
        violations.append(f"C11 + 2 C12 > 0 fails: {c11 + 2.0 * c12:.2f}")
    if c44 is not None and not float(c44) > 0.0:
        violations.append(f"C44 > 0 fails: {float(c44):.2f}")

    return {
        "C11": c11,
        "C12": c12,
        "C44": None if c44 is None else float(c44),
        "B": b,
        "shear_modulus": d / 2.0,
        "anisotropy": None if c44 is None else 2.0 * float(c44) / d if d != 0.0 else None,
        "stable": not violations,
        "violations": violations,
    }
