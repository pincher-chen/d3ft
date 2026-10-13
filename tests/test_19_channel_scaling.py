"""Per-channel KB scaling has to hit the channel it names.

``scaled_d_matrix`` multiplies the Kleinman-Bylander coupling matrix by
``1 + theta_l`` for each angular-momentum channel. Two things can go wrong
silently: the angular momenta can fall out of step with ``concat_d_matrix``'s
per-atom layout, so the scaling lands on the wrong projectors and still returns a
perfectly well-formed matrix; and the uniform special case can drift from the
scalar path it is supposed to reproduce exactly.

The second matters more than it looks. The physical result that justified this
parameterisation -- ``dB/d(theta_l) = [46.4, 54.1, -2.6]`` GPa for s, p, d on AlP,
summing to 97.91 against the uniform +97.909 -- rests on the sum rule
``sum_l dB/dtheta_l = dB/dtheta_uniform``. That rule is an identity of the linear
parameterisation, so it holds even if the labels are permuted, and it is only a
meaningful check if the uniform case really does degenerate to the scalar path.
The channel assignment itself was confirmed separately, by bumping one channel at
a time and finite-differencing B: +4.32 measured against +4.64 predicted for s,
+5.31 against +5.41 for p, the shortfall being the known curvature of B(theta).

Synthetic pseudopotentials, no DFT, no GPU.
"""

import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from d3ft.pseudopotential import concat_d_matrix, scaled_d_matrix  # noqa: E402


def _fake_pseudopot(per_atom_l):
    """A pseudopotential stub whose D entries encode their own position.

    ``D[i, j] = 100 * i + j`` inside each atom block, so any misplacement of the
    scaling is visible in the result rather than hidden behind plausible numbers.
    """
    blocks, ls = [], []
    for l_list in per_atom_l:
        n = len(l_list)
        blocks.append(jnp.asarray(
            [[100.0 * i + j for j in range(n)] for i in range(n)]))
        ls.append(np.asarray(l_list))
    return types.SimpleNamespace(
        nonlocal_d_matrix=blocks, nonlocal_angular_momentum=ls)


def check_zero_is_the_identity(failures):
    pp = _fake_pseudopot([[0, 0, 1, 1, 2, 2], [0, 1]])
    raw = concat_d_matrix(pp.nonlocal_d_matrix)
    a = scaled_d_matrix(pp, theta_nonlocal=0.0)
    b = scaled_d_matrix(pp, theta_nonlocal_l=[0.0, 0.0, 0.0])
    da = float(jnp.max(jnp.abs(a - raw)))
    db = float(jnp.max(jnp.abs(b - raw)))
    print(f"  zero: scalar path off by {da:.1e}, per-channel path off by {db:.1e}")
    if da > 0 or db > 0:
        failures.append(f"theta = 0 is not the identity: {da:.2e}, {db:.2e}")


def check_uniform_degenerates_to_the_scalar(failures):
    """The sum rule is only meaningful if this holds exactly."""
    pp = _fake_pseudopot([[0, 0, 1, 1, 2, 2], [0, 1]])
    for t in (0.1, -0.25, 1.0):
        a = scaled_d_matrix(pp, theta_nonlocal=t)
        b = scaled_d_matrix(pp, theta_nonlocal_l=[t, t, t])
        d = float(jnp.max(jnp.abs(a - b)))
        print(f"  uniform t={t:+.2f}: max|scalar - per-channel| = {d:.2e}")
        if d > 1e-13:
            failures.append(
                f"uniform theta_l={t} does not reproduce the scalar path: {d:.2e}")


def check_only_the_named_channel_moves(failures):
    """Bump one l; every entry outside that channel must be untouched."""
    per_atom = [[0, 0, 1, 1, 2, 2], [0, 1]]
    pp = _fake_pseudopot(per_atom)
    raw = np.asarray(concat_d_matrix(pp.nonlocal_d_matrix))
    ls = np.concatenate([np.asarray(l) for l in per_atom])
    for target in (0, 1, 2):
        tl = [0.0, 0.0, 0.0]
        tl[target] = 0.5
        got = np.asarray(scaled_d_matrix(pp, theta_nonlocal_l=tl))
        inside = (ls[:, None] == target) & (ls[None, :] == target)
        outside_moved = float(np.max(np.abs(got[~inside] - raw[~inside])))
        want_in = raw[inside] * 1.5
        inside_err = float(np.max(np.abs(got[inside] - want_in))) if inside.any() else 0.0
        n_in = int(inside.sum())
        print(f"  l={target}: {n_in} entries scaled, "
              f"inside error {inside_err:.1e}, outside moved {outside_moved:.1e}")
        if inside_err > 1e-12:
            failures.append(f"l={target}: scaled entries are wrong ({inside_err:.2e})")
        # Entries coupling the target channel to another one get the symmetrised
        # half-factor; only entries touching neither channel must be pinned.
        neither = (ls[:, None] != target) & (ls[None, :] != target)
        pinned = float(np.max(np.abs(got[neither] - raw[neither])))
        if pinned > 1e-12:
            failures.append(
                f"l={target}: entries in no way involving it moved by {pinned:.2e}")


def check_layout_follows_the_atom_order(failures):
    """Angular momenta and D blocks must concatenate the same way.

    Two atoms with different projector counts: if the l vector were built in the
    other order the second atom's channels would be scaled with the first's
    parameters, and every entry would still look reasonable.
    """
    pp = _fake_pseudopot([[0, 0, 0], [1, 1]])
    got = np.asarray(scaled_d_matrix(pp, theta_nonlocal_l=[0.0, 2.0, 0.0]))
    raw = np.asarray(concat_d_matrix(pp.nonlocal_d_matrix))
    first = float(np.max(np.abs(got[:3, :3] - raw[:3, :3])))
    second_ratio = got[3:, 3:] / np.where(raw[3:, 3:] == 0, 1.0, raw[3:, 3:])
    nz = raw[3:, 3:] != 0
    ratio = float(np.max(np.abs(second_ratio[nz] - 3.0))) if nz.any() else 0.0
    print(f"  atom 0 (all l=0) unchanged to {first:.1e}; "
          f"atom 1 (all l=1) scaled by 3 to {ratio:.1e}")
    if first > 1e-12:
        failures.append(f"the l=0 atom moved when only l=1 was bumped: {first:.2e}")
    if ratio > 1e-12:
        failures.append(f"the l=1 atom is not scaled by 1+2: off by {ratio:.2e}")


def check_result_stays_symmetric(failures):
    """D is symmetric; the scaling must not break that."""
    pp = _fake_pseudopot([[0, 1, 1, 2]])
    sym_in = jnp.asarray([[1.0, 2.0, 3.0, 4.0],
                          [2.0, 5.0, 6.0, 7.0],
                          [3.0, 6.0, 8.0, 9.0],
                          [4.0, 7.0, 9.0, 10.0]])
    pp.nonlocal_d_matrix = [sym_in]
    got = np.asarray(scaled_d_matrix(pp, theta_nonlocal_l=[0.3, -0.2, 0.7]))
    asym = float(np.max(np.abs(got - got.T)))
    print(f"  symmetry preserved to {asym:.1e}")
    if asym > 1e-12:
        failures.append(f"scaling broke the symmetry of D: {asym:.2e}")


def main():
    failures = []
    for check in (
        check_zero_is_the_identity,
        check_uniform_degenerates_to_the_scalar,
        check_only_the_named_channel_moves,
        check_layout_follows_the_atom_order,
        check_result_stays_symmetric,
    ):
        check(failures)

    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_19_channel_scaling")
        return 1
    print("[OK]  per-channel scaling hits the channel it names, and the uniform "
          "case is the scalar path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
