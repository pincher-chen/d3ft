"""The implicit-chain arithmetic, against systems with a known analytic answer.

No DFT here -- this is where the sign of the implicit function theorem
(dR*/dtheta = -(dF/dR)^-1 (dF/dtheta), and Phi := -dF/dR, so the working formula
is dR*/dtheta = +pinv(Phi) @ dF/dtheta) is most likely to go wrong, and it did:
an earlier draft of combine_implicit_chain carried a spurious extra minus sign,
caught only because separating the arithmetic out from the DFT plumbing made it
re-derivable and checkable in isolation. That is the point of this test file --
the four numbers that go into combine_implicit_chain are expensive (each comes
from a chain of ground-state solves), so the combination step has to be right
before spending any GPU time on it.
"""

import numpy as np

from d3ft.calc.implicit import combine_implicit_chain


def check_1d_harmonic(failures):
    """F(R, theta) = -k(R - R0) + c*theta. R*(theta) = R0 + c*theta/k exactly.

    sigma(R, theta) = a*R + b*theta, so
    d(sigma)/dtheta = b + a*c/k analytically.
    """
    k, c, a, b = 3.7, -1.4, 2.1, 0.6
    phi = np.array([[k]])
    dforce_dtheta = np.array([c])
    dsigma_dr = np.array([a])
    dsigma_dtheta_r = b

    dr_dtheta, total = combine_implicit_chain(dsigma_dtheta_r, dforce_dtheta, phi, dsigma_dr)
    want_dr = c / k
    want_total = b + a * c / k
    print(f"  1D: dR*/dtheta = {dr_dtheta[0]:.6f} (want {want_dr:.6f}), "
          f"total = {total:.6f} (want {want_total:.6f})")
    if abs(dr_dtheta[0] - want_dr) > 1e-10:
        failures.append(f"1D dR*/dtheta off: got {dr_dtheta[0]}, want {want_dr}")
    if abs(total - want_total) > 1e-10:
        failures.append(f"1D total off: got {total}, want {want_total} -- "
                         "this is exactly the sign error caught during development")


def check_2d_coupled(failures):
    """A 2x2 Phi with off-diagonal coupling, so the test exercises the matrix
    inverse rather than a scalar division that could hide a sign error.

    F_i(R, theta) = -sum_j K_ij (R_j - R0_j) + c_i theta, K symmetric positive
    definite. R*(theta) - R0 = K^-1 c theta, exactly.
    """
    K = np.array([[4.0, 1.0], [1.0, 3.0]])
    c = np.array([0.8, -1.2])
    a = np.array([1.5, -0.4])
    b = 0.25

    dr_dtheta, total = combine_implicit_chain(b, c, K, a)
    want_dr = np.linalg.solve(K, c)
    want_total = b + float(a @ want_dr)
    print(f"  2D: dR*/dtheta = {dr_dtheta} (want {want_dr})")
    print(f"      total = {total:.6f} (want {want_total:.6f})")
    if np.abs(dr_dtheta - want_dr).max() > 1e-10:
        failures.append(f"2D dR*/dtheta off: got {dr_dtheta}, want {want_dr}")
    if abs(total - want_total) > 1e-10:
        failures.append(f"2D total off: got {total}, want {want_total}")


def check_null_space_is_ignored_not_blown_up(failures):
    """A Phi with an exact zero mode (translation), and a dforce_dtheta with no
    component along it -- the physical case, since translating the whole
    crystal cannot feel a Hamiltonian parameter. pinv must reproduce the unique
    solution in the orthogonal complement, not raise or return nonsense.
    """
    # Phi for two coupled masses with a rigid-translation zero mode, ASR-like:
    # rows/columns sum to zero.
    k = 5.0
    phi = np.array([[k, -k], [-k, k]])
    # dforce_dtheta must be in the range of Phi, i.e. sum to zero, to be physical.
    dforce_dtheta = np.array([0.9, -0.9])
    dsigma_dr = np.array([0.3, -0.3])

    dr_dtheta, total = combine_implicit_chain(0.1, dforce_dtheta, phi, dsigma_dr)
    # Analytic: in the (1,-1)/sqrt(2) subspace, effective stiffness is 2k, so
    # dR*/dtheta along that direction is dforce_dtheta_component / (2k).
    proj = dforce_dtheta @ np.array([1.0, -1.0]) / np.sqrt(2)
    want_dr = proj / (2 * k) * np.array([1.0, -1.0]) / np.sqrt(2)
    print(f"  null-space: dR*/dtheta = {dr_dtheta} (want {want_dr})")
    if not np.all(np.isfinite(dr_dtheta)):
        failures.append("dR*/dtheta is not finite with a singular Phi")
    if np.abs(dr_dtheta - want_dr).max() > 1e-8:
        failures.append(f"null-space case off: got {dr_dtheta}, want {want_dr}")
    # pinv should also give the MINIMUM-NORM solution, i.e. no component along
    # the null space (1,1)/sqrt(2), which is the physically sensible choice
    # since a rigid shift of R* is not observable.
    null_component = dr_dtheta @ np.array([1.0, 1.0]) / np.sqrt(2)
    if abs(null_component) > 1e-8:
        failures.append(f"pinv put a component ({null_component:.2e}) along "
                         "Phi's null space; expected the minimum-norm solution")


def check_electronic_only_differs_from_total(failures):
    """Sanity check that the two pieces reported separately are actually
    different when the ionic term is nonzero -- guards against a refactor that
    silently drops the ionic contribution and still "passes" the other checks
    because it returns the right TYPE of answer.
    """
    phi = np.array([[2.0]])
    dforce_dtheta = np.array([1.0])
    dsigma_dr = np.array([1.0])
    dsigma_dtheta_r = 0.0  # electronic term deliberately zero

    _, total = combine_implicit_chain(dsigma_dtheta_r, dforce_dtheta, phi, dsigma_dr)
    print(f"  electronic-only=0 but total={total:.4f}: ionic term is load-bearing")
    if abs(total) < 1e-8:
        failures.append("total equals the (zero) electronic term -- ionic "
                         "contribution was not added")


def main():
    print("[test_16] implicit-chain arithmetic against known analytic solutions")
    failures = []
    for check in (
        check_1d_harmonic,
        check_2d_coupled,
        check_null_space_is_ignored_not_blown_up,
        check_electronic_only_differs_from_total,
    ):
        check(failures)

    for f_ in failures:
        print(f"  - {f_}")
    if failures:
        print("[FAILED] test_16_implicit")
        return 1
    print("[OK]  implicit-chain combination matches analytic solutions, "
          "including the sign of the IFT and pinv's null-space handling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
