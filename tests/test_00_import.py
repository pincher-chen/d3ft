"""Level-0 smoke test: pure Python imports, no GPU needed.

Run on the login node:
    python tests/test_00_import.py
"""
import sys


def test_jrystal_import():
    import jrystal
    print(f"  jrystal {jrystal.__version__}  @ {jrystal.__file__}")


def test_d3ft_import():
    import d3ft
    print(f"  d3ft {d3ft.__version__}  @ {d3ft.__file__}")


def test_submodules():
    from d3ft import calc, config, occupation, optim, spectrum, utils
    print("  d3ft.optim      OK")
    print("  d3ft.occupation OK")
    print("  d3ft.spectrum   OK")
    print("  d3ft.calc       OK")
    print("  d3ft.config     OK")
    print("  d3ft.utils      OK")


def test_stiefel_import():
    from d3ft.optim.stiefel import project_tangent, retract_qr
    from d3ft.optim.preconditioner import kinetic_preconditioner
    from d3ft.optim.riemannian_adam import riemannian_adam
    print("  D3FT-Riem ops   OK")


def test_occupation_import():
    from d3ft.occupation.sinkhorn import sinkhorn_occupation, epsilon_schedule
    print("  D3FT-OT ops     OK")


def test_spectrum_import():
    from d3ft.spectrum.lanczos import lanczos_tridiag, lanczos_eigvals
    from d3ft.spectrum.implicit import eigvals_with_grad
    print("  D3FT-Lanczos ops OK")


def test_config_load():
    import os
    from d3ft.config import load_config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    cfg = load_config(cfg_path)
    assert cfg.crystal == "si", f"Expected 'si', got {cfg.crystal!r}"
    assert hasattr(cfg, "d3ft"), "Config missing 'd3ft' key"
    assert cfg.d3ft.optim.enabled is True
    assert cfg.d3ft.occupation.enabled is True
    print(f"  config OK  (crystal={cfg.crystal}, d3ft.optim.enabled={cfg.d3ft.optim.enabled})")


TESTS = [
    test_jrystal_import,
    test_d3ft_import,
    test_submodules,
    test_stiefel_import,
    test_occupation_import,
    test_spectrum_import,
    test_config_load,
]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            print(f"[RUN] {fn.__name__}")
            fn()
            print(f"[OK]  {fn.__name__}\n")
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}\n")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(TESTS)} tests FAILED")
        sys.exit(1)
    else:
        print(f"All {len(TESTS)} tests passed.")
