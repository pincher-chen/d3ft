"""Bisection diagnostic: which D3FT module breaks Si ground-state energy?

Runs the Jrystal baseline, then D3FT under all four (optim, occupation) toggles,
and prints a comparison table. The all-electron Si reference energy should be a
large negative number (~ -283 Ha at this low cutoff); a positive energy signals
a broken gradient/occupation path.
"""
import os
import time

from absl import logging

logging.set_verbosity(logging.WARNING)

D3FT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _base_cfg():
    from d3ft.config import load_config
    cfg = load_config(os.path.join(D3FT_DIR, "config.yaml"))
    cfg.crystal = "si"
    cfg.xc = "lda_x+lda_c_pw"
    cfg.use_pseudopotential = False
    cfg.cutoff_energy = 20
    cfg.grid_sizes = 12
    cfg.k_grid_sizes = 1
    cfg.epoch = 500
    cfg.convergence_condition = 1e-4
    cfg.verbose = False
    cfg.jax_enable_x64 = True
    cfg.jax_debug_nans = False
    cfg.d3ft.occupation.epsilon_init = 1.0
    cfg.d3ft.occupation.epsilon_final = 1e-3
    return cfg


def run_jrystal():
    from jrystal.config import get_config
    from jrystal.calc.calc_ground_state_energy_all_electrons import calc
    cfg = get_config()
    cfg.crystal = "si"
    cfg.xc = "lda_x+lda_c_pw"
    cfg.use_pseudopotential = False
    cfg.cutoff_energy = 20
    cfg.grid_sizes = 12
    cfg.k_grid_sizes = 1
    cfg.occupation = "simplex-projector"
    cfg.epoch = 500
    cfg.convergence_condition = 1e-4
    cfg.verbose = False
    cfg.jax_enable_x64 = True
    cfg.jax_debug_nans = False
    t0 = time.time()
    out = calc(cfg)
    return float(out.total_energy), time.time() - t0


def run_d3ft(optim_on, occ_on, free_energy=False, lr=None):
    from d3ft.calc.ground_state import calc
    cfg = _base_cfg()
    cfg.occupation = "uniform" if occ_on else "simplex-projector"
    cfg.d3ft.optim.enabled = optim_on
    cfg.d3ft.occupation.enabled = occ_on
    cfg.d3ft.free_energy.enabled = free_energy
    if lr is not None:
        cfg.optimizer_args.learning_rate = lr
    t0 = time.time()
    out = calc(cfg)
    return out.total_energy, len(out.energy_history), time.time() - t0


if __name__ == "__main__":
    print("=== D3FT module bisection on Si (all-electron, LDA, gamma) ===\n")

    e_jr, t_jr = run_jrystal()
    print(f"[jrystal baseline]            E = {e_jr:+.4f} Ha   t={t_jr:.1f}s\n")

    combos = [
        (False, False, "optim=OFF occ=OFF  (pure Euclidean Adam + simplex)"),
        (True,  False, "optim=ON  occ=OFF  (Riemannian Adam + simplex)"),
        (False, True,  "optim=OFF occ=ON   (Euclidean Adam + Sinkhorn)"),
        (True,  True,  "optim=ON  occ=ON   (full D3FT)"),
    ]
    print(f"{'configuration':<52}{'E (Ha)':>14}{'dE vs jr (Ha)':>16}{'steps':>8}{'t(s)':>8}")
    print("-" * 98)
    for optim_on, occ_on, label in combos:
        try:
            e, steps, t = run_d3ft(optim_on, occ_on)
            print(f"{label:<52}{e:>+14.4f}{e - e_jr:>+16.4f}{steps:>8}{t:>8.1f}")
        except Exception as exc:
            print(f"{label:<52}  FAILED: {exc}")
    print("-" * 98)
    print("\nInterpretation: the row closest to dE~0 is the correct path; a large")
    print("positive E isolates the broken module.")

    # --- Free-energy matched comparison: same objective (F = E - T*S) and same LR (0.01,
    # jrystal's default) as the baseline, so optim=OFF/occ=OFF is a genuine apples-to-apples
    # comparison against jrystal (pure optimizer comparison under identical physics). ---
    print("\n=== Free-energy matched objective (F = E - T*S, lr=0.01 as jrystal) ===\n")
    print(f"{'configuration':<52}{'E (Ha)':>14}{'dE vs jr (Ha)':>16}{'steps':>8}{'t(s)':>8}")
    print("-" * 98)
    for optim_on, label in [
        (False, "free-E  optim=OFF occ=OFF  (Euclidean Adam, matched)"),
        (True,  "free-E  optim=ON  occ=OFF  (Riemannian Adam, matched)"),
    ]:
        try:
            e, steps, t = run_d3ft(optim_on, False, free_energy=True, lr=0.01)
            print(f"{label:<52}{e:>+14.4f}{e - e_jr:>+16.4f}{steps:>8}{t:>8.1f}")
        except Exception as exc:
            print(f"{label:<52}  FAILED: {exc}")
    print("-" * 98)
    print("\nThe matched optim=OFF row should track the jrystal baseline closely (same")
    print("objective + LR); the optim=ON row isolates the pure Riemannian-optimizer gain.")
