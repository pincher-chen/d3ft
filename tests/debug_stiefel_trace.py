"""Diagnose the D3FT-Riem (optim=ON) path: trace energy vs learning rate.

If the step-0 energy matches the Euclidean path but later steps blow up, the issue
is optimiser step-size on the Stiefel manifold (motivating the kinetic preconditioner),
not the parameterisation.
"""
import os
import time

from absl import logging

logging.set_verbosity(logging.ERROR)

D3FT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_cfg():
    from d3ft.config import load_config
    cfg = load_config(os.path.join(D3FT_DIR, "config.yaml"))
    cfg.crystal = "si"
    cfg.xc = "lda_x+lda_c_pw"
    cfg.use_pseudopotential = False
    cfg.cutoff_energy = 20
    cfg.grid_sizes = 12
    cfg.k_grid_sizes = 1
    cfg.epoch = 200
    cfg.convergence_condition = 1e-9   # disable early stop, see full trace
    cfg.verbose = False
    cfg.jax_enable_x64 = True
    cfg.occupation = "simplex-projector"
    cfg.d3ft.occupation.enabled = False     # isolate the optimiser
    cfg.d3ft.optim.enabled = True
    return cfg


def run(lr):
    from d3ft.calc.ground_state import calc
    cfg = base_cfg()
    cfg.optimizer_args.learning_rate = lr
    t0 = time.time()
    out = calc(cfg)
    h = out.energy_history
    return h, time.time() - t0


if __name__ == "__main__":
    print("=== D3FT-Riem lr sweep (Si, optim=ON occ=OFF, no preconditioner) ===\n")
    print(f"{'lr':>10}{'E[0]':>14}{'E[min]':>14}{'E[final]':>14}{'t(s)':>8}")
    print("-" * 60)
    for lr in [0.05, 0.01, 0.003, 0.001, 0.0003]:
        try:
            h, t = run(lr)
            print(f"{lr:>10}{h[0]:>14.4f}{min(h):>14.4f}{h[-1]:>14.4f}{t:>8.1f}")
        except Exception as exc:
            print(f"{lr:>10}  FAILED: {exc}")
    print("-" * 60)
    print("(reference: jrystal -282.78 Ha; Euclidean Adam -287.9 Ha)")
