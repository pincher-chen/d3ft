#!/usr/bin/env python3
"""Matched-initial-state 2x2 metallic pilot, pending physical KKT validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from d3ft.config import load_config
from d3ft.calc.ground_state import calc
from d3ft.grid.kpoints import k_vectors
from d3ft.parallel import plain_fft_patch
from d3ft.occupation.sinkhorn import sinkhorn_occupation
from jrystal._src import entropy, occupation as jr_occ, pw
from jrystal._src.grid import proper_grid_size
from jrystal._src.unitary_module import unitary_matrix
from jrystal.calc.opt_utils import create_crystal, create_freq_mask

ROOT = Path(os.environ.get("D3FT_RUN_ROOT", Path(__file__).resolve().parents[1]))
BASE = ROOT / "inputs/reference_config_base_snapshot.yaml"


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--material", choices=("Al", "Na", "Mg", "Si", "SiC", "GaAs"), required=True)
    p.add_argument("--orbital", choices=("qr", "postkp"), required=True)
    p.add_argument("--occupation", choices=("idempotent", "capped", "fd"), required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--occupation-lr", type=float)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--lr-final", type=float, default=1e-7)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--empty-bands", type=int, default=4)
    p.add_argument("--initial-temperature", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--anneal-steps", type=int, default=3000)
    p.add_argument("--preconditioner", choices=("kinetic","none"), default="kinetic")
    p.add_argument("--structure", type=Path)
    p.add_argument("--occupation-temperature", type=float)
    p.add_argument("--entropy-eps", type=float, default=1e-8)
    p.add_argument("--metric-orbitals", type=int, choices=(0,1), required=True)
    p.add_argument("--metric-occupations", type=int, choices=(0,1), required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--endpoint-check", action="store_true")
    p.add_argument("--restart-from", type=Path)
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--record-work", action="store_true")
    a = p.parse_args()
    if not 0 < a.temperature <= a.initial_temperature:
        raise ValueError("Initial temperature must be at least the positive target")
    if a.occupation_temperature is not None and a.occupation_temperature <= 0:
        raise ValueError("Occupation coordinate temperature must be positive")
    if a.orbital == "qr" and a.occupation_lr is not None and a.occupation_lr != a.lr:
        raise ValueError("Frozen QR branch uses a joint optimizer; separate occupation LR is unsupported")
    if a.orbital == "qr" and a.lr_schedule != "constant":
        raise ValueError("Frozen QR branch does not support this postKP learning-rate schedule")
    occ_lr = a.lr if a.occupation_lr is None else a.occupation_lr
    if a.lr_schedule != "constant" and not 0 < a.lr_final <= min(a.lr, occ_lr):
        raise ValueError("Final LR must be positive and no greater than either initial LR")
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == "gpu", devices
    ident = f"matched_{a.material}_{a.orbital}_{a.occupation}_lr{a.lr:g}_s{a.seed}"
    if a.tag:
        ident += f"_{a.tag}"
    if a.grid != 64:
        ident += f"_g{a.grid}"
    if a.empty_bands != 4:
        ident += f"_eb{a.empty_bands}"
    result = ROOT / "results" / ident
    result.mkdir(parents=True, exist_ok=False)
    (result / "driver_snapshot.py").write_bytes(Path(__file__).read_bytes())
    manifest = json.loads((ROOT / "inputs" / "manifest.json").read_text())[a.material]
    cfg = load_config(str(BASE))
    cfg.crystal = None
    structure=a.structure.resolve() if a.structure else ROOT / "inputs" / "structures" / f"{a.material}.xyz"
    cfg.crystal_file_path_path = str(structure)
    manifest=dict(manifest, structure_path=str(structure), structure_sha256=hashlib.sha256(structure.read_bytes()).hexdigest())
    cfg.spin = 0
    cfg.spin_restricted = True
    cfg.k_grid_sizes = manifest["kmesh"]
    cfg.grid_sizes = a.grid
    cfg.cutoff_energy = manifest["cutoff_Ha"]
    cfg.xc = manifest["xc"]
    cfg.empty_bands = a.empty_bands
    cfg.occupation = "idempotent"
    cfg.smearing = a.temperature
    cfg.epoch = a.epochs
    cfg.seed = a.seed
    cfg.pseudopotential_file_dir = str(ROOT / "inputs" / "pseudopotentials") + "/"
    cfg.convergence_condition = 0.0
    cfg.xla_preallocate = False
    if os.environ.get("JCTC_DEBUG_NANS") == "1":
        cfg.jax_debug_nans = True
    cfg.verbose = False
    if not 0 < a.entropy_eps <= 1e-8:
        raise ValueError("Entropy EPS must be positive and no larger than 1e-8")
    cfg.eps = a.entropy_eps
    cfg.d3ft.optim.metric_trial = True
    cfg.d3ft.optim.record_work = a.record_work
    cfg.d3ft.optim.metric_orbitals = bool(a.metric_orbitals)
    cfg.d3ft.optim.metric_occupations = bool(a.metric_occupations)
    cfg.optimizer_args.learning_rate = a.lr
    cfg.d3ft.optim.enabled = a.orbital == "postkp"
    cfg.d3ft.optim.init = "random"
    cfg.d3ft.optim.preconditioner = a.preconditioner if a.orbital == "postkp" else "none"
    cfg.d3ft.optim.lr_schedule = a.lr_schedule
    cfg.d3ft.optim.lr_final = a.lr_final
    cfg.d3ft.occupation.enabled = a.occupation in ("capped", "fd")
    map_temperature = (a.occupation_temperature or cfg.smearing) if a.occupation == "fd" else 0.03
    cfg.d3ft.occupation.epsilon_init = map_temperature
    cfg.d3ft.occupation.epsilon_final = map_temperature
    cfg.d3ft.occupation.epsilon_schedule = "constant"
    cfg.d3ft.occupation.learning_rate = a.lr if a.occupation_lr is None else a.occupation_lr
    cfg.d3ft.free_energy.enabled = True
    cfg.d3ft.free_energy.temp_init = a.initial_temperature
    cfg.d3ft.free_energy.anneal_steps = a.anneal_steps
    cfg.d3ft.kpoints.mesh = "gamma"
    cfg.d3ft.parallel.k_pool = False
    cfg.d3ft.xc_grid_factor = 1
    if os.environ.get("JCTC_RHO_DEBUG") == "1":
        from jrystal._src import xc as xc_module
        original_lda = xc_module._lda

        def traced_lda(xc_type, density):
            jax.debug.print("RHO_DIAG min={lo} max={hi} finite={ok}",
                            lo=jnp.min(density), hi=jnp.max(density),
                            ok=jnp.all(jnp.isfinite(density)))
            return original_lda(xc_type, density)

        xc_module._lda = traced_lda
    if os.environ.get("JCTC_CORE_DEBUG") == "1":
        from jrystal._src import energy as energy_module
        original_xc_energy = energy_module.xc_energy

        def traced_xc_energy(density, g_vector_grid, vol, xc_type,
                             kohn_sham=False, core_density_grid=None):
            if core_density_grid is not None:
                jax.debug.print("CORE_DIAG val_min={v} core_min={c} total_min={t}",
                    v=jnp.min(density), c=jnp.min(core_density_grid),
                    t=jnp.min(density + core_density_grid[None, ...] / density.shape[0]))
            return original_xc_energy(density, g_vector_grid, vol, xc_type,
                                      kohn_sham=kohn_sham,
                                      core_density_grid=core_density_grid)

        energy_module.xc_energy = traced_xc_energy
    crystal = create_crystal(cfg)
    mask = create_freq_mask(cfg)
    n_k = len(k_vectors(crystal.cell_vectors, proper_grid_size(cfg.k_grid_sizes), "gamma"))
    ne = manifest["valence_electrons"]
    n_band = ne // 2 + cfg.empty_bands
    raw = pw.param_init(jax.random.PRNGKey(a.seed), n_band, n_k, mask, True)
    physical = unitary_matrix(raw, complex=True)
    idem = jr_occ.param_init(jax.random.PRNGKey(a.seed + 10000),
                             n_band, ne, n_k, 0, "idempotent", True)
    initial_f = jr_occ.occupation(idem, n_k, ne, 0, "idempotent", True)
    if a.occupation == "fd":
        from d3ft.occupation.fd import fermi_dirac_occupation
        occupation_map = fermi_dirac_occupation
        levels = -map_temperature * jnp.log(initial_f / (2/n_k - initial_f))
    else:
        occupation_map = sinkhorn_occupation
        levels = -map_temperature * jnp.log(initial_f)
    init = {"pw": raw if a.orbital == "qr" else physical,
            "occ": idem if a.occupation == "idempotent" else {"band_energies": levels}}
    restart_sha = None
    if a.restart_from is not None:
        checkpoint = a.restart_from.resolve()
        restart_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with np.load(checkpoint, allow_pickle=True) as saved:
            occ_state = {}
            pw_state = {}
            for key in saved.files:
                if key.startswith(("occ__", "pw__")):
                    path = key.split("__")[1:]
                    target = occ_state if key.startswith("occ__") else pw_state
                    for component in path[:-1]:
                        target = target.setdefault(component, {})
                    target[path[-1]] = jnp.asarray(saved[key])
                elif key.startswith("occ_"):
                    # Read only our older campaign checkpoints, whose nested
                    # Jrystal idempotent dictionaries became NumPy objects.
                    value = saved[key]
                    if value.dtype == object and value.shape == ():
                        value = value.item()
                    occ_state[key.removeprefix("occ_")] = jax.tree.map(jnp.asarray, value)
            if pw_state:
                saved_pw = pw_state
            else:
                saved_pw = saved["params_pw"]
                if saved_pw.dtype == object and saved_pw.shape == ():
                    saved_pw = saved_pw.item()
                saved_pw = jax.tree.map(jnp.asarray, saved_pw)
            init = {"pw": saved_pw, "occ": occ_state}
        if not init["occ"]:
            raise ValueError(f"No occupation parameters in {checkpoint}")
        physical = (unitary_matrix(init["pw"], complex=True)
                    if a.orbital == "qr" else init["pw"])
        if a.occupation == "idempotent":
            initial_f = jr_occ.occupation(init["occ"], n_k, ne, 0,
                                           "idempotent", True)
        else:
            initial_f = occupation_map(init["occ"]["band_energies"],
                                       ne, map_temperature, True, 50)
    elif a.occupation == "fd":
        reconstructed_f = occupation_map(levels, ne, map_temperature, True, 50)
        if float(jnp.max(jnp.abs(reconstructed_f-initial_f))) > 1e-10:
            raise RuntimeError("FD initial-state occupation parity failed")
    init_hashes = {"physical_orbital_sha256": digest(physical),
                   "occupation_sha256": digest(initial_f),
                   "restart_state_sha256": restart_sha}
    (result / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + "\n")
    serializable_args = {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()}
    (result / "provenance.json").write_text(json.dumps({"arguments": serializable_args,
        "source": os.environ.get("JCTC_FROZEN_SOURCE"),
        "source_tree_sha256": os.environ.get("JCTC_CANDIDATE_TREE_SHA256"),
        "jrystal_archive_sha256": os.environ.get("JCTC_FROZEN_JRYSTAL_ARCHIVE_SHA256"),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input": manifest, "initial": init_hashes, "jax": jax.__version__,
        "device": str(devices[0]), "job": os.environ.get("SLURM_JOB_ID"),
        "step": os.environ.get("SLURM_STEP_ID"), "cpu_affinity": sorted(os.sched_getaffinity(0)), "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}, indent=2) + "\n")
    import gc
    warm_seconds = 0.0
    if not a.skip_warmup:
        cfg.d3ft.free_energy.timing_warmup_only = True
        warm_start = time.perf_counter()
        with plain_fft_patch():
            warm_out = calc(cfg, init_params=init, freq_mask=mask)
        warm_seconds = time.perf_counter()-warm_start
        del warm_out
        gc.collect()
    cfg.d3ft.free_energy.timing_warmup_only = False
    from timing_observer import TimingObserver
    timing_observer = TimingObserver(result)
    timing_observer.install()
    observer = None
    if os.environ.get('JCTC_MECHANISM_OBSERVER') == '1':
        from mechanism_observer import Observer
        observer = Observer(result, cfg, init, mask)
        observer.install()
    t0 = time.perf_counter()
    with plain_fft_patch():
        out = calc(cfg, init_params=init, freq_mask=mask)
    elapsed = time.perf_counter() - t0
    if observer is not None:
        observer.uninstall()
    timing_observer.finish()
    final_f = np.asarray(out.get_occupation(out.params_occ, out.final_eps))
    ent = float(entropy.fermi_dirac(final_f, eps=cfg.eps))
    E = float(out.total_energy)
    # Jrystal's entropy uses weighted f directly. It includes the nonzero
    # empty-state constant -sum(c log c); subtract that constant for comparison
    # with QE's physical F=E-TS. The constant does not affect optimisation.
    cap = 2 / n_k
    entropy_constant = -final_f.size * cap * float(np.log(cap))
    physical_entropy = ent - entropy_constant
    summary = {"id": ident, "scope": "matched pilot, not KKT-certified",
        "iterations": len(out.energy_history), "energy_Ha": E,
        "free_energy_Ha": E - float(out.temperature)*ent,
        "free_energy_physical_Ha": E - float(out.temperature)*physical_entropy,
        "jrystal_entropy_constant": entropy_constant,
        "temperature_Ha": float(out.temperature), "charge": float(final_f.sum()),
        "expected_electrons": ne, "minimum_f": float(final_f.min()),
        "maximum_f": float(final_f.max()), "capacity": cap,
        "wall_seconds": elapsed, "initial": init_hashes, "warmup_seconds": warm_seconds, "warmup_includes_compile_and_one_step": not a.skip_warmup, "cold_first_solve": a.skip_warmup, "timing_scope": "Cold process solve including compilation; sampled fixed-budget endpoint, not exact first-hit time. Endpoint/spectral checks separately recorded."}
    summary["device_allocator_memory_stats_after_solve"] = devices[0].memory_stats()
    np.save(result / "energy_history.npy", np.asarray(out.energy_history))
    np.save(result / "final_occupation.npy", final_f)
    if a.endpoint_check:
        from endpoint_residuals import evaluate
        with plain_fft_patch():
            endpoint = evaluate(out, cfg)
        summary["last_recorded_energy_Ha"] = summary["energy_Ha"]
        summary["energy_Ha"] = endpoint["final_state_energy_Ha"]
        summary["free_energy_physical_Ha"] = endpoint["final_state_free_energy_physical_Ha"]
        summary["free_energy_Ha"] = summary["energy_Ha"] - float(out.temperature)*ent
        (result / "endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
        endpoint_source = Path(__file__).with_name("endpoint_residuals.py")
        (result / "endpoint_snapshot.py").write_bytes(endpoint_source.read_bytes())
        print(json.dumps({"endpoint": endpoint}), flush=True)
        arrays = {"density_grid": np.asarray(out.density_grid)}
        if out.core_density_grid is not None:
            arrays["core_density_grid"] = np.asarray(out.core_density_grid)
        def add_occ(prefix, value):
            if isinstance(value, dict):
                for key, child in value.items():
                    add_occ(f"{prefix}__{key}", child)
            else:
                arrays[prefix] = np.asarray(value)

        add_occ("occ", out.params_occ)
        if isinstance(out.params_pw, dict):
            add_occ("pw", out.params_pw)
        else:
            arrays["params_pw"] = np.asarray(out.params_pw)
        np.savez_compressed(result / "final_state.npz", **arrays)
    (result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if (not all(np.isfinite(summary[key]) for key in
                ("energy_Ha", "free_energy_Ha", "charge", "minimum_f", "maximum_f"))
            or abs(summary["charge"] - ne) > 1e-8
            or summary["minimum_f"] < -1e-12
            or summary["maximum_f"] > 2/n_k + 1e-10):
        raise RuntimeError("Feasibility failed")


if __name__ == "__main__":
    main()
