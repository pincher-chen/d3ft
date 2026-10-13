#!/usr/bin/env python3
"""Level 4 regression: AlP frozen-density KS spectrum must match QE 7.5.

This is the regression guard for the Kleinman-Bylander conjugation bug that made
every binary (two-species) system wrong: the projectors of an atom at R were
effectively placed at -R, so for the zincblende anion at a/4(1,1,1) -- where 2R
is not a lattice vector -- the deep state bound by that atom's local potential
never saw the KB repulsion. It showed up as a ~-42.7 eV state that was invariant
under scaling the whole nonlocal term, and it put the CBM at Gamma instead of X.

The test drives the curated AlP parity probe with QE's own converged density, so
it isolates Hamiltonian assembly from the direct-optimization density. It needs
the curated data package (pseudopotentials + QE reference) and is skipped when
that is not present.

Run:
  python tests/test_07_alp_qe_parity.py
  D3FT_ALP_ROOT=/path/to/d3ft python tests/test_07_alp_qe_parity.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

# Reference values from QE 7.5 on the curated data package, frozen-density probe.
# `cbm_at` is the k-label QE puts the CBM on; getting this wrong is the primary
# symptom of the KB conjugation bug (it drags the CBM to Gamma for AlP and to L
# for GaAs). `ghost_floor` is system-specific: the spurious deep state only shows
# up where the local potential is deep enough to bind one (AlP's dojo Al/P do,
# GaAs's BHS Ga/As do not), which is why it cannot be the only guard --
# test_08_translation_invariance.py is the system-independent one.
SYSTEMS = {
    "AlP": dict(
        sp1="Al", sp2="P", a0="5.467", xc="gga_x_pbe+gga_c_pbe",
        pp="alp", qe_dir="reference/qe_alp", qe_bands="alp.bands.dat",
        qe_cube="alp.rho.cube",
        indirect_gap=1.560, gamma_gap=3.246, cbm_at="X", ghost_floor=-25.0,
    ),
    "GaAs": dict(
        sp1="Ga", sp2="As", a0="5.653", xc="lda_x+lda_c_pz",
        pp="gaas", qe_dir="reference/qe_gaas", qe_bands="gaas.bands.dat",
        qe_cube="gaas.rho.cube",
        indirect_gap=0.846, gamma_gap=0.846, cbm_at="G", ghost_floor=-25.0,
    ),
}
GAP_TOL = 0.15          # eV; direct-optimization/basis differences, not a bug


def _root():
    """Directory holding the curated validation data package.

    Order: the D3FT_ALP_ROOT override, then this repository's ``validation/``,
    then the cluster-style ``<root>/software/d3ft/tests/`` layout.
    """
    env = os.environ.get("D3FT_ALP_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    repo_validation = here.parents[1] / "validation"
    if repo_validation.is_dir():
        return repo_validation
    return here.parents[3]


def check_system(name, spec, root):
    probe = root / "workflows" / "alp" / "_alp_qe_parity_probe.py"
    qe_dir = root / spec["qe_dir"]
    cube = qe_dir / spec["qe_cube"]
    bands = qe_dir / spec["qe_bands"]
    pp_dir = root / "pseudopotentials" / spec["pp"]
    upfs = [pp_dir / f"{spec[k]}.upf" for k in ("sp1", "sp2")]

    missing = [p for p in [probe, cube, bands, *upfs] if not p.exists()]
    if missing:
        print(f"[SKIP] {name}: reference data not found ({missing[0]})")
        return None

    env = dict(os.environ)
    env.update(
        D3FT_ALP_ROOT=str(root),
        DENSITY="qe_cube",
        PRECALC_BETA_SERIAL="1",
        SYSTEM=name,
        SP1=spec["sp1"], SP2=spec["sp2"], A0=spec["a0"], XC=spec["xc"],
        PP_DIR=str(pp_dir),
        QE_BANDS=str(bands), QE_CUBE=str(cube),
    )
    print(f"[RUN] {name} frozen-QE-density parity probe (a few minutes)")
    proc = subprocess.run(
        [sys.executable, str(probe)],
        cwd=str(probe.parent), env=env,
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out[-4000:])
        print(f"  probe exited nonzero for {name}")
        return [f"{name}: probe exited nonzero"]

    m = re.search(
        r"jr VBM=(-?[\d.]+)@(\w+)\s+CBM=(-?[\d.]+)@(\w+)\s+gap=(-?[\d.]+)", out
    )
    if not m:
        print(out[-4000:])
        return [f"{name}: could not parse the jr extrema line"]
    _, vbm_k, _, cbm_k, gap = m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(5))

    lowest = min(
        float(v)
        for line in re.findall(r"jr raw lowest8=\[([^\]]+)\]", out)
        for v in line.split()
    )

    gm = re.search(r"^G:.*?jr VBM=(-?[\d.]+) CBM=(-?[\d.]+) gap=(-?[\d.]+)",
                   out, re.S | re.M)
    gamma_gap = float(gm.group(3)) if gm else float("nan")

    print(f"  VBM @ {vbm_k}   CBM @ {cbm_k}   (QE CBM @ {spec['cbm_at']})")
    print(f"  indirect gap = {gap:.4f} eV   (QE {spec['indirect_gap']:.3f})")
    print(f"  Gamma gap    = {gamma_gap:.4f} eV   (QE {spec['gamma_gap']:.3f})")
    print(f"  lowest state = {lowest:.3f} eV   (floor {spec['ghost_floor']})")

    failures = []
    if vbm_k != "G":
        failures.append(f"{name}: VBM at {vbm_k}, expected Gamma")
    if cbm_k != spec["cbm_at"]:
        failures.append(
            f"{name}: CBM at {cbm_k}, expected {spec['cbm_at']} "
            "(the KB conjugation bug moves it)"
        )
    if abs(gap - spec["indirect_gap"]) > GAP_TOL:
        failures.append(
            f"{name}: indirect gap {gap:.4f} eV differs from QE "
            f"{spec['indirect_gap']} by more than {GAP_TOL} eV"
        )
    if gamma_gap == gamma_gap and abs(gamma_gap - spec["gamma_gap"]) > GAP_TOL:
        failures.append(
            f"{name}: Gamma gap {gamma_gap:.4f} eV differs from QE "
            f"{spec['gamma_gap']} by more than {GAP_TOL} eV"
        )
    if lowest < spec["ghost_floor"]:
        failures.append(
            f"{name}: spurious deep state at {lowest:.3f} eV "
            f"(below {spec['ghost_floor']}); KB projectors are not reaching that atom"
        )
    for f in failures:
        print(f"  - {f}")
    if not failures:
        print(f"  [OK] {name} matches the QE reference")
    return failures


def main():
    root = _root()
    only = os.environ.get("PARITY_SYSTEMS")
    names = only.split(",") if only else list(SYSTEMS)

    all_failures, ran = [], 0
    for name in names:
        spec = SYSTEMS.get(name.strip())
        if spec is None:
            print(f"[SKIP] unknown system {name!r}")
            continue
        res = check_system(name.strip(), spec, root)
        if res is None:
            continue
        ran += 1
        all_failures.extend(res)

    if ran == 0:
        print("[SKIP] no reference data available for any system")
        return 0
    if all_failures:
        print(f"[FAILED] test_07_alp_qe_parity ({len(all_failures)} problem(s))")
        return 1
    print(f"[OK]  {ran} system(s) match the QE reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
