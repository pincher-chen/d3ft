# D3FT

**Direct Differentiable Density Functional Theory**

D3FT is a downstream package built on top of [`jrystal`](https://github.com/sail-sg/jrystal). It contributes three
new algorithmic modules for direct-optimization plane-wave DFT:

| Module | What it does | Replaces in Jrystal |
|---|---|---|
| `d3ft.optim` (**D3FT-Riem**) | Kinetic-preconditioned Riemannian Adam on the Stiefel manifold | Euclidean Adam over QR-lifted parameters |
| `d3ft.occupation` (**D3FT-OT**) | Capacity-bounded Sinkhorn-OT occupations with principled annealing | `simplex_projector` + exponential annealing |
| `d3ft.spectrum` (**D3FT-Lanczos**) | Differentiable stochastic Lanczos for band structures and forces | `complex_hessian` serial Python loop |
| `d3ft.calc.stress` | Stress tensor as `(1/Ω) dF/dε` by direct differentiation of the Mermin free energy | not available |
| `d3ft.pseudopotential` | Norm-conserving pseudopotentials: correct Kleinman-Bylander conjugation, and projectors evaluated inside the JAX graph so gradients reach the cell | `jrystal.pseudopotential` |

> D3FT is unrelated to Grimme's **DFT-D3** dispersion correction.

See [`DESIGN.md`](DESIGN.md) for architectural details.

---

## Install

```bash
# 1. jrystal is a runtime dependency and is not vendored here
git clone https://github.com/sail-sg/jrystal.git
pip install -e ./jrystal

# 2. Install d3ft (this package)
git clone https://github.com/<owner>/d3ft.git
pip install -e ./d3ft
```

Requires JAX with GPU support (developed against JAX/jaxlib 0.6.2, Python 3.10).

## Run

```bash
# Ground-state energy minimization
d3ft -m energy -c config.yaml

# Band structure (requires a ground-state run first)
d3ft -m band -c config.yaml

# Ablation: disable a module
d3ft -m energy -c config.yaml --d3ft.optim.enabled=False    # baseline Adam
d3ft -m energy -c config.yaml --d3ft.occupation.enabled=False  # use jrystal's simplex projector
```

## Tests

```bash
bash tests/run_tests.sh cpu   # levels 0-3, no GPU needed
bash tests/run_tests.sh all   # adds the Si integration and QE parity tests
```

Two tests guard pseudopotential correctness and are worth knowing about:

- `tests/test_07_alp_qe_parity.py` — frozen-density Kohn-Sham spectra for AlP and
  GaAs against Quantum ESPRESSO 7.5 references. Checks the band-edge locations and
  gaps, so it catches errors that leave the gap plausible but the physics wrong.
  Needs a curated reference data package; skips itself without one.
- `tests/test_10_stress.py` — the stress must equal the strain derivative of the
  free energy. It also reports `max |dF/d(params)|`: the Hellmann-Feynman step
  behind the stress assumes a stationary point, and a finite-difference check
  validates the derivative, not the state it is taken at.
- `tests/test_08_translation_invariance.py` — the KS spectrum must not change when
  the whole crystal is rigidly translated. Needs no reference data and holds for
  any system, so it is the general guard for local/nonlocal pseudopotential terms
  being placed at inconsistent positions.

## Validation

`validation/` holds the reference data and drivers behind the pseudopotential
tests: Quantum ESPRESSO 7.5 references for two zincblende binaries with opposite
band-gap character, plus the QE inputs, pseudopotentials and plotting scripts.

| | AlP | GaAs |
|---|---|---|
| Pseudopotentials / functional | ONCVPSP (PseudoDojo), PBE | BHS, LDA-PZ |
| Gap character | indirect, CBM at X | direct, CBM at Γ |
| QE indirect gap | 1.560 eV | 0.846 eV |
| d3ft, frozen QE density | 1.571 eV | 0.885 eV |
| d3ft, self-consistent | 1.786 eV | 0.962 eV |

Agreement over all bands at L, Γ and X on the frozen QE density is 0.015 eV (AlP)
and 0.039 eV (GaAs). See [`validation/README.md`](validation/README.md).

![AlP bands](validation/results/alp_bands_vs_qe.png)

## Repository layout

See [`DESIGN.md`](DESIGN.md) §2 for the full tree.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Acknowledgements

D3FT builds on [jrystal](https://github.com/sail-sg/jrystal)
(Copyright 2025 Garena Online Private Limited, Apache-2.0), which provides the
plane-wave basis, pseudopotential handling and crystal/grid utilities. jrystal is
a runtime dependency and its source is not redistributed here; see
[`NOTICE`](NOTICE) for details.
