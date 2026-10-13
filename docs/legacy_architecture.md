# D3FT — Design and Implementation Plan

> **D3FT** — *Direct Differentiable Density Functional Theory*
> A downstream package built on top of [`jrystal`](https://github.com/sail-sg/jrystal), providing three new
> algorithmic modules for direct optimization of plane-wave DFT.

---

## 1. Relationship to Jrystal

D3FT does **not** fork or copy Jrystal's source code. Instead, it installs Jrystal as a
runtime dependency and replaces only the three algorithmic components that differ:

```
d3ft  ──depends on──►  jrystal
                       (plane-wave basis, energies, Ewald, XC, FFT, pseudopotentials)
```

| Subsystem | Where it lives |
|---|---|
| Plane-wave basis, FFT grids, G/k vectors | `jrystal._src.grid`, `jrystal._src.pw` |
| Energy components (kinetic, Hartree, external, XC) | `jrystal._src.energy` |
| Ewald sum, exchange-correlation, FFT (SPMD) | `jrystal._src.ewald`, `jrystal._src.xc`, `jrystal._src.spmd` |
| Pseudopotentials, SBT | `jrystal.pseudopotential`, `jrystal.sbt` |
| Crystal structure loader | `jrystal._src.crystal.Crystal` |
| **Optimizer (NEW)** | `d3ft.optim` — replaces `optax.adam` |
| **Occupation (NEW)** | `d3ft.occupation` — replaces `jrystal.occupation.simplex_projector` |
| **Spectrum / Hessian (NEW)** | `d3ft.spectrum` — replaces `jrystal._src.hessian.complex_hessian` |
| **Calculation orchestrator (NEW)** | `d3ft.calc` — analogue of `jrystal.calc` |

### Why not fork?

1. **Clean baseline.** When the ICML paper compares D3FT against Jrystal, both are running
   the *same* plane-wave / energy / FFT code—only the three replaced modules differ. This
   isolates the algorithmic contribution from incidental engineering differences.
2. **Easy upstream sync.** If Jrystal is updated, we just bump the dependency pin.
3. **Smaller surface area.** D3FT v0.1 contains ~10–15 source files, all directly
   corresponding to algorithmic claims in the ICML paper.

### Data files

Geometry (`*.xyz`) and pseudopotential (`*.UPF`) files are not duplicated. The D3FT CLI
resolves them through `jrystal.get_pkg_path()`, so both packages share the same data.

---

## 2. Directory Layout

```
d3ft/
├── README.md
├── DESIGN.md                  # this file
├── setup.py
├── pyproject.toml
├── requirements.txt
├── config.yaml                # default config (D3FT-flavoured)
│
├── d3ft/                      # the Python package
│   ├── __init__.py
│   ├── cli.py                 # entry point: `d3ft -m {energy,band} -c config.yaml`
│   ├── config.py              # extends JrystalConfigDict with d3ft options
│   │
│   ├── optim/                 # Module 1: D3FT-Riem
│   │   ├── __init__.py
│   │   ├── stiefel.py         # Stiefel manifold ops (projection, retraction)
│   │   ├── preconditioner.py  # kinetic preconditioner (T + μI)⁻¹
│   │   └── riemannian_adam.py # Riemannian Adam with preconditioner
│   │
│   ├── occupation/            # Module 2: D3FT-OT
│   │   ├── __init__.py
│   │   └── sinkhorn.py        # capacity-bounded Sinkhorn-OT occupations
│   │
│   ├── spectrum/              # Module 3: D3FT-Lanczos
│   │   ├── __init__.py
│   │   ├── lanczos.py         # stochastic Lanczos quadrature on H(x)
│   │   └── implicit.py        # implicit differentiation through Lanczos
│   │
│   ├── calc/                  # Orchestrators
│   │   ├── __init__.py
│   │   ├── ground_state.py    # GS with d3ft modules wired in
│   │   └── band_structure.py  # band with DSLQ
│   │
│   └── utils/
│       ├── __init__.py
│       └── crystal_loader.py  # find geometries via jrystal data path
│
└── tests/                     # unit tests (numerical correctness)
    ├── test_stiefel.py
    ├── test_sinkhorn.py
    └── test_lanczos.py
```

---

## 3. v0.1 Module Specifications

### 3.1 `d3ft.optim` — D3FT-Riem

* `d3ft.optim.stiefel`
  * `project_tangent(X, V)` → tangent-space projection $V - X(X^{\dagger}V + V^{\dagger}X)/2$
  * `retract_qr(X, V)` → QR retraction
  * `retract_cayley(X, V)` → Cayley retraction (numerically more stable for small steps)
* `d3ft.optim.preconditioner`
  * `kinetic_preconditioner(g_vec, kpts, mu=1.0)` → diagonal $(T + \mu I)^{-1}$ over G-vectors
* `d3ft.optim.riemannian_adam`
  * `riemannian_adam(learning_rate, b1, b2, eps, preconditioner=None)` → `optax.GradientTransformation`
  * Internally: project gradient → preconditioner → tangent first/second moments → retract

### 3.2 `d3ft.occupation` — D3FT-OT

* `d3ft.occupation.sinkhorn`
  * `sinkhorn_occupation(eps, num_electrons, ..., n_iter=50)` → `(spin, kpt, band)` occupations
  * Reformulates the occupation problem as a capacity-bounded entropic-OT problem:
    sources = $N_e$ units of mass, sinks = bands with capacity ≤ 1 (spin-restricted: ≤ 2),
    cost = band energy.
  * Vanishing-regularisation schedule: $\epsilon_t = \epsilon_0 / \log(t)$.
  * Log-domain implementation for numerical stability as $\epsilon \to 0$.

### 3.3 `d3ft.spectrum` — D3FT-Lanczos

* `d3ft.spectrum.lanczos`
  * `lanczos_eigvals(Hop, dim, k_iter, n_probes, key)` →
    approximate lowest $k$ eigenvalues via stochastic Lanczos quadrature.
  * `Hop(v)` is provided by JAX `jvp/vjp` so no explicit Hessian is ever assembled.
* `d3ft.spectrum.implicit`
  * `eigvals_with_grad(matrix_fn, params, ...)` →
    eigenvalues + gradients with respect to `params` via the adjoint method
    (Eigenvalue perturbation: $\partial \lambda_i / \partial p = \langle v_i | \partial H / \partial p | v_i \rangle$).

### 3.4 `d3ft.calc` — Orchestrators

* `d3ft.calc.ground_state.calc(config)` →
  mirrors `jrystal.calc.calc_ground_state_energy_normcons.calc` but swaps in
  `riemannian_adam` and `sinkhorn_occupation`.
* `d3ft.calc.band_structure.calc(config, gs_output)` →
  mirrors `jrystal.calc.calc_band_structure_normcons.calc` but uses `d3ft.spectrum.lanczos`.

---

## 4. CLI

`d3ft -m {energy,band} -c config.yaml [--module {riem,ot,lanczos}=on/off ...]`

The `--module` flag controls which D3FT modules are active. With all three off, D3FT
reproduces Jrystal's behaviour (this is the baseline mode for ablation experiments).

---

## 5. Config File

`config.yaml` extends Jrystal's keys with three D3FT-specific blocks:

```yaml
# Inherits all Jrystal keys (crystal, xc, cutoff_energy, ...)
crystal: "si"
xc: "lda_x+lda_c_pw"
use_pseudopotential: true
cutoff_energy: 40
k_grid_sizes: [4, 4, 4]
epoch: 2000

d3ft:
  # Module 1: D3FT-Riem
  optim:
    enabled: true
    preconditioner: "kinetic"   # "none" | "kinetic"
    retraction: "qr"            # "qr" | "cayley"
    mu: 1.0

  # Module 2: D3FT-OT
  occupation:
    enabled: true
    epsilon_init: 1.0
    epsilon_schedule: "log"     # "log" | "exp" | "constant"
    sinkhorn_iters: 50

  # Module 3: D3FT-Lanczos
  spectrum:
    enabled: true
    n_lanczos_iters: 64
    n_probes: 8
    implicit_grad: true
```

---

## 6. Testing Strategy

`tests/` contains numerical unit tests that verify each D3FT module in isolation:

* `test_stiefel.py` — projection / retraction satisfy $X^{\dagger}X = I$ to 1e-10
* `test_sinkhorn.py` — Sinkhorn occupations converge to FD as $\epsilon \to 0$;
  occupations sum to $N_e$
* `test_lanczos.py` — Lanczos eigenvalues match `jnp.linalg.eigvalsh` on small dense
  matrices to 1e-6

Full DFT-level integration tests live in a separate benchmark package, not here, to keep this folder
fast to test.

---

## 7. Implementation Order

1. `d3ft.optim` (stiefel → preconditioner → riemannian_adam) — 1 week
2. `d3ft.occupation.sinkhorn` — 0.5 week
3. `d3ft.spectrum.lanczos` + `implicit` — 1 week
4. `d3ft.calc.ground_state` (wire Module 1 + Module 2) — 0.5 week
5. `d3ft.calc.band_structure` (wire Module 3) — 0.5 week
6. CLI + config validation — 0.5 week
7. Unit tests + numerical correctness checks — 1 week

Total: ~5 weeks to a working v0.1 ready for benchmarking.
