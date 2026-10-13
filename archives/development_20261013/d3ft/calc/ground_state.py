"""Ground-state energy minimisation with D3FT modules.

This is a thin orchestration layer. The bulk of the physics (plane-wave basis, energy
components, FFT, pseudopotentials) comes from :mod:`jrystal._src` and
:mod:`jrystal.pseudopotential`. D3FT replaces:

* The Euclidean Adam update                → :func:`d3ft.optim.riemannian_adam`
* The simplex-projector occupations        → :func:`d3ft.occupation.sinkhorn`

Both replacements are gated on the config flags ``d3ft.optim.enabled`` and
``d3ft.occupation.enabled`` so that the same script can run all four ablation
combinations.
"""
from __future__ import annotations

import os
import time
import gc
from dataclasses import dataclass
from functools import partial
from math import ceil
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from tqdm import tqdm

from jrystal._src import energy, entropy as jr_entropy, occupation as jr_occupation, pw
from jrystal._src.crystal import Crystal
from jrystal._src.grid import proper_grid_size
from d3ft import grid as d3grid
from jrystal.calc.convergence import create_convergence_checker
from jrystal.calc.opt_utils import (
    create_crystal,
    create_freq_mask,
    create_grids,
    create_pseudopotential,
    get_ewald_coulomb_repulsion,
    set_env_params,
)
from .. import pseudopotential as d3pp

from ..occupation.sinkhorn import epsilon_schedule, sinkhorn_occupation
from ..optim.preconditioner import kinetic_preconditioner
from ..optim.riemannian_adam import riemannian_adam
from ..optim.stiefel import retract_qr, stiefel_coeff, stiefel_param_init


@dataclass
class GroundStateOutput:
    config: Any
    crystal: Crystal
    params_pw: dict
    params_occ: dict
    total_energy: float
    energy_history: list
    # Converged real-space electron density grid. Consumed by the band-structure
    # calculator (and any other post-processing) as the fixed Kohn-Sham density.
    density_grid: Any = None
    # NLCC pseudo-core charge density rho_core(r) on the FFT grid (None if the
    # pseudopotentials have no non-linear core correction). The band readout must
    # pass this to make_ks_hvp so the KS potential matches the SCF XC potential.
    core_density_grid: Any = None
    # Block-diagonal Kleinman-Bylander D matrix (None without pseudopotentials).
    # The band readout passes this to make_ks_hvp for the nonlocal term.
    nl_d_matrix: Any = None
    # --- everything below is what d3ft.calc.stress needs to rebuild the free
    # energy at a strained cell, at these same converged parameters.
    freq_mask: Any = None          # fixed Miller-index basis; must NOT follow the cell
    k_frac: Any = None             # k-points in fractional coords (cell-independent)
    pseudopot: Any = None
    beta_tables: Any = None        # cell-independent SBT tables, see pseudopotential.beta
    beta_kmax: float = None
    grid_kmax: float = None        # max |G| of the nominal grid; sizes V_loc/NLCC tables
    temperature: float = 0.0       # smearing the run ended at, for F = E - T S
    final_eps: float = None        # last Sinkhorn epsilon; occupations depend on it
    make_coeff: Any = None         # params_pw -> [spin, kpt, n_g, band]
    get_occupation: Any = None     # (params_occ, eps) -> occupations


def _projectors_reciprocal_kpool(
    pseudopot, crystal, g_vec, k_vec, freq_mask, beta_kmax, beta_tables, mesh
):
    """Build nonlocal projectors from each device's local k-point slice.

    Building ``G+k``, radial projectors and spherical harmonics eagerly first
    materialises the complete ``[k, projector, m, G]`` result on GPU 0. Moving
    that result to a k-sharded layout afterwards releases the live source, but
    the allocator's high-water pages remain charged to GPU 0. At GaP k=10 this
    is the roughly 25 GiB asymmetry visible in ``nvidia-smi``; k=12 can exceed an
    80 GiB device before the later scatter is reached.

    ``shard_map`` makes the k slice local before any of those intermediates are
    formed, so no device constructs the global projector tensor. The returned
    numerical expression is identical to the eager path, only partitioned.
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def local_build(local_k):
        local_gk = d3pp.gk_vectors(g_vec, local_k, freq_mask)
        local_beta = d3pp.beta_grid(
            pseudopot.r_grid,
            pseudopot.nonlocal_beta_grid,
            pseudopot.nonlocal_angular_momentum,
            local_gk,
            beta_kmax,
            tables=beta_tables,
        )
        return d3pp.projectors_reciprocal(
            crystal.positions,
            local_gk,
            pseudopot.nonlocal_angular_momentum,
            local_beta,
        )

    build = shard_map(
        local_build,
        mesh=mesh,
        in_specs=P("k", None),
        out_specs=P("k", None, None, None),
        check_rep=False,
    )
    return build(k_vec)


def _make_atomic_init(
    pseudopot, crystal, g_vec, k_vec, freq_mask,
    potential_loc, potential_nl, nl_d_matrix, vol,
    num_bands, spin_restricted, xc, core_density_grid=None,
):
    """Return Stiefel-manifold wavefunction params initialised from PP_RHOATOM.

    Builds the KS Hamiltonian at the superposition of atomic valence densities and
    returns the ``num_bands`` lowest eigenvectors for every k-point as the initial
    wavefunction.  Returns ``None`` when no atom has a PP_RHOATOM field so the
    caller can fall back to random initialisation.
    """
    import jax.numpy as jnp
    from ..pseudopotential import core_charge_real
    from ..spectrum.ks_matvec import make_ks_hvp

    atomic_grids = getattr(pseudopot, "atomic_charge_grid", None)
    if atomic_grids is None or not any(c is not None for c in atomic_grids):
        return None  # caller checks for None before unpacking

    # Build superposition-of-atomic-densities on the FFT grid.
    # UPF PP_RHOATOM stores Q(r) = 4π r² ρ(r) (the radial charge density with
    # the angular factor), NOT ρ(r) itself.  core_charge_real expects the actual
    # 3D charge density ρ(r), so we divide Q(r) / (4π r²) before passing in.
    idx = [i for i, c in enumerate(atomic_grids) if c is not None]
    pos_arr = np.asarray(crystal.positions)[idx]
    rgrid_atom = [pseudopot.r_grid[i] for i in idx]
    rhoatom = [
        atomic_grids[i] / (4 * np.pi * pseudopot.r_grid[i]**2)
        for i in idx
    ]
    init_density = core_charge_real(
        jnp.asarray(pos_arr), g_vec, rgrid_atom, rhoatom, vol
    )
    init_density = jnp.abs(init_density)  # ensure positive definite
    # make_ks_hvp expects a spin axis: [spin, x, y, z].
    init_density = init_density[None, ...]

    num_kpts = k_vec.shape[0]
    mask = np.asarray(freq_mask).astype(bool)
    n_g = int(np.sum(mask))
    num_spin = 1 if spin_restricted else 2

    # Ghost floor in Hartree: skip KB ghost states in the initial wavefunction.
    # Ghost states are KB artefacts bound by the local potential alone; they appear
    # at anomalously low energies and are absent from the physical spectrum.
    # Env var GHOST_INIT_FLOOR_EV (default -40 eV) sets the threshold: any
    # eigenvector with eigenvalue below the floor is replaced by the next physical
    # eigenvector above the floor.  Setting to a very negative value disables this.
    _HA2EV = 27.211386245988
    _ghost_floor_ha = float(
        np.asarray(os.environ.get("GHOST_INIT_FLOOR_EV", "-40.0"), dtype=float)
    ) / _HA2EV

    # For each k-point diagonalise H_KS(init_density) and take the lowest bands.
    coeff_list = []
    ghost_vecs_list = []  # per-k ghost eigenvectors for penalty projector
    for ik in range(num_kpts):
        kpt = jnp.asarray(k_vec[ik])
        pot_nl_k = jax.tree.map(lambda x: x[ik], potential_nl)
        H_op, _ = make_ks_hvp(
            init_density, potential_loc, pot_nl_k, g_vec, kpt, vol,
            freq_mask, xc=xc, core_density_grid=core_density_grid,
            nl_d_matrix=nl_d_matrix,
        )
        eye = jnp.eye(n_g, dtype=jnp.complex128)
        H_mat = jax.vmap(H_op, in_axes=1, out_axes=1)(eye)
        H_sym = 0.5 * (H_mat + jnp.conj(H_mat).T)
        vals, vecs = jnp.linalg.eigh(H_sym)  # ascending eigenvalues
        # Partition into ghost (below floor) and physical (above floor) eigenvectors.
        physical_mask_k = np.asarray(vals) >= _ghost_floor_ha
        ghost_mask_k = ~physical_mask_k
        n_physical = int(physical_mask_k.sum())

        # Store ghost eigenvectors for this k-point (used later as projectors).
        ghost_vecs_k = vecs[:, ghost_mask_k]           # [n_g, n_ghost_k]
        ghost_vecs_list.append(np.asarray(ghost_vecs_k))

        if n_physical < num_bands:
            # Not enough physical states — fall back to all eigenvectors.
            coeff_list.append(vecs[:, :num_bands])
        else:
            physical_vecs = vecs[:, physical_mask_k]   # [n_g, n_physical]
            coeff_list.append(physical_vecs[:, :num_bands])  # [n_g, num_bands]

    # Stack into [num_spin, num_kpts, n_g, num_bands] and enforce orthonormality.
    coeff = jnp.stack(coeff_list, axis=0)          # [num_kpts, n_g, num_bands]
    coeff = coeff[None, ...]                        # [1, num_kpts, n_g, num_bands]
    if not spin_restricted:
        coeff = jnp.concatenate([coeff, coeff], axis=0)  # duplicate spin channel

    # Return (wavefunction_init, ghost_projectors) so the caller can add a
    # ghost-penalty term to the energy and prevent re-contamination during SCF.
    # ghost_projectors: list over k-points of [n_g, n_ghost] arrays (may be empty).
    return coeff, ghost_vecs_list


def _match_sharding(opt_state, reference, num_kpts):
    """Give the optimiser state the same k-sharding the parameters have.

    Adam's first and second moments have the parameters' shape, so they carry a
    k axis and belong on the same devices. Leaving them replicated while the
    parameters are sharded makes ``jax.jit`` try to reconcile a committed
    sharding against an unspecified one, which on jax 0.4.27 fails inside
    ``_array_shard_arg`` with ``UnspecifiedValue has no attribute
    addressable_devices_indices_map`` -- an error that names neither sharding nor
    the array responsible.

    A no-op when the parameters are not sharded, which is the default path.
    """
    sharding = getattr(jnp.asarray(reference), "sharding", None)
    if sharding is None or not hasattr(sharding, "spec"):
        return opt_state

    from ..parallel import kpool_sharding

    mesh = sharding.mesh

    def place(leaf):
        # A PartitionSpec is tied to a rank, so it has to be rebuilt per array
        # rather than reused from the parameters: the occupation state carries
        # rank-3 arrays (1, kpt, band) alongside the rank-4 coefficients, and
        # applying the parameters' spec to them raises "only valid for values of
        # rank at least 4".
        arr = jnp.asarray(leaf)
        if arr.ndim > 1 and arr.shape[1] == num_kpts:
            return jax.device_put(arr, kpool_sharding(mesh, arr.ndim, k_axis=1))
        return leaf

    return jax.tree_util.tree_map(place, opt_state)


def _learning_rate_schedule(config, base=None):
    """Step-size schedule, or a plain float when the schedule is "constant".

    Adam at a fixed rate does not converge to the minimum; it settles into an
    orbit whose radius is set by the step size. Measured on AlP at u = 0.27, the
    variational residual stayed between 2.4e-3 and 2.8e-3 from 4000 steps to
    32000 with no trend, the total energy wandered over 0.3 mRy, and the acoustic
    sum rule went 2.9e-2, 8.6e-3, 3.9e-3, 2.9e-2 eV/A. That single behaviour is
    the most likely common cause of the 1.3 mRy seed spread, the C11 and C12
    scatter in the elastic constants, and the residual net force.

    Default stays "constant" so existing results reproduce.
    """
    base = config.optimizer_args.learning_rate if base is None else base
    optim = config.d3ft.get("optim", {})
    kind = optim.get("lr_schedule", "constant")
    if kind == "constant":
        return base
    final = optim.get("lr_final", 1.0e-4)
    steps = max(int(config.epoch), 1)
    if kind == "cosine":
        return optax.cosine_decay_schedule(
            init_value=base, decay_steps=steps, alpha=final / base
        )
    if kind == "exponential":
        return optax.exponential_decay(
            init_value=base, transition_steps=steps, decay_rate=final / base
        )
    raise ValueError(
        f"unknown optim.lr_schedule {kind!r}; use 'constant', 'cosine' or 'exponential'"
    )


def _inscribed_g_radius(gv):
    """Radius of the largest |G| sphere that fits inside the FFT grid's G box.

    The box corner is the wrong measure. A density component is representable
    only if its G lies in the box, so what matters is the largest sphere
    *contained* in the box -- the shortest |G| on the Nyquist faces, where a
    frequency index sits at -n/2. In a primitive fcc basis the corner runs
    2.28x longer than that, and using it made this check inert: at the
    grid_sizes actually in use it reported ratios of 2.26 to 3.15 (a
    comfortable pass) where the true ratios were 1.00 to 1.38 (aliased). It
    never fired once, including on the very GaAs runs whose 0.0247 Ha offset
    from Quantum ESPRESSO prompted writing it.

    ``gv`` is the G grid, shape ``[nx, ny, nz, 3]``.
    """
    best = np.inf
    for axis in range(3):
        # Locate the -n/2 face without assuming FFT or centred ordering: walk
        # the axis with the other indices pinned, and take the step vector's
        # most negative projection. Both layouts land on the same face.
        probe = [0, 0, 0]
        probe[axis] = slice(None)
        line = gv[tuple(probe)]                      # [n, 3]
        step = line[1] - line[0]
        face_index = int(np.argmin(line @ step))
        sl = [slice(None)] * 3
        sl[axis] = face_index
        face = gv[tuple(sl)].reshape(-1, 3)
        best = min(best, float(np.linalg.norm(face, axis=1).min()))
    return best


def _check_density_grid(config, g_vec, freq_mask):
    """Warn when the FFT grid is too coarse to hold the density.

    ``grid_sizes`` is an independent config entry: it does not follow
    ``cutoff_energy``. The wavefunctions live inside |G| <= G_cut, so the
    density rho = |psi|^2 reaches 2 G_cut and the grid has to span at least
    that. Nothing enforced it, and the default 24 quietly failed twice:

      * Above cutoff_energy ~ 56 Ha at a = 5.65 Ang the 24^3 grid no longer
        reaches 2 G_cut at all, and GaAs's pressure walked away from the
        converged answer as the cutoff rose (+8.27/+8.90/+9.50/+9.88 GPa at
        40/50/65/90 Ha, against Quantum ESPRESSO's +3.3).
      * Even at ratio 2.54, where the density itself does fit, the GGA
        integrand eps_xc(rho, grad rho) is not band-limited, and a semicore
        charge is sharp enough that the real-space quadrature is visibly off:
        GaAs (Ga 3d) came out 0.0247 Ha below QE on a byte-identical
        pseudopotential, dropping to 3.2e-4 Ha at 32^3. AlP and Si, which have
        no semicore states, move by 4e-7 Ha between the two grids.

    So 2 is a hard floor and still not enough for semicore pseudopotentials --
    hence the second, softer threshold.

    Measured at cutoff_energy = 50 Ha on the four zincblende systems (AlP, GaP,
    GaAs, AlAs), inscribed radius over G_cut:

        grid_sizes   24     32     40     48     64
        ratio       1.00   1.33   1.38   2.07   2.76

    so **48 is the smallest grid that holds the density** at this cutoff, and
    the 2.5 threshold wants 64. Everything run at 32 was aliased. The symptom
    is a violation of the crystal's own point group: with one atom displaced
    along x, C2x is an exact symmetry of the cell, so the force must be purely
    along x -- yet the transverse component reached 106% of the longitudinal
    one on GaAs and 23% on GaP, while soft-pseudopotential AlP and AlAs stayed
    at 7e-4 and 3e-3. The aliasing is common to all four; only the hard Ga 3d
    projector still carries amplitude out at the Nyquist face to expose it.
    """
    import warnings

    gv = np.asarray(g_vec)
    mask = np.asarray(freq_mask)
    if not mask.any():
        return
    norms = np.linalg.norm(gv, axis=-1)
    g_cut = float(norms[mask].max())
    if g_cut <= 0.0:
        return
    g_grid = _inscribed_g_radius(gv)
    ratio = g_grid / g_cut
    if ratio < 2.0:
        warnings.warn(
            f"grid_sizes={config.grid_sizes} spans |G| <= {g_grid:.2f}, only "
            f"{ratio:.2f}x the wavefunction cutoff {g_cut:.2f}. The density "
            f"needs 2x, so it is being aliased. Raise grid_sizes to at least "
            f"{int(np.ceil(config.grid_sizes * 2.0 / ratio))}.",
            RuntimeWarning, stacklevel=2,
        )
    elif ratio < 2.5:
        warnings.warn(
            f"grid_sizes={config.grid_sizes} spans only {ratio:.2f}x the "
            f"wavefunction cutoff. That holds the density but is thin for the "
            f"GGA integrand; semicore pseudopotentials need more.",
            RuntimeWarning, stacklevel=2,
        )


def calc(config, init_params: dict = None, freq_mask=None) -> GroundStateOutput:
    """Run ground-state energy minimisation under D3FT.

    Args:
        config: A ConfigDict produced by :func:`d3ft.config.load_config`.
        init_params: optional ``{"pw": ..., "occ": ...}`` to start from, e.g. the
            converged parameters of a neighbouring geometry.

            **Continuation is not a convenience, it is a correctness
            requirement for any sweep.** The energy landscape has many minima; a
            cold start at each point of a strain sweep can land on different
            branches, and the result is a discontinuous curve whose finite
            differences are meaningless. Measured on AlP over a +-1% hydrostatic
            sweep with atomic initialisation: the total energy jumped by 0.033 Ha
            between eta = 0 and eta = +0.5%, and the band gap by 0.24 eV, giving
            a "derivative" that changed by 40% when the step size was doubled.
            Shapes must match, which means the same freq_mask, k-mesh and band
            count. **A strain sweep does not give that for free**: the mask is
            ``|G| <= cutoff`` evaluated on the deformed reciprocal lattice, so
            the number of plane waves changes from one strain to the next (1243
            to 1227 for AlP at eps_xx = 0.5%). Pass ``freq_mask`` to hold it.
        freq_mask: override the mask that would be built from ``config``.

            Required, not optional, whenever geometries are being compared. The
            AD stress in :mod:`d3ft.calc.stress` differentiates at a **fixed set
            of Miller indices** -- that is precisely why it carries no Pulay
            term. An outer finite difference over strained cells has to hold the
            same set fixed, or it folds a basis-set discontinuity into the
            derivative and no longer differentiates the quantity the inner AD
            computed. :mod:`d3ft.calc.elastic` passes the reference mask to every
            deformed solve for this reason.

    Returns:
        :class:`GroundStateOutput`.
    """
    set_env_params(config)
    key = jax.random.PRNGKey(config.seed)

    crystal = create_crystal(config)
    g_vec, _, k_vec = create_grids(config)
    # Jrystal samples with ASE's half-shifted Monkhorst-Pack mesh, which is not
    # closed under the cubic point group in a primitive fcc basis. Since D3FT
    # differentiates the free energy with respect to a Cartesian strain, the
    # functional itself has to carry the symmetry -- see d3ft.grid.kpoints.
    k_mesh = config.d3ft.get("kpoints", {}).get("mesh", "gamma")
    if k_mesh != "mp":
        k_vec = jnp.asarray(
            d3grid.k_vectors(
                crystal.cell_vectors, proper_grid_size(config.k_grid_sizes), k_mesh
            )
        )
    num_kpts = k_vec.shape[0]
    freq_mask = create_freq_mask(config) if freq_mask is None else freq_mask
    _check_density_grid(config, g_vec, freq_mask)
    # Establish the device mesh before projector construction. Waiting until
    # optimiser setup is too late: the global nonlocal tensor has already made
    # GPU 0 the memory high-water device by then.
    kpool_mesh_obj = None
    if config.d3ft.get("parallel", {}).get("k_pool", False):
        from ..parallel import jrystal_jax_compat, kpool_mesh

        jrystal_jax_compat()
        n_dev = len(jax.devices())
        if num_kpts % n_dev:
            raise ValueError(
                f"k_pool needs the {num_kpts} k-points to divide across "
                f"{n_dev} devices. Restrict the visible devices, or choose a "
                "k-grid whose point count the device count divides."
            )
        kpool_mesh_obj = kpool_mesh()
        logging.info(f"k-pool: {num_kpts} k-points over {n_dev} devices.")
    beta_tables = None
    beta_kmax = None
    grid_kmax = None
    pseudopot = None
    convergence_checker = create_convergence_checker(config)

    if config.use_pseudopotential:
        pseudopot = create_pseudopotential(config)
        num_electrons = float(np.sum(pseudopot.valence_charges))
        # Ion-ion repulsion must use the pseudo-ion (valence) charges, not the
        # full nuclear charges (atomic numbers).
        ew = get_ewald_coulomb_repulsion(
            config, charges=np.asarray(pseudopot.valence_charges, dtype=float)
        )

        logging.info("Pre-computing local pseudopotential...")
        # Frozen so d3ft.calc.stress can rebuild the identical V_loc at a
        # strained cell: kmax sizes a NumPy tabulation grid and cannot be read
        # off a traced grid.
        grid_kmax = float(np.max(np.linalg.norm(np.asarray(g_vec), axis=-1)))
        potential_loc = d3pp.potential_local_reciprocal(
            crystal.positions, g_vec,
            pseudopot.r_grid, pseudopot.local_potential_grid,
            pseudopot.local_potential_charge, crystal.vol, kmax=grid_kmax,
        )
        logging.info("Pre-computing non-local beta projectors (SBT)...")
        # G + k on the plane-wave cutoff sphere -- the only G-vectors the
        # wavefunctions occupy, hence the only ones the projectors are ever
        # contracted against. The FFT box holds ~6.5x as many, all of them
        # multiplied by an identically zero coefficient, and it is N^3 rather
        # than n_g that grows when the grid is refined.
        # Tabulated once and kept: the SBT depends only on the UPF radial data, so
        # a strain sweep can reuse it (see d3ft.calc.stress). kmax still spans the
        # whole box (see kmax_for_grid): it sizes the tabulation grid, and holding
        # it fixed is what makes the masked projectors bit-identical to the
        # full-grid ones rather than merely equivalent.
        beta_kmax = d3pp.kmax_for_grid(g_vec, k_vec)
        beta_tables = d3pp.build_tables(pseudopot, beta_kmax)
        use_local_projectors = (
            kpool_mesh_obj is not None
            and str(getattr(config.d3ft.optim, "init", "atomic")).lower() == "random"
            and os.environ.get("GHOST_INIT_FLOOR_EV") is None
            and os.environ.get("D3FT_LEGACY_EAGER_PROJECTORS") != "1"
        )
        if use_local_projectors:
            if os.environ.get("D3FT_TRACE_STAGES") == "1":
                print("D3FT_STAGE local_projectors_begin", flush=True)
            potential_nl = _projectors_reciprocal_kpool(
                pseudopot, crystal, g_vec, k_vec, freq_mask,
                beta_kmax, beta_tables, kpool_mesh_obj,
            )
            jax.block_until_ready(potential_nl)
            if os.environ.get("D3FT_TRACE_STAGES") == "1":
                print("D3FT_STAGE local_projectors_complete", flush=True)
        else:
            # Atomic initialisation diagonalises each full Hamiltonian eagerly;
            # keep its historical layout until that path is separately sharded.
            gk_vec = d3pp.gk_vectors(g_vec, k_vec, freq_mask)
            beta_gk = d3pp.beta_grid(
                pseudopot.r_grid, pseudopot.nonlocal_beta_grid,
                pseudopot.nonlocal_angular_momentum, gk_vec, beta_kmax,
                tables=beta_tables,
            )
            potential_nl = d3pp.projectors_reciprocal(
                crystal.positions, gk_vec,
                pseudopot.nonlocal_angular_momentum, beta_gk,
            )
        # Scalar multiplier on the KB D matrix, baked into the SCF solve so the
        # wavefunction actually converges to the minimum of the perturbed
        # functional. Zero (the default) reproduces the unperturbed
        # pseudopotential exactly. Exists for d3ft.calc.implicit's finite-
        # difference check: computing the same quantity by re-solving at
        # theta +/- h is the only trustworthy way to validate the implicit-chain
        # formula that differentiates through this same knob post hoc (see
        # d3ft.calc.stress.free_energy_at_strain's theta_nonlocal argument).
        theta_nonlocal = float(config.d3ft.get("theta_nonlocal", 0.0))
        # Per-angular-momentum strengths when the config supplies them; the scalar
        # is the uniform special case. See d3ft.pseudopotential.scaled_d_matrix for
        # why the distinction matters: the uniform knob moves B and not C44.
        theta_nonlocal_l = config.d3ft.get("theta_nonlocal_l", None)
        nl_d_matrix = d3pp.scaled_d_matrix(
            pseudopot, theta_nonlocal=theta_nonlocal,
            theta_nonlocal_l=theta_nonlocal_l,
        )
        if not use_local_projectors:
            del beta_gk, gk_vec

        # Non-linear core correction (NLCC): build the fixed pseudo-core charge
        # density rho_core(r) on the FFT grid. It enters ONLY the XC functional
        # (rho_xc = rho_valence + rho_core). None when no atom has NLCC.
        core_density_grid = None
        core_grids = getattr(pseudopot, "core_charge_grid", None)
        if core_grids is not None and any(c is not None for c in core_grids):
            from ..pseudopotential import core_charge_real
            idx = [i for i, c in enumerate(core_grids) if c is not None]
            pos_arr = np.asarray(crystal.positions)[idx]
            rgrid_nlcc = [pseudopot.r_grid[i] for i in idx]
            core_nlcc = [core_grids[i] for i in idx]
            core_density_grid = core_charge_real(
                jax.numpy.asarray(pos_arr), g_vec, rgrid_nlcc, core_nlcc, crystal.vol
            )
            _n_core = float(
                jax.numpy.sum(core_density_grid) * crystal.vol
                / np.prod(np.asarray(core_density_grid.shape))
            )
            logging.info(
                "NLCC enabled for %d/%d atoms; pseudo-core charge integral = %.4f e",
                len(idx), len(core_grids), _n_core,
            )
    else:
        core_density_grid = None
        nl_d_matrix = None
        num_electrons = float(crystal.num_electron)
        ew = get_ewald_coulomb_repulsion(config)

    num_bands = ceil(num_electrons / 2) + config.empty_bands

    if config.d3ft.optim.enabled:
        # Initialise wavefunctions by diagonalising H_KS at the atomic superposition
        # density (PP_RHOATOM). This prevents the SCF from being trapped by spurious
        # ghost-state minima that arise with random initialisation, mirroring DFTK's
        # default atomic density initialisation.
        _ghost_projectors = None   # per-k list of [n_g, n_ghost] ghost eigenvectors
        # Atomic-density initialisation, controlled by config.d3ft.optim.init.
        #
        # This used to be off by default, on the grounds that it produced
        # "conditional ghosts" -- a -44 eV state for Si.pz-vbc at the atomic
        # density that was absent with random initialisation. That was a
        # misdiagnosis of the Kleinman-Bylander conjugation defect since fixed in
        # d3ft.pseudopotential. Measured on Si (diamond, a = 5.43 A, Si.pz-vbc,
        # LDA-PZ, Gamma point, lowest eigenvalue):
        #
        #   init      HNL_CONJ_FIX   E (Ha)      lowest state (eV)
        #   atomic    0              -7.951246   -67.10
        #   random    0              -7.838178   -67.05     <- also present
        #   atomic    1              -7.868735    -5.76
        #   random    1              -7.838178    -5.80
        #
        # The deep state appears with *either* initialisation and disappears with
        # the fix, so it was never a property of the atomic density: the stated
        # reason for defaulting to random initialisation was a misdiagnosis.
        #
        # Atomic is now the default because it is deterministic -- no seed
        # lottery -- and NOT because it finds the best minimum. It does not.
        # At the same geometry (AlP, a = 5.467 A, PBE, 16 Ha, 2x2x2, 5000 steps):
        #
        #   atomic  seed 0   -9.325541 Ha
        #   random  seed 0   -9.336224
        #   random  seed 1   -9.313522
        #   random  seed 2   -9.310922
        #   random  seed 3   -9.367471   <- lowest, 0.042 Ha below atomic
        #
        # a spread of 0.057 Ha = 1.5 eV across local minima. Atomic at 15000
        # steps only reaches -9.326162, so this is basin trapping, not slow
        # convergence, and warm-starting from -9.367471 moves by 3e-5 Ha, so that
        # one is a genuine stationary point. Until this is resolved, no derived
        # quantity -- stress, band-edge derivative, anything entering a design
        # loop -- can be trusted from a single run. See PLAN.md blocker 5.4.
        _init_mode = str(getattr(config.d3ft.optim, "init", "atomic")).lower()
        if _init_mode not in ("atomic", "random"):
            raise ValueError(
                f"config.d3ft.optim.init must be 'atomic' or 'random', got {_init_mode!r}"
            )
        # Legacy escape hatch: setting GHOST_INIT_FLOOR_EV forces atomic init.
        _use_atomic_init = (
            _init_mode == "atomic" or os.environ.get("GHOST_INIT_FLOOR_EV") is not None
        )
        _atomic_init = None
        if config.use_pseudopotential and _use_atomic_init:
            _result = _make_atomic_init(
                pseudopot, crystal, g_vec, k_vec, freq_mask,
                potential_loc, potential_nl, nl_d_matrix, crystal.vol,
                num_bands, config.spin_restricted, config.xc,
                core_density_grid=core_density_grid,
            )
            if _result is not None:
                _atomic_init, _ghost_projectors = _result
        if _atomic_init is not None:
            n_ghost_total = sum(g.shape[1] for g in _ghost_projectors)
            params_pw = _atomic_init
            logging.info(
                "Initialised wavefunctions from PP_RHOATOM atomic density "
                "(GHOST_INIT_FLOOR_EV=%s). Ghost projectors: %d.",
                os.environ.get("GHOST_INIT_FLOOR_EV"), n_ghost_total,
            )
        else:
            params_pw = stiefel_param_init(
                key, num_bands, num_kpts, freq_mask,
                spin_restricted=config.spin_restricted,
            )
            logging.info("Using random wavefunction initialisation.")
        make_coeff = lambda p: stiefel_coeff(p, freq_mask)
    else:
        params_pw = pw.param_init(
            key, num_bands, num_kpts, freq_mask,
            spin_restricted=config.spin_restricted,
        )
        make_coeff = lambda p: pw.coeff(p, freq_mask)
    if config.d3ft.occupation.enabled:
        params_occ = {"band_energies": jax.numpy.zeros((1, num_kpts, num_bands))}
    else:
        params_occ = jr_occupation.param_init(
            key, num_bands, int(num_electrons), num_kpts,
            crystal.spin, config.occupation,
        )

    if config.d3ft.optim.enabled and config.d3ft.optim.preconditioner == "kinetic":
        precond_diag = kinetic_preconditioner(
            g_vec, k_vec, freq_mask=freq_mask, mu=config.d3ft.optim.mu,
        )
        # [kpt, n_g] -> [1, kpt, n_g, 1] to broadcast over (spin, kpt, g, band).
        precond_diag = precond_diag[None, :, :, None]
    else:
        precond_diag = None

    if init_params is not None:
        supplied = {k: init_params[k] for k in ("pw", "occ") if k in init_params}
        for key_, val in supplied.items():
            ref = params_pw if key_ == "pw" else params_occ
            ref_shapes = jax.tree.map(jnp.shape, ref)
            new_shapes = jax.tree.map(jnp.shape, val)
            if ref_shapes != new_shapes:
                raise ValueError(
                    f"init_params[{key_!r}] has shape {new_shapes}, expected "
                    f"{ref_shapes}. Continuation needs the same freq_mask, "
                    "k-mesh and band count as the run it came from."
                )
        params_pw = supplied.get("pw", params_pw)
        params_occ = supplied.get("occ", params_occ)
        logging.info("Warm-starting from supplied parameters.")

    params = {"pw": params_pw, "occ": params_occ}

    # k-point pool parallelism, the level that makes a *single* ground state
    # faster rather than a batch of them. It has to happen here, before the
    # optimiser is built and `update` is first traced: jit bakes the input
    # shardings into the compiled executable, so handing it an already-sharded
    # array afterwards fails with `UnspecifiedValue has no attribute
    # addressable_devices_indices_map`. Sharding at initialisation makes the
    # first trace see the layout and everything downstream inherits it.
    #
    # Callers must be inside `d3ft.parallel.plain_fft_patch()`: jrystal's
    # custom-partitioned FFT raises on CUDA as soon as a second device is
    # visible, and it exists for grid sharding, which is the opposite of what a
    # k-pool wants. See d3ft/parallel/kpool.py.
    if config.d3ft.get("parallel", {}).get("k_pool", False):
        from ..parallel import jrystal_jax_compat, kpool_mesh, shard_over_k

        # jrystal's custom partitioner calls jax.tree_map, removed in JAX 0.6.0.
        # It is only reached once GSPMD has real partitioning to do, so the
        # unsharded path never trips it -- and when it does trip, the exception
        # escapes from inside XLA's compilation callback and JAX reports an
        # unrelated UnspecifiedValue error somewhere else.
        jrystal_jax_compat()

        n_dev = len(jax.devices())
        # Keep the expanded FFT coefficients distributed as well as the compact
        # Stiefel parameters. An internal constraint alone does not specify the
        # compiled program's input/output device assignment.
        from ..parallel import kpool_sharding
        raw_make_coeff = make_coeff
        coeff_sharding = kpool_sharding(kpool_mesh_obj, 6, k_axis=1)

        if os.environ.get("D3FT_LEGACY_EAGER_COEFF") == "1":
            # Diagnostic escape hatch used to compare the local-expansion path
            # with historical results on meshes small enough to fit.
            def make_coeff(p):
                coeff = raw_make_coeff(p)
                return jax.lax.with_sharding_constraint(coeff, coeff_sharding)
        else:
            # A constraint applied *after* ``expand_coefficient`` is too late:
            # XLA may first scatter the complete [spin,k,band,x,y,z] tensor on
            # every GPU.  For GaP k=10 that is roughly 54 GiB per device.  Put
            # the scatter itself in a shard_map so each device constructs only
            # its local k-point batch, matching the local FFT immediately after
            # it in ``plain_fft_patch``.
            from jax.experimental.shard_map import shard_map
            from jax.sharding import PartitionSpec as P

            def input_spec(leaf):
                arr = jnp.asarray(leaf)
                if arr.ndim > 1 and arr.shape[1] == num_kpts:
                    spec = [None] * arr.ndim
                    spec[1] = "k"
                    return P(*spec)
                return P()

            coeff_input_specs = jax.tree_util.tree_map(input_spec, params_pw)
            local_make_coeff = shard_map(
                raw_make_coeff,
                mesh=kpool_mesh_obj,
                in_specs=coeff_input_specs,
                out_specs=P(None, "k", None, None, None, None),
                check_rep=False,
            )

            def make_coeff(p):
                return local_make_coeff(p)

    if config.d3ft.optim.enabled:
        # Wavefunction lives on the Stiefel manifold -> Riemannian Adam + QR retraction.
        # Occupation parameters are Euclidean -> ordinary Adam. The two MUST use separate
        # optimisers: feeding the Euclidean occupation params through `project_tangent`
        # corrupts their gradient (it assumes an orthonormal manifold point).
        lr_sched = _learning_rate_schedule(config)
        pw_optimizer = riemannian_adam(
            learning_rate=lr_sched,
            b1=config.optimizer_args.b1,
            b2=config.optimizer_args.b2,
            preconditioner=precond_diag,
            retraction=config.d3ft.optim.retraction,
        )
        # The band energies act on the occupations only through
        # ``band_energies / epsilon``, so a step of size ``lr`` moves the
        # exponent by ``lr / epsilon``. Sharing the plane-wave learning rate
        # therefore couples the occupation dynamics to epsilon in the wrong
        # direction: annealing epsilon down makes the effective step larger, and
        # at epsilon = 1e-3 a shared lr of 0.01 steps ten times the entire scale
        # on which the occupations vary.
        occ_lr = config.d3ft.get("occupation", {}).get("learning_rate", None)
        if occ_lr is None:
            occ_lr = config.optimizer_args.learning_rate
        occ_optimizer = optax.adam(
            learning_rate=_learning_rate_schedule(config, base=occ_lr),
            b1=config.optimizer_args.b1,
            b2=config.optimizer_args.b2,
        )
        opt_state = (pw_optimizer.init(params_pw), occ_optimizer.init(params_occ))
    else:
        optimizer = optax.adam(
            learning_rate=config.optimizer_args.learning_rate,
            b1=config.optimizer_args.b1,
            b2=config.optimizer_args.b2,
        )
        opt_state = optimizer.init(params)


    def get_occupation(params_occ, eps):
        if config.d3ft.occupation.enabled:
            return sinkhorn_occupation(
                params_occ["band_energies"],
                num_electrons=num_electrons,
                epsilon=eps,
                spin_restricted=config.spin_restricted,
                n_iter=config.d3ft.occupation.sinkhorn_iters,
            )
        return jr_occupation.occupation(
            params_occ, num_kpts, int(num_electrons),
            spin=crystal.spin, method=config.occupation,
            spin_restricted=config.spin_restricted,
        )

    # Ghost-projection retraction: after each Riemannian Adam QR retraction, project
    # the wavefunction Stiefel parameters out of the ghost subspace and re-QR.
    # This is a HARD CONSTRAINT — unlike an energy penalty, it guarantees that the
    # optimised wavefunctions always have zero overlap with the ghost eigenvectors
    # identified at the atomic-density initialisation.
    #
    # Ghost projectors: per k-point list of [n_g, n_ghost] complex arrays.
    # _ghost_projectors may be undefined when optim.enabled=False (no atomic init path).
    if "_ghost_projectors" not in dir():
        _ghost_projectors = None
    _ghost_proj_jax = None
    if _ghost_projectors is not None:
        _gp = [jnp.asarray(g, dtype=jnp.complex128) for g in _ghost_projectors]
        if any(g.shape[1] > 0 for g in _gp):
            _ghost_proj_jax = _gp   # only enable if at least one ghost was found

    def _apply_ghost_projection(pw_params):
        """Remove ghost-subspace overlap from Stiefel params and re-QR.

        pw_params: [spin, kpt, n_g, band].
        Returns same shape, guaranteed ghost-free.
        """
        if _ghost_proj_jax is None:
            return pw_params
        nk = pw_params.shape[1]
        slices = []
        for ik in range(nk):
            gp = _ghost_proj_jax[ik]       # [n_g, n_ghost]
            c_k = pw_params[:, ik, :, :]   # [spin, n_g, band]
            if gp.shape[1] > 0:
                # c_proj = c_k − gp (gp† c_k)  (orthogonal projection)
                ovlp = jnp.einsum('gj,sgb->sjb', jnp.conj(gp), c_k)  # [spin, n_ghost, band]
                c_proj = c_k - jnp.einsum('gj,sjb->sgb', gp, ovlp)    # [spin, n_g, band]
                # Re-QR spin-by-spin to restore orthonormality on Stiefel manifold.
                c_qr = jax.vmap(lambda m: jnp.linalg.qr(m)[0])(c_proj)  # [spin, n_g, band]
            else:
                c_qr = c_k
            slices.append(c_qr)
        return jnp.stack(slices, axis=1)   # [spin, kpt, n_g, band]

    # The functional's own constants, when the config supplies them. Without this
    # the SCF would minimise the stock functional and only the post-hoc energy
    # would see the override -- fine for a Hellmann-Feynman first derivative,
    # wrong for a finite change of the parameter, which is what a fit takes.
    _xc_params = config.d3ft.get("xc_params", None)
    _xc_grid_factor = config.d3ft.get("xc_grid_factor", 1)

    def _xc(density, g_vec_, vol_, core_):
        if _xc_params is None and _xc_grid_factor == 1:
            return energy.xc_energy(density, g_vec_, vol_, config.xc,
                                    kohn_sham=False, core_density_grid=core_)
        from .xc_tunable import xc_energy_tunable
        return xc_energy_tunable(density, g_vec_, vol_, config.xc,
                                 core_density_grid=core_, xc_params=_xc_params,
                                 grid_factor=_xc_grid_factor)

    if config.use_pseudopotential:

        # pot_nl is threaded in rather than closed over. Closing over it makes
        # XLA embed the projector array in the compiled executable as a
        # compile-time constant: [216 k, beta, m, 32^3] complex128 is 5.7 GB for
        # GaAs, which every run reported as "A large amount of constants were
        # captured during lowering (6.80GB total)" and nobody read. The constant
        # pool is a second copy on top of the runtime buffer, it inflates
        # compile time, and it is what pushes k-pool partitioning past XLA's
        # 2 GB protobuf limit (see d3ft.parallel.kpool).
        def total_energy(params_pw, params_occ, eps, pot_nl):
            coeff = make_coeff(params_pw)
            occ = get_occupation(params_occ, eps)
            density = pw.density_grid(coeff, crystal.vol, occ)
            # ``density_grid_reciprocal`` calls ``density_grid`` internally.
            # Reusing the density here avoids a second wavefunction IFFT,
            # absolute-square tensor and k reduction in both the forward pass
            # and its reverse-mode tape.
            density_rec = jnp.fft.fftn(density, axes=range(-3, 0))
            e_kin = energy.kinetic(g_vec, k_vec, coeff, occ)
            e_har = energy.hartree(density_rec, g_vec, crystal.vol)
            e_loc = d3pp.energy_local(density_rec, potential_loc, vol=crystal.vol)
            e_nloc = d3pp.energy_nonlocal(
                coeff, pot_nl, vol=crystal.vol, occupation=occ,
                d_matrix=nl_d_matrix, freq_mask=freq_mask,
            )
            e_xc = _xc(density, g_vec, crystal.vol, core_density_grid)
            return e_kin + e_har + e_loc + e_nloc + e_xc
    else:
        # pot_nl is unused without a pseudopotential; the signature stays
        # uniform so loss_and_energy does not have to branch.
        def total_energy(params_pw, params_occ, eps, pot_nl):
            coeff = make_coeff(params_pw)
            occ = get_occupation(params_occ, eps)
            density = pw.density_grid(coeff, crystal.vol, occ)
            density_rec = jnp.fft.fftn(density, axes=range(-3, 0))
            e_kin = energy.kinetic(g_vec, k_vec, coeff, occ)
            e_har = energy.hartree(density_rec, g_vec, crystal.vol)
            e_ext = energy.external(
                density_rec, crystal.positions, crystal.charges, g_vec, crystal.vol
            )
            e_xc = _xc(density, g_vec, crystal.vol, core_density_grid)
            return e_kin + e_har + e_ext + e_xc

    # Optional Mermin free-energy target F = E - T*S, matching Jrystal's native pipeline so
    # that the optim/occupation ablations can be compared against the Jrystal baseline under
    # an identical objective. Gradients flow through F; the monitored/returned quantity is
    # always the total energy E (as in Jrystal), so the convergence check and energy_history
    # remain on E.
    use_free_energy = config.d3ft.free_energy.enabled

    def get_entropy(params_occ, eps):
        occ = get_occupation(params_occ, eps)
        return jr_entropy.fermi_dirac(occ, eps=config.eps)

    if use_free_energy and config.smearing > 0:
        temperature_scheduler = optax.exponential_decay(
            init_value=config.d3ft.free_energy.temp_init,
            transition_steps=max(config.epoch // 2, 1),
            decay_rate=config.smearing / config.d3ft.free_energy.temp_init,
            end_value=config.smearing,
        )
    else:
        def temperature_scheduler(i):
            return 0.0

    def loss_and_energy(params, eps, temp, pot_nl):
        total = total_energy(params["pw"], params["occ"], eps, pot_nl)
        if use_free_energy:
            free = total - temp * get_entropy(params["occ"], eps)
            return free, total
        return total, total

    def _kshard(params):
        """Declare the k layout inside the traced graph, not on the inputs.

        Committing the arrays outside with ``device_put`` and passing them to a
        plain ``jax.jit`` sends JAX down ``_array_shard_arg``, which needs a
        resolved destination sharding and does not have one. Constraining inside
        the graph leaves every argument uncommitted, so that path is never
        entered, and GSPMD still lays the coefficients out over the mesh and
        inserts the one collective the density sum needs.
        """
        if kpool_mesh_obj is None:
            return params
        from ..parallel import kpool_sharding

        coeff = params["pw"]
        return {
            **params,
            "pw": jax.lax.with_sharding_constraint(
                coeff, kpool_sharding(kpool_mesh_obj, jnp.ndim(coeff), k_axis=1)
            ),
        }

    compile_update = jax.jit
    if kpool_mesh_obj is not None:
        from jax.sharding import NamedSharding, PartitionSpec
        from ..parallel import kpool_sharding

        replicated = NamedSharding(kpool_mesh_obj, PartitionSpec())

        def layout(leaf):
            arr = jnp.asarray(leaf)
            if arr.ndim > 1 and arr.shape[1] == num_kpts:
                return kpool_sharding(kpool_mesh_obj, arr.ndim, k_axis=1)
            return replicated

        params_layout = jax.tree_util.tree_map(layout, params)
        state_layout = jax.tree_util.tree_map(layout, opt_state)
        pot_layout = (kpool_sharding(kpool_mesh_obj, potential_nl.ndim, k_axis=0)
                      if config.use_pseudopotential else None)
        # Explicit signatures prevent a replicated executable from satisfying
        # only an internal constraint and still expanding the full FFT batch on
        # every GPU. Place persistent inputs once, rather than copying each step.
        # Initialisation happens before the distributed layout exists, so JAX
        # places the source parameter and optimiser trees on GPU 0.  A normal
        # device_put leaves those sources alive after making the eight shards;
        # at GaP k=12 that accounted for the persistent ~32.6-GiB GPU-0 excess.
        # Donation transfers ownership to the distributed arrays and frees the
        # source buffers once placement completes.
        params = jax.device_put(params, params_layout, donate=True)
        opt_state = jax.device_put(opt_state, state_layout, donate=True)
        jax.block_until_ready((params, opt_state))
        params_pw = None
        params_occ = None
        gc.collect()
        compile_update = partial(
            jax.jit,
            in_shardings=(params_layout, state_layout, replicated, replicated, pot_layout),
            out_shardings=(params_layout, state_layout, replicated),
            # The loop immediately replaces both trees with the returned ones.
            # Donation lets XLA reuse their large wavefunction/Adam buffers
            # instead of retaining an input and output copy during every step.
            donate_argnums=(0, 1),
        )

    if config.d3ft.optim.enabled:
        @compile_update
        def update(params, opt_state, eps, temp, pot_nl):
            params = _kshard(params)
            pw_state, occ_state = opt_state
            (loss, total), grad = jax.value_and_grad(loss_and_energy, has_aux=True)(
                params, eps, temp, pot_nl
            )
            pw_dir, pw_state = pw_optimizer.update(grad["pw"], pw_state, params["pw"])
            occ_upd, occ_state = occ_optimizer.update(grad["occ"], occ_state, params["occ"])
            new_pw = retract_qr(params["pw"], pw_dir)
            # Hard ghost-projection: remove ghost-subspace overlap and re-QR.
            # This is a constraint step (not part of the gradient) that keeps the
            # wavefunction in the physical subspace throughout the SCF.
            new_pw = _apply_ghost_projection(new_pw)
            new_occ = optax.apply_updates(params["occ"], occ_upd)
            return {"pw": new_pw, "occ": new_occ}, (pw_state, occ_state), total
    else:
        @compile_update
        def update(params, opt_state, eps, temp, pot_nl):
            params = _kshard(params)
            (loss, total), grad = jax.value_and_grad(loss_and_energy, has_aux=True)(
                params, eps, temp, pot_nl
            )
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, total

    # None without a pseudopotential: jit treats it as an empty pytree, so the
    # uniform signature costs nothing.
    pot_nl_arg = potential_nl if config.use_pseudopotential else None
    if kpool_mesh_obj is not None and pot_nl_arg is not None:
        # ``potential_nl`` is initially constructed as one full array on the
        # default GPU.  At large k meshes that source buffer alone can occupy
        # most of GPU 0 even after a distributed copy has been made.  The source
        # is dead after this transfer, so donate it and wait for the asynchronous
        # placement to finish before dropping Python's remaining reference.
        pot_nl_arg = jax.device_put(pot_nl_arg, pot_layout, donate=True)
        jax.block_until_ready(pot_nl_arg)
        potential_nl = None
        gc.collect()

    energy_history = []
    trace_stages = os.environ.get("D3FT_TRACE_STAGES") == "1"
    if trace_stages:
        print("D3FT_STAGE update_begin", flush=True)
    iters = tqdm(range(config.epoch), disable=not config.verbose)
    for i in iters:
        eps_t = epsilon_schedule(
            i,
            epsilon_init=config.d3ft.occupation.epsilon_init,
            epsilon_final=config.d3ft.occupation.epsilon_final,
            total_steps=config.epoch,
            schedule=config.d3ft.occupation.epsilon_schedule,
        )
        temp_t = temperature_scheduler(i)
        params, opt_state, total = update(
            params, opt_state, eps_t, temp_t, pot_nl_arg
        )
        total = float(jax.block_until_ready(total))
        energy_history.append(total)
        if trace_stages and (i == 0 or (i + 1) % 500 == 0 or i == config.epoch - 1):
            print(f"D3FT_STAGE update_complete {i + 1}/{config.epoch}", flush=True)

        if convergence_checker.check(total):
            logging.info("Converged at step %d.", i)
            break
        if i % 50 == 0 or i == config.epoch - 1:
            logging.info(
                "step %d: Energy=%.6f Ha | eps=%.2e | temp=%.2e",
                i, total + ew, eps_t, temp_t,
            )
        iters.set_description(f"E={total + ew:.6f} eps={eps_t:.2e}")

    # Reconstruct the converged real-space density so downstream consumers (the
    # band-structure calculator) can use it as the fixed Kohn-Sham density.
    if kpool_mesh_obj is not None and os.environ.get("D3FT_FUSE_FINAL_DENSITY") == "1":
        # Keep expansion, local FFT and density reduction in one partitioned
        # executable. Eager expansion first creates a full-grid zero array on
        # the default device, before the downstream sharding constraint.
        if trace_stages:
            print("D3FT_STAGE fused_final_density_begin", flush=True)
        def final_density(p):
            coeff = make_coeff(p["pw"])
            occ = get_occupation(p["occ"], eps_t)
            return pw.density_grid(coeff, crystal.vol, occ)
        density_grid = jax.jit(
            final_density, in_shardings=(params_layout,), out_shardings=replicated,
        )(params)
    else:
        if trace_stages:
            print("D3FT_STAGE final_coeff_begin", flush=True)
        final_coeff = make_coeff(params["pw"])
        if trace_stages:
            jax.block_until_ready(final_coeff)
            print("D3FT_STAGE final_coeff_complete", flush=True)
        final_occ = get_occupation(params["occ"], eps_t)
        density_grid = pw.density_grid(final_coeff, crystal.vol, final_occ)
    if trace_stages:
        jax.block_until_ready(density_grid)
        print("D3FT_STAGE final_density_complete", flush=True)

    return GroundStateOutput(
        config=config, crystal=crystal,
        params_pw=params["pw"], params_occ=params["occ"],
        total_energy=energy_history[-1] + ew,
        energy_history=energy_history,
        density_grid=density_grid,
        core_density_grid=core_density_grid if config.use_pseudopotential else None,
        nl_d_matrix=nl_d_matrix if config.use_pseudopotential else None,
        freq_mask=freq_mask,
        k_frac=np.asarray(k_vec) @ np.linalg.inv(np.asarray(crystal.reciprocal_vectors)),
        pseudopot=pseudopot if config.use_pseudopotential else None,
        beta_tables=beta_tables if config.use_pseudopotential else None,
        beta_kmax=beta_kmax if config.use_pseudopotential else None,
        grid_kmax=grid_kmax if config.use_pseudopotential else None,
        temperature=float(temp_t),
        final_eps=float(eps_t),
        make_coeff=make_coeff,
        get_occupation=get_occupation,
    )
