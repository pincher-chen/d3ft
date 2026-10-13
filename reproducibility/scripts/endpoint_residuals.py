"""Diagnostic KS/orbital/occupation residuals on a completed metal pilot.

These are fixed-density diagnostics with an occupation energy-gradient parity
check. They do not certify the global minimum or the lowest empty-state spectrum.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import hashlib
from pathlib import Path
from scipy.optimize import brentq
from scipy.special import expit

from d3ft import pseudopotential as d3pp
from d3ft.spectrum.ks_matvec import make_ks_hvp
from jrystal._src import entropy, energy, pw
from jrystal._src.grid import g_vectors
from jrystal.calc.opt_utils import get_ewald_coulomb_repulsion


def evaluate(out, cfg):
    crystal = out.crystal
    mask = out.freq_mask
    gvec = g_vectors(crystal.cell_vectors, np.asarray(mask.shape))
    kvec = jnp.asarray(out.k_frac) @ jnp.asarray(crystal.reciprocal_vectors)
    pp = out.pseudopot
    vloc = d3pp.potential_local_reciprocal(
        crystal.positions, gvec, pp.r_grid, pp.local_potential_grid,
        pp.local_potential_charge, crystal.vol, kmax=out.grid_kmax,
    )
    gk = d3pp.gk_vectors(gvec, kvec, mask)
    beta = d3pp.beta_grid(
        pp.r_grid, pp.nonlocal_beta_grid, pp.nonlocal_angular_momentum,
        gk, out.beta_kmax, tables=out.beta_tables,
    )
    projectors = d3pp.projectors_reciprocal(
        crystal.positions, gk, pp.nonlocal_angular_momentum, beta,
    )
    coeff = out.make_coeff(out.params_pw)
    occ = np.asarray(out.get_occupation(out.params_occ, out.final_eps))
    temperature = float(out.temperature)
    cap = 2.0 / occ.shape[1]
    entropy_grad = np.asarray(jax.grad(entropy.fermi_dirac)(jnp.asarray(occ), cfg.eps))
    orbital_outside = []
    orbital_unweighted = []
    orbital_commutator = []
    orbital_hermiticity = []
    matrices = []
    band_diagonal = []
    orthonormality = []
    for ik in range(occ.shape[1]):
        pn_k = jax.tree.map(lambda x: x[ik], projectors)
        hop, ng = make_ks_hvp(
            out.density_grid, vloc, pn_k, gvec, kvec[ik], crystal.vol,
            mask, xc=cfg.xc, core_density_grid=out.core_density_grid,
            nl_d_matrix=out.nl_d_matrix,
        )
        # The SCF API expands coefficients onto [band, x, y, z]. The matrix-free
        # action uses the compact masked-G representation [n_g, band].
        c = jnp.asarray(coeff[0, ik][:, np.asarray(mask, dtype=bool)].T)
        if c.shape[0] != ng:
            raise ValueError(f"basis mismatch at k={ik}: {c.shape} vs {ng}")
        hc = jax.vmap(hop, in_axes=1, out_axes=1)(c)
        orthonormality.append(float(jnp.max(jnp.abs(jnp.conj(c.T) @ c - jnp.eye(c.shape[1])))))
        hband = jnp.conj(c.T) @ hc
        h_np = np.asarray(hband)
        matrices.append(h_np)
        e = np.real(np.diag(h_np))
        band_diagonal.append(e)
        f = occ[0, ik]
        projected = hc - c @ hband
        weighted = np.asarray(projected) * f[None, :]
        orbital_outside.append(float(np.linalg.norm(weighted)))
        orbital_unweighted.append(float(np.linalg.norm(np.asarray(projected))))
        orbital_commutator.append(float(np.linalg.norm(h_np * f[None, :] - f[:, None] * h_np)))
        orbital_hermiticity.append(float(np.linalg.norm(h_np - h_np.conj().T)))
    import os
    if os.environ.get('JCTC_HAMILTONIAN_OUTPUT'):
        np.savez_compressed(os.environ['JCTC_HAMILTONIAN_OUTPUT'], hamiltonian_Ha=np.asarray(matrices), occupations=occ, temperature_Ha=temperature)
    band_diagonal = np.asarray(band_diagonal)
    ne = float(occ.sum())
    mu_fd = brentq(lambda mu: float(np.sum(cap*expit((mu-band_diagonal)/temperature))-ne),
                   float(band_diagonal.min()-100*temperature),
                   float(band_diagonal.max()+100*temperature), xtol=1e-14)
    f_fd = cap*expit((mu_fd-band_diagonal)/temperature)

    def energy_at_f(f_value):
        density = pw.density_grid(coeff, crystal.vol, f_value)
        reciprocal_density = jnp.fft.fftn(density, axes=(-3,-2,-1))
        return (energy.kinetic(gvec, kvec, coeff, f_value)
                + energy.hartree(reciprocal_density, gvec, crystal.vol)
                + d3pp.energy_local(reciprocal_density, vloc, vol=crystal.vol)
                + d3pp.energy_nonlocal(coeff, projectors, vol=crystal.vol,
                    occupation=f_value, d_matrix=out.nl_d_matrix, freq_mask=mask)
                + energy.xc_energy(density, gvec, crystal.vol, cfg.xc,
                    kohn_sham=False, core_density_grid=out.core_density_grid))

    f_jax = jnp.asarray(occ)
    occupation_energy_gradient = np.asarray(jax.grad(energy_at_f)(f_jax))
    energy_gradient_error = float(np.max(abs(occupation_energy_gradient[0]-band_diagonal)))
    free_indices = np.flatnonzero((occ.reshape(-1)>1e-4) & (occ.reshape(-1)<cap-1e-4))
    fd_direction_error = None
    if len(free_indices) >= 2:
        direction = np.zeros(occ.size)
        direction[free_indices[0]] = 1
        direction[free_indices[1]] = -1
        direction = jnp.asarray(direction.reshape(occ.shape))
        h = 1e-5
        difference = float((energy_at_f(f_jax+h*direction)-energy_at_f(f_jax-h*direction))/(2*h))
        analytic_direction = float(np.sum(occupation_energy_gradient*np.asarray(direction)))
        fd_direction_error = abs(difference-analytic_direction)
    ew = get_ewald_coulomb_repulsion(cfg, charges=np.asarray(pp.valence_charges, dtype=float))
    final_energy = float(energy_at_f(f_jax)+ew)
    entropy_constant = -occ.size*cap*np.log(cap)
    physical_entropy = float(entropy.fermi_dirac(f_jax, eps=cfg.eps))-entropy_constant
    q = band_diagonal - temperature * entropy_grad[0]
    f = occ[0]
    margin = min(1e-5, cap * 1e-3)
    interior = (f > margin) & (f < cap - margin)
    lower = f <= margin
    upper = f >= cap - margin
    if np.any(interior):
        mu = float(np.median(q[interior]))
        inside = float(np.max(np.abs(q[interior] - mu)))
    else:
        mu = float(np.median(q))
        inside = None
    lower_violation = float(np.max(np.maximum(mu - q[lower], 0))) if np.any(lower) else 0.0
    upper_violation = float(np.max(np.maximum(q[upper] - mu, 0))) if np.any(upper) else 0.0
    return {
        "scope": "fixed-density endpoint residuals with occupation energy-gradient parity; not a global-minimum certificate",
        "endpoint_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "final_state_energy_Ha": final_energy,
        "final_state_free_energy_physical_Ha": final_energy-temperature*physical_entropy,
        "reported_energy_minus_final_state_Ha": float(out.total_energy)-final_energy,
        "occupation_energy_gradient_vs_ks_diagonal_max_Ha": energy_gradient_error,
        "occupation_energy_fd_direction_error_Ha": fd_direction_error,
        "fermi_dirac_occupation_max_error": float(np.max(abs(occ[0]-f_fd))),
        "fermi_dirac_mu_Ha": float(mu_fd),
        "orbital_outside_max_Ha": max(orbital_outside),
        "orbital_outside_unweighted_max_Ha": max(orbital_unweighted),
        "orbital_commutator_max_Ha": max(orbital_commutator),
        "orbital_orthonormality_max": max(orthonormality),
        "band_hamiltonian_antihermitian_max_Ha": max(orbital_hermiticity),
        "occupation_kkt_interior_max_Ha": inside,
        "occupation_kkt_lower_violation_max_Ha": lower_violation,
        "occupation_kkt_upper_violation_max_Ha": upper_violation,
        "occupation_mu_Ha": mu,
        "interior_count": int(np.sum(interior)),
        "lower_count": int(np.sum(lower)),
        "upper_count": int(np.sum(upper)),
        "diagonal_ks_energy_Ha": band_diagonal.tolist(),
        "occupation_stationarity_q_Ha": q.tolist(),
    }
