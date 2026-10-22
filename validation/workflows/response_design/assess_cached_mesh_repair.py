"""Assess a reused FD correction candidate; never submit or relabel a GPU run."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[3]
R = P / "validation/results"
HASHES = {}


def sha(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = digest
    return digest


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def audit_spectrum(path):
    spectrum = read(path)
    source = Path(spectrum["source"])
    point = read(source / "point.json")
    meta = read(source / "metadata.json")
    assert point["stationarity_pass"] and point["residual"] <= 1e-6
    assert point["acoustic_sum_rule_Ha_Bohr"] <= 5e-8
    assert np.linalg.norm(np.sum(point["forces_Ha_Bohr"], axis=0)) <= 5e-8
    assert abs(point["integrated_valence_density"] - 96) <= 0.00096
    assert sha(source / "density.npy") == spectrum["density_sha256"]
    assert sha(source / "mask.npy") == spectrum["mask_sha256"]
    points = spectrum["points"]
    assert sorted(x["index"] for x in points) == list(range(len(point["k_fractional"])))
    assert spectrum["full_mesh"] and spectrum["all_registered_gates_passed"]
    for row in points:
        eigenvalues = row["eigenvalues_Ha"]
        assert np.isfinite(eigenvalues).all() and np.isfinite(row["residuals_Ha"]).all()
        assert max(row["residuals_Ha"]) <= 1e-7
        assert np.allclose(row["k_frac"], point["k_fractional"][row["index"]], atol=1e-10, rtol=0)
        assert abs(eigenvalues[47] - row["vbm_Ha"]) < 1e-13
        assert abs(eigenvalues[48] - row["cbm_Ha"]) < 1e-13
    factor = 27.211386245988
    sampled = (min(x["cbm_Ha"] for x in points) - max(x["vbm_Ha"] for x in points)) * factor
    gamma = next(x for x in points if max(abs(v) for v in x["k_frac"]) < 1e-10)
    gg = (gamma["cbm_Ha"] - gamma["vbm_Ha"]) * factor
    assert abs(sampled - spectrum["gap_eV"]) < 1e-12
    assert abs(gg - spectrum["gamma_gap_eV"]) < 1e-12
    return spectrum, point, meta


def main():
    design = R / "response_design_20261004_response_design"
    protocol = read(design / "protocol.json")
    reverse = read(design / "reverse_history.json")[-1]
    target = protocol["target_gamma_eV"]
    historical = R / "response_design_20261003_k2_target_refinement_v2"
    proposal = read(historical / "correction_proposal.json")
    minus, _, _ = audit_spectrum(historical / "spectra/endpoint_fd_minus_k2_complete.json")
    plus, _, _ = audit_spectrum(historical / "spectra/endpoint_fd_plus_k2_complete.json")
    slope = (plus["gamma_gap_eV"] - minus["gamma_gap_eV"]) / (2 * proposal["h"])
    assert abs(slope - proposal["slope_Gamma_eV"]) < 1e-12
    source2, _, _ = audit_spectrum(R / "response_design_20261004_reverse_endpoint_k2/result.json")
    assert source2["theta"] == reverse["theta"]
    assert source2["fixed_original_target_eV"] == target == proposal["original_target_eV"]
    step = (target - source2["gamma_gap_eV"]) / slope
    nominal_t = reverse["t"] + step
    selected_t = proposal["t_new"]
    assert abs(step) <= protocol["design_max_step"]
    assert abs(selected_t) <= protocol["design_bound"]
    comparisons = []
    for k, candidate_path in [
        (2, historical / "spectra/corrected_endpoint_k2_complete.json"),
        (4, R / "response_design_20261004_corrected_k4_spectrum_recovery/corrected_endpoint_k4_complete.json"),
    ]:
        original, point, meta = audit_spectrum(R / f"response_design_20261004_reverse_endpoint_k{k}/result.json")
        candidate, cpoint, cmeta = audit_spectrum(candidate_path)
        assert original["theta"] == reverse["theta"]
        assert original["mask_sha256"] == candidate["mask_sha256"]
        assert meta["code_sha256"] == cmeta["code_sha256"]
        assert meta["source_sha256"] == cmeta["source_sha256"]
        for setting in ["symbols", "spin", "cutoff", "grid", "k", "empty", "xc", "pp", "xc_grid_factor", "local_radial_quadrature", "radial_patch_sha256"]:
            assert meta["physical"][setting] == cmeta["physical"][setting]
        for setting in ["cutoff", "grid", "k", "xc_grid_factor", "local_radial_quadrature", "structure", "pp_dir", "displacements"]:
            assert meta["arguments"][setting] == cmeta["arguments"][setting]
        assert len(point["k_fractional"]) == len(cpoint["k_fractional"]) == k**3
        assert np.allclose(point["k_fractional"], cpoint["k_fractional"], atol=1e-10, rtol=0)
        fractions = np.asarray(point["positions_bohr"]) @ np.linalg.inv(point["cell_bohr"])
        cfractions = np.asarray(cpoint["positions_bohr"]) @ np.linalg.inv(cpoint["cell_bohr"])
        assert np.max(abs(fractions - cfractions)) < 1e-10
        v = np.asarray(protocol["theta0"]) + selected_t * np.asarray(protocol["direction"])
        strain = [[v[0], v[5]/2, v[4]/2], [v[5]/2, v[1], v[3]/2], [v[4]/2, v[3]/2, v[2]]]
        assert np.allclose(json.loads(cmeta["arguments"]["strain"]), strain, atol=1e-15, rtol=0)
        before, after = original["gamma_gap_eV"] - target, candidate["gamma_gap_eV"] - target
        assert abs(after) < abs(before) and abs(after) <= protocol["design_tolerance_eV"]
        comparisons.append(dict(mesh=k, cached_candidate_source=str(candidate_path),
                                original_reverse_error_meV=1000*before, cached_candidate_error_meV=1000*after,
                                fixed_target_passed=True, numerical_source_gates_passed=True,
                                fraction_of_original_target_loss_removed=1-(after/before)**2))
    report = dict(time=datetime.datetime.now().astimezone().isoformat(),
                  state="existing_FD_candidate_audited_for_possible_postprocessing",
                  fixed_target_eV=target, target_tolerance_meV=1000*protocol["design_tolerance_eV"],
                  actual_reverse_Gamma_terminal_t=reverse["t"],
                  historical_FD_slope_origin_t=proposal["t_old"],
                  slope_origin_distance_from_actual_reverse_t=reverse["t"]-proposal["t_old"],
                  reused_selfconsistent_FD_slope_eV_per_t=slope,
                  nominal_Newton_t_from_actual_reverse_error=nominal_t,
                  exact_cached_candidate_t=selected_t,
                  nominal_minus_cached_candidate_t=nominal_t-selected_t,
                  comparisons=comparisons, source_sha256=HASHES,
                  new_GPU_calculations=False, reverse_history_modified=False,
                  dense_mesh_reverse_adjoint_implemented=False,
                  scope="A previously computed FD candidate, not a new reverse update. Cache reuse is valid only for the exact selected geometry and recorded physical/numerical inputs. Original reverse endpoint failures remain unchanged.")
    output = R / "response_design_20261005_mesh_repair_assessment"
    output.mkdir(exist_ok=True)
    (output / "assessment.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
