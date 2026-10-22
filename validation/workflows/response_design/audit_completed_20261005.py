"""Read-only independent audit of the October 4 overnight response campaign."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[3]
R = P / "validation/results"
D = R / "response_design_20261004_response_design"
HASHES = {}


def sha(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def audit_mesh(k, reverse_theta, target):
    root = R / f"response_design_20261004_reverse_endpoint_k{k}"
    cfg = read(root / "protocol.json")
    result = read(root / "result.json")
    status = read(root / "status.json")
    source = root / "runs/endpoint"
    parent = Path(cfg["parent"])
    point = read(source / "point.json")
    meta = read(source / "metadata.json")
    assert status["stage"] == "completed"
    assert np.allclose(result["theta"], reverse_theta, atol=1e-15, rtol=0)
    assert point["stationarity_pass"] and point["residual"] <= 1e-6
    assert point["acoustic_sum_rule_Ha_Bohr"] <= 5e-8
    assert abs(point["integrated_valence_density"] - 96) <= 0.00096
    assert len(point["k_fractional"]) == k**3
    assert sha(source / "mask.npy") == sha(parent / "mask.npy") == result["mask_sha256"]
    assert meta["code_sha256"] == read(parent / "metadata.json")["code_sha256"]
    density_sha = sha(source / "density.npy")
    assert density_sha == result["density_sha256"]
    assert all(x["returncode"] == 0 for x in read(root / "exit_codes.json"))
    rows = []
    for shard in cfg["shards"]:
        output, view = Path(shard["output"]), Path(shard["view"])
        smeta = read(output / "metadata.json")
        assert smeta["indices"] == shard["indices"]
        assert smeta["density_sha256"] == density_sha
        assert smeta["occupied_bands"] == 48
        assert smeta["solver"] == "block" and smeta["lightweight_reconstruction"]
        assert len(smeta["mesh"]) == k**3
        assert sha(view / "metadata.json") == smeta["source_metadata_sha256"]
        bindings = read(view / "metadata.json")["generic_source_binding"]
        for name in ["point.json", "metadata.json", "density.npy", "mask.npy", "input.xyz"]:
            assert bindings[str(source / name)] == sha(source / name)
        for index in shard["indices"]:
            row = read(output / f"k_{index:04d}.json")
            assert row["index"] == index
            e = row["eigenvalues_Ha"]
            residuals = row["residuals_Ha"]
            assert len(e) >= 49 and np.isfinite(e).all() and np.isfinite(residuals).all()
            assert np.min(np.diff(e)) >= -1e-12
            assert max(residuals) <= 1e-7
            assert abs(e[47] - row["vbm_Ha"]) < 1e-13
            assert abs(e[48] - row["cbm_Ha"]) < 1e-13
            assert np.allclose(row["k_frac"], point["k_fractional"][index], atol=1e-10, rtol=0)
            rows.append(row)
    rows.sort(key=lambda x: x["index"])
    assert [x["index"] for x in rows] == list(range(k**3))
    assert rows == result["points"]
    factor = 27.211386245988
    gap = (min(x["cbm_Ha"] for x in rows) - max(x["vbm_Ha"] for x in rows)) * factor
    gamma = next(x for x in rows if max(abs(y) for y in x["k_frac"]) < 1e-10)
    gamma_gap = (gamma["cbm_Ha"] - gamma["vbm_Ha"]) * factor
    old = read(cfg["start_spectrum"])
    error = gamma_gap - target
    assert abs(gap - result["gap_eV"]) < 1e-12
    assert abs(gamma_gap - result["gamma_gap_eV"]) < 1e-12
    assert abs(error - result["target_transfer_error_eV"]) < 1e-12
    assert result["fixed_original_target_eV"] == target
    assert result["target_transfer_passed"] == (abs(error) <= 0.001)
    assert abs(gap - old["gap_eV"] - result["mesh_gap_gain_eV"]) < 1e-12
    return dict(completed=True, time=result["time"], numerical_source_gates_passed=True,
                points=len(rows), gamma_gap_eV=gamma_gap, sampled_mesh_gap_eV=gap,
                sampled_mesh_gap_gain_eV=gap-old["gap_eV"], original_target_error_meV=error*1000,
                original_target_transfer_passed=abs(error) <= 0.001,
                electronic_residual=point["residual"], raw_ASR_Ha_Bohr=point["acoustic_sum_rule_Ha_Bohr"],
                integrated_valence_density=point["integrated_valence_density"],
                max_eigen_residual_Ha=max(max(x["residuals_Ha"]) for x in rows),
                scope="Actual reverse-designed geometry, stationary density on its own mesh; target failure retained.")


def main():
    now = datetime.datetime.now().astimezone()
    protocol = read(D / "protocol.json")
    target = protocol["target_gamma_eV"]
    reverse = read(D / "reverse_history.json")
    forward = read(D / "forward_history.json")
    native = read(D / "native_path_v4/history.json")
    trajectories = {}
    for name, history, status_path in [
        ("reverse", reverse, D / "reverse_status.json"),
        ("forward", forward, D / "forward_status.json"),
        ("native", native, D / "native_path_v4/status.json"),
    ]:
        status = read(status_path)
        assert status["stage"] == "completed" and status["converged"] and status["updates"] == 3
        assert len(history) == 4
        assert all(x["actual_loss_accepted"] for x in history[1:])
        errors = [abs(x["gap_eV"] - target) for x in history]
        assert all(b < a for a, b in zip(errors, errors[1:]))
        assert errors[-1] <= protocol["design_tolerance_eV"]
        if name == "native":
            assert all(max(x["residuals_Ha"]) <= 1e-7 for x in history[1:])
            assert all(x["plane_waves"] == 56087 for x in history[1:])
        trajectories[name] = dict(completed=True, time=status["time"], updates=3, actual_objective_monotonically_improves=True,
                                  t=history[-1]["t"], gap_eV=history[-1]["gap_eV"],
                                  original_target_error_meV=(history[-1]["gap_eV"]-target)*1000,
                                  Gamma_target_passed=True)
    adapter = read(D / "native_path_v4/adapter_checks.json")
    assert adapter["passed"] and adapter["primal_and_dual_fixed_basis_checked"]
    native_responses = []
    for i in [1, 2]:
        response = read(D / f"native_path_v4/response_iter{i}.json")
        assert np.isfinite(response["directional_gap_derivative_eV"])
        assert all(x["converged"] and np.isfinite(x["residual_norm"]) for x in response["response"])
        native_responses.append(response)
    endpoints = {f"k{k}": audit_mesh(k, reverse[-1]["theta"], target) for k in [2, 4]}
    resources = []
    for job, cards in [(247061, 4), (250212, 4), (250285, 8)]:
        command = ["timeout", "25s", "srun", f"--jobid={job}", "--overlap", "--cpu-bind=none",
                   "-N1", "-n1", "--cpus-per-task=1", f"--gres=gpu:{cards}", "--gpu-bind=none",
                   "bash", "-c", "hostname; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"]
        result = subprocess.run(command, capture_output=True, text=True)
        resources.append(dict(jobid=job, cards=cards, checked_at=datetime.datetime.now().astimezone().isoformat(),
                              returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    audit = dict(time=now.isoformat(), fixed_target_eV=target, target_tolerance_meV=1,
                 trajectories=trajectories, native_adapter=adapter, native_responses=native_responses,
                 reverse_native_final_gap_difference_eV=abs(reverse[-1]["gap_eV"]-native[-1]["gap_eV"]),
                 reverse_native_final_t_difference=abs(reverse[-1]["t"]-native[-1]["t"]),
                 comparison_scope="Distinct optimizer endpoints; final observable agreement, not same-geometry derivative agreement.",
                 endpoints=endpoints, resources=resources, source_sha256=HASHES,
                 audit_recomputation_passed=True, all_scientific_targets_passed=False,
                 new_jobs_submitted=False, next_steps=["Mesh-aware correction and dense recheck, explicitly identify any FD correction.",
                                                     "Merge completed native trajectory and elastic data into manuscript figures/text; retain failed transfer."])
    output = D / f"progress_audit_{now:%Y%m%d_%H%M}.json"
    output.write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    print(output)
    print(json.dumps({k: v for k, v in audit.items() if k not in ["source_sha256", "native_responses", "native_adapter"]}, indent=2))


if __name__ == "__main__":
    main()
