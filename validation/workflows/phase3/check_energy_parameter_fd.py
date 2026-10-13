"""Certify stationary total-energy parameter derivatives from a central-FD plan."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


STATE_RECORDS = {
    "zero": "eq_1.json",
    "iso_p005": "iso_+0.005000.json",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    base = Path(plan["base"])
    base_meta = json.loads((base / "metadata.json").read_text())
    base_mask = np.load(base / "mask.npy")
    samples = {}
    input_fingerprints = {
        str(args.plan.resolve()): sha256(args.plan),
        str((base / "metadata.json").resolve()): sha256(base / "metadata.json"),
        str((base / "mask.npy").resolve()): sha256(base / "mask.npy"),
    }

    for item in plan["samples"]:
        run = args.plan.parent / item["tag"]
        summary_path = run / "summary.json"
        metadata_path = run / "metadata.json"
        summary = json.loads(summary_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        expected = np.zeros(7)
        expected[item["parameter_index"]] = item["delta"]
        np.testing.assert_allclose(metadata["parameter_vector"], expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(summary["strain"], item["strain"], rtol=0, atol=1e-14)
        if summary["residual"] > 1e-6:
            raise ValueError(f"{item['tag']}: stationarity failed")
        if not np.array_equal(np.load(run / "mask.npy"), base_mask):
            raise ValueError(f"{item['tag']}: Miller basis changed")
        for key in ("grid_sizes", "cutoff_energy", "k_grid_sizes", "empty_bands"):
            if metadata["config"][key] != base_meta["config"][key]:
                raise ValueError(f"{item['tag']}: {key} changed")
        if metadata["pseudopotentials"] != base_meta["pseudopotentials"]:
            raise ValueError(f"{item['tag']}: pseudopotential changed")
        samples[(item["state"], item["parameter_index"], item["sign"])] = summary
        input_fingerprints[str(summary_path.resolve())] = sha256(summary_path)
        input_fingerprints[str(metadata_path.resolve())] = sha256(metadata_path)

    states = []
    for state in sorted({item["state"] for item in plan["samples"]}):
        reference_path = base / STATE_RECORDS[state]
        reference = json.loads(reference_path.read_text())
        ad = np.asarray(reference["dE_dp"], float)
        fd = np.empty(7)
        residuals = []
        rows = []
        for index, name in enumerate(plan["parameter_names"]):
            minus = samples[(state, index, -1)]
            plus = samples[(state, index, 1)]
            step = plan["alpha"] * plan["parameter_scales"][index]
            fd[index] = (plus["energy_Ha"] - minus["energy_Ha"]) / (2 * step)
            residuals.extend([minus["residual"], plus["residual"]])
        norm = float(np.linalg.norm(fd))
        for index, name in enumerate(plan["parameter_names"]):
            error = abs(ad[index] - fd[index])
            denominator = max(abs(fd[index]), 1e-3 * norm)
            relative_error = error / denominator if denominator else error
            rows.append(
                {
                    "parameter_index": index,
                    "parameter_name": name,
                    "ad_dE_dp": float(ad[index]),
                    "fd_dE_dp": float(fd[index]),
                    "absolute_error": float(error),
                    "relative_error": float(relative_error),
                    "passed": bool(relative_error <= args.rtol),
                }
            )
        input_fingerprints[str(reference_path.resolve())] = sha256(reference_path)
        states.append(
            {
                "state": state,
                "reference_record": str(reference_path.resolve()),
                "strain": reference["strain"],
                "fd_gradient_norm": norm,
                "max_sample_residual": max(residuals),
                "passed": all(row["passed"] for row in rows),
                "rows": rows,
            }
        )

    result = {
        "test": "Si stationary total-energy parameter AD versus central FD",
        "scope": "alpha=0.04 pilot; alpha-squared trend requires alpha=0.02 and 0.08",
        "plan": str(args.plan.resolve()),
        "alpha": plan["alpha"],
        "rtol": args.rtol,
        "passed": all(state["passed"] for state in states),
        "states": states,
        "input_fingerprints": input_fingerprints,
    }
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
