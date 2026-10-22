"""Opt-in physical electron-count acceptance; never rescales a density."""
import json
import math
from pathlib import Path


def check_charge(charge, expected, tolerance=1e-6, *, step, report_path=None):
    charge, expected = float(charge), float(expected)
    passed = math.isfinite(charge) and abs(charge - expected) <= tolerance
    row = dict(step=int(step), charge=charge if math.isfinite(charge) else None,
               expected_electrons=expected, absolute_tolerance=float(tolerance),
               passed=passed, quantity="integral of the actual occupied FFT density")
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print("D3FT_CHARGE_GATE " + json.dumps(row), flush=True)
    if not passed:
        raise ValueError(f"Physical charge gate failed at step {step}: "
                         f"{charge:.16g} != {expected:.16g} +/- {tolerance:g}")
    return row
