"""Build an auditable inventory of completed Phase 3 scientific artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path


FINAL_BASELINES = {
    "Si": "evaluate_Si_k10_h005",
    "diamond": "evaluate_diamond_g80_c100_h0025_k12_4gpu_remat_platform_233252",
    "BN": "evaluate_BN_g60_c100_xc4_h0025_k10",
    "AlAs": "evaluate_AlAs_xc4_h005_k12_8gpu_remat_platform_233254",
    "SiC": "evaluate_SiC_xc8_h0025_k12",
    "AlP": "evaluate_AlP_xc4_k10_8gpu_memfix_233254",
    "GaP": "evaluate_GaP_g72_c100_xc2_h005_k12_8gpu_remat_platform_233254",
    "GaAs": "evaluate_GaAs_g72_c100_xc2_h0025",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def material_from_name(name: str) -> str:
    for material in ("diamond", "AlAs", "AlP", "GaAs", "GaP", "SiC", "BN", "Si", "Ge"):
        if material in name:
            return material
    return ""


def certificate_state(data: dict) -> str:
    value = data.get(
        "passed",
        data.get("pass", data.get("overall_pass", data.get("certificate_pass"))),
    )
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return "diagnostic"


def append(rows: list[dict], base: Path, category: str, status: str, path: Path, **fields) -> None:
    tag = fields.get("tag", path.parent.name if path.name == "summary.json" else path.stem)
    rows.append(
        {
            "category": category,
            "status": status,
            "material": fields.get("material") or material_from_name(tag),
            "tag": tag,
            "path": str(path.relative_to(base)),
            "details": json.dumps(fields.get("details", {}), sort_keys=True, separators=(",", ":")),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    final_rows: list[dict] = []
    final_tags = set(FINAL_BASELINES.values())

    for path in sorted(base.glob("evaluate_*/summary.json")):
        data = load(path)
        status = "final_baseline_completed" if path.parent.name in final_tags else "supporting_completed"
        append(
            rows,
            base,
            "d3ft_evaluate",
            status,
            path,
            material=data.get("material", ""),
            details={"points": len(load(path.parent / "points.json")) if (path.parent / "points.json").is_file() else None},
        )

    for material, tag in FINAL_BASELINES.items():
        path = base / tag / "summary.json"
        if not path.is_file():
            continue
        data = load(path)
        config = load(base / tag / "metadata.json").get("config", {})
        final_rows.append(
            {
                "material": material,
                "role": "external_holdout" if material == "GaAs" else "fit_candidate",
                "tag": tag,
                "grid": config.get("grid_sizes"),
                "cutoff_Ha": config.get("cutoff_energy"),
                "k": config.get("k_grid_sizes"),
                "strain": data.get("strain_step"),
                "a0_A": data.get("a0_A"),
                "B_GPa": data.get("B_GPa"),
                "Bprime": data.get("Bprime"),
                "D_GPa": data.get("C11_minus_C12_GPa"),
                "C44_GPa": data.get("C44_GPa"),
                "equilibrium_residual": data.get("equilibrium_residual"),
            }
        )

    for path in sorted(base.glob("*certificate.json")):
        data = load(path)
        append(rows, base, "certificate", certificate_state(data), path, material=data.get("material", ""))

    for path in sorted((base / "spectra").rglob("summary.json")):
        data = load(path)
        if data.get("full_mesh") is not True:
            state = "probe"
        elif data.get("mesh_fixed_occupations_passed") is not True:
            state = "failed"
        elif "production_20260913" in path.parts:
            state = "final_pass"
        else:
            state = "supporting_pass"
        append(
            rows,
            base,
            "spectrum",
            state,
            path,
            details={
                "gap_eV": data.get("gap_eV"),
                "max_eigen_residual_Ha": data.get("max_eigen_residual_Ha"),
                "solver": data.get("solver"),
            },
        )

    for prefix, category in (
        ("qe_bulk_*", "qe_bulk"),
        ("qe_elastic_*", "qe_elastic"),
        ("qe_force_*", "qe_force"),
        ("stationarity_*", "stationarity"),
    ):
        for directory in sorted(p for p in base.glob(prefix) if p.is_dir()):
            path = directory / "summary.json"
            if path.is_file():
                append(rows, base, category, "completed", path)

    for path in sorted(base.glob("evaluate_*/equilibrium_jacobian.json")):
        append(rows, base, "equilibrium_jacobian", "completed", path)

    for path in sorted(base.glob("*comparison.json")):
        append(rows, base, "comparison", "completed_diagnostic", path)

    rows.sort(key=lambda row: (row["category"], row["status"], row["material"], row["tag"]))
    final_rows.sort(key=lambda row: row["material"])

    inventory_path = out / "completed_data_inventory.csv"
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "status", "material", "tag", "path", "details"])
        writer.writeheader()
        writer.writerows(rows)

    baselines_path = out / "final_baselines.csv"
    with baselines_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["category"], row["status"])
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "generated": dt.datetime.now().astimezone().isoformat(),
        "base": str(base),
        "definitions": {
            "completed": "A summary or derived result exists and parses as JSON.",
            "passed": "The artifact explicitly records passed=true (or equivalent).",
            "final_baseline": "The latest selected material setting, not a completed nonlinear fit.",
            "final_spectrum": "A production_20260913 full mesh with fixed occupations passed.",
        },
        "counts": [
            {"category": category, "status": status, "count": count}
            for (category, status), count in sorted(counts.items())
        ],
        "final_baselines": final_rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    final_spectra = [
        row for row in rows if row["category"] == "spectrum" and row["status"] == "final_pass"
    ]
    report = [
        "# Phase 3 completed-data inventory",
        "",
        f"Generated: {manifest['generated']}",
        "",
        "This snapshot counts parseable completed artifacts. Categories overlap: certificates,",
        "comparisons, Jacobians, and spectra are derived from calculation directories and are not",
        "independent calculations. Failed certificates and diagnostic probes are retained but are",
        "not counted as accepted scientific evidence.",
        "",
        "## Artifact counts",
        "",
        "| Category | Status | Count |",
        "|---|---|---:|",
    ]
    report.extend(
        f"| {entry['category']} | {entry['status']} | {entry['count']} |"
        for entry in manifest["counts"]
    )
    report.extend(
        [
            "",
            "## Selected final-setting baselines",
            "",
            "| Material | Role | grid/cutoff/k | strain | a0 (A) | B (GPa) | B' | D (GPa) | C44 (GPa) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        fmt = lambda value: "" if value is None else f"{value:.6g}"
        report.append(
            f"| {row['material']} | {row['role']} | {row['grid']}/{row['cutoff_Ha']}/{row['k']} | "
            f"{fmt(row['strain'])} | {fmt(row['a0_A'])} | {fmt(row['B_GPa'])} | "
            f"{fmt(row['Bprime'])} | {fmt(row['D_GPa'])} | {fmt(row['C44_GPa'])} |"
        )
    report.extend(
        [
            "",
            "AlP contributes only a0 to the fit target; its elastic quantities remain diagnostic.",
            "GaAs is the external holdout. Missing C44 values for GaP/GaAs reflect the selected",
            "force-validity scope rather than a parser omission.",
            "",
            "## Final-setting production spectra passed",
            "",
            "| Material | Tag | Gap (eV) | Max residual (Ha) |",
            "|---|---|---:|---:|",
        ]
    )
    for row in final_spectra:
        details = json.loads(row["details"])
        report.append(
            f"| {row['material']} | {row['tag']} | {details['gap_eV']:.8g} | "
            f"{details['max_eigen_residual_Ha']:.4g} |"
        )
    report.extend(
        [
            "",
            "The complete row-level inventory is in `completed_data_inventory.csv`; selected",
            "material baselines are in `final_baselines.csv`; machine-readable definitions and",
            "counts are in `manifest.json`.",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(report))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
