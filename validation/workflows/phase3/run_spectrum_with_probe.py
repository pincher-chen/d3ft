"""Run a one-k-point occupation probe before a full SCF-mesh spectrum check."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def validate_result(source, output, full):
    """A successful process exit alone is not an occupation certificate."""
    summary = json.loads((output / 'summary.json').read_text())
    meta = json.loads((output / 'metadata.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if (Path(meta['source']).resolve() != source.resolve()
            or meta['source_metadata_sha256'] != digest(source / 'metadata.json')):
        raise ValueError('Spectrum belongs to a different source or source revision')
    points = json.loads((source / 'points.json').read_text())
    eq = [p for p in points if p['label'].startswith('eq_')][-1]
    if not meta['density_sha256'] == eq['density_sha256'] == digest(source / eq['density_file']):
        raise ValueError('Spectrum density provenance mismatch')
    gap, residual = summary['gap_eV'], summary['max_eigen_residual_Ha']
    if (summary['solver'] != 'block' or not math.isfinite(gap)
            or not math.isfinite(residual) or not 0 <= residual <= 1e-7
            or gap <= 2e-7 * 27.211386245988):
        raise ValueError('Spectrum gap or eigen-residual gate failed')
    if full and not (summary.get('full_mesh') is True
                     and summary.get('mesh_fixed_occupations_passed') is True):
        raise ValueError('Full mesh fixed occupations did not pass')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--probe-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--accept-existing",
        type=Path,
        help="Skip the retry when this directory already contains a passing full-mesh summary.",
    )
    args = parser.parse_args()

    for required in ("metadata.json", "points.json", "summary.json", "mask.npy"):
        if not (args.source / required).is_file():
            raise FileNotFoundError(args.source / required)
    for output in (args.probe_out, args.out, args.record):
        if output.exists():
            raise FileExistsError(output)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    if args.accept_existing is not None:
        summary_path = args.accept_existing / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            if (
                summary.get("full_mesh") is True
                and summary.get("mesh_fixed_occupations_passed") is True
            ):
                summary = validate_result(args.source, args.accept_existing, full=True)
                args.record.write_text(
                    json.dumps(
                        {
                            "state": "skipped_existing_pass",
                            "accepted_summary": str(summary_path.resolve()),
                            "summary": summary,
                            "finished": datetime.datetime.now().isoformat(),
                        },
                        indent=2,
                    )
                )
                print(
                    json.dumps(
                        {
                            "state": "skipped_existing_pass",
                            "accepted_summary": str(summary_path.resolve()),
                        }
                    ),
                    flush=True,
                )
                return

    checker = Path(__file__).with_name("check_occupations.py")
    probe = [
        sys.executable,
        "-u",
        str(checker),
        str(args.source),
        "--out",
        str(args.probe_out),
        "--indices",
        "[0]",
        "--solver",
        "block",
    ]
    full = [
        sys.executable,
        "-u",
        str(checker),
        str(args.source),
        "--out",
        str(args.out),
        "--solver",
        "block",
    ]
    policy_path=Path(__file__).with_name('spectrum_sharding_policy.json')
    sharding_validation=None
    if policy_path.exists():
        policy=json.loads(policy_path.read_text())
        source_meta=json.loads((args.source/'metadata.json').read_text())
        material=json.loads((args.source/'summary.json').read_text())['material']
        validation=policy.get('validated_materials',{}).get(material)
        if validation:
            proof=json.loads(Path(validation).read_text())
            if proof.get('passed') is not True:raise ValueError('Sharding validation did not pass')
            for name,digest in proof['code_sha256'].items():
                if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Validated sharding code changed: '+name)
            baseline=json.loads((Path(proof['source'])/'metadata.json').read_text())
            for name in ('grid_sizes','cutoff_energy','k_grid_sizes','empty_bands'):
                if baseline['config'][name]!=source_meta['config'][name]:
                    raise ValueError('Sharding validation settings changed: '+name)
            if baseline['pseudopotentials']!=source_meta['pseudopotentials']:
                raise ValueError('Sharding validation pseudopotentials changed')
            if baseline['config']['d3ft'].get('xc_grid_factor',1)!=source_meta['config']['d3ft'].get('xc_grid_factor',1):
                raise ValueError('Sharding validation XC grid changed')
            devices=os.environ.get('CUDA_VISIBLE_DEVICES','')
            if not devices:raise ValueError('Sharding requires explicit visible devices')
            full=[sys.executable,'-u',str(Path(__file__).with_name('sharded_spectrum.py')),
                  str(args.source),'--out',str(args.out),'--devices',devices]
            sharding_validation=validation
    state = {
        "source": str(args.source.resolve()),
        "probe_command": probe,
        "full_command": full,
        "started": datetime.datetime.now().isoformat(),
        "sharding_validation": sharding_validation,
    }

    def write_status(**fields):
        state.update(fields, updated=datetime.datetime.now().isoformat())
        temporary = args.record.with_suffix(args.record.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(args.record)

    write_status(state="probe_running")
    result = subprocess.run(probe)
    write_status(state="probe_exited", probe_returncode=result.returncode)
    if result.returncode:
        raise SystemExit(result.returncode)
    try:
        validate_result(args.source, args.probe_out, full=False)
    except (ValueError, KeyError, OSError, IndexError) as exc:
        write_status(state='probe_gate_failed', error=str(exc))
        raise SystemExit(2)

    write_status(state="full_running")
    result = subprocess.run(full)
    if result.returncode == 0:
        try:
            validate_result(args.source, args.out, full=True)
        except (ValueError, KeyError, OSError, IndexError) as exc:
            write_status(state='full_gate_failed', full_returncode=0, error=str(exc))
            raise SystemExit(2)
    write_status(
        state="complete" if result.returncode == 0 else "full_failed",
        full_returncode=result.returncode,
        finished=datetime.datetime.now().isoformat(),
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
