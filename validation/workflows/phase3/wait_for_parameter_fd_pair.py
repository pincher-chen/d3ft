"""Wait for two GPU sequences, then certify a re-equilibrated parameter FD pair."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("minus", type=Path)
    parser.add_argument("plus", type=Path)
    parser.add_argument("--minus-record", type=Path, required=True)
    parser.add_argument("--plus-record", type=Path, required=True)
    parser.add_argument("--minus-pid", type=int, required=True)
    parser.add_argument("--plus-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    args.record.mkdir(parents=True, exist_ok=False)

    def status(state, **fields):
        payload = {
            "state": state,
            "time": datetime.datetime.now().astimezone().isoformat(),
            **fields,
        }
        temporary = args.record / "status.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.record / "status.json")

    dependencies = [
        ("minus", args.minus_record, args.minus_pid),
        ("plus", args.plus_record, args.plus_pid),
    ]
    status(
        "waiting_for_sequences",
        dependencies=[str(record.resolve()) for _, record, _ in dependencies],
    )
    while True:
        complete = []
        for label, record, pid in dependencies:
            state_path = record / "status.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get("state") == "sequence_exited":
                failed = [row for row in state.get("results", []) if row["returncode"]]
                if failed:
                    status("dependency_failed", dependency=label, failures=failed)
                    raise SystemExit(1)
                complete.append(label)
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                status(
                    "dependency_controller_missing",
                    dependency=label,
                    dependency_state=state,
                )
                raise SystemExit(1)
        if len(complete) == len(dependencies):
            break
        status("waiting_for_sequences", completed=complete)
        time.sleep(args.poll_seconds)

    command = [
        sys.executable,
        str(Path(__file__).with_name("check_parameter_fd.py")),
        str(args.base),
        str(args.minus),
        str(args.plus),
        "--out",
        str(args.out),
        "--rtol",
        str(args.rtol),
    ]
    (args.record / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    status("certifying")
    with (args.record / "certificate.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    status(
        "complete" if result.returncode == 0 else "certificate_failed",
        returncode=result.returncode,
        certificate=str(args.out.resolve()),
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
