"""Wait for an energy-FD GPU sequence and run its immutable certificate."""

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
    parser.add_argument("plan", type=Path)
    parser.add_argument("--sequence-record", type=Path, required=True)
    parser.add_argument("--sequence-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    args.record.mkdir(parents=True, exist_ok=False)

    def status(state, **fields):
        payload = {"state": state, "time": datetime.datetime.now().astimezone().isoformat(), **fields}
        temporary = args.record / "status.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.record / "status.json")

    while True:
        state_path = args.sequence_record / "status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("state") == "sequence_exited":
            failures = [row for row in state.get("results", []) if row["returncode"]]
            if failures:
                status("dependency_failed", failures=failures)
                raise SystemExit(1)
            break
        try:
            os.kill(args.sequence_pid, 0)
        except ProcessLookupError:
            status("dependency_controller_missing", dependency_state=state)
            raise SystemExit(1)
        status("waiting_for_sequence", dependency=str(args.sequence_record.resolve()))
        time.sleep(args.poll_seconds)

    command = [sys.executable, str(Path(__file__).with_name("check_energy_parameter_fd.py")),
               str(args.plan), "--out", str(args.out), "--rtol", str(args.rtol)]
    (args.record / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    status("certifying")
    with (args.record / "certificate.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    status("complete" if result.returncode == 0 else "certificate_failed",
           returncode=result.returncode, certificate=str(args.out.resolve()))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
