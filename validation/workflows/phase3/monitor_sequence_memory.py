"""Record per-GPU memory peaks for one experiment in an existing allocation."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--jobid", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    raw = args.out.with_suffix(".csv")
    peak = {}
    started = False
    samples = 0
    start_time = None

    while True:
        if not args.status.exists():
            time.sleep(5)
            continue
        state = json.loads(args.status.read_text())
        active = state.get("state") == "running" and state.get("experiment") == args.experiment
        if active:
            if not started:
                started = True
                start_time = datetime.datetime.now().isoformat()
            try:
                output = subprocess.check_output([
                    "srun", f"--jobid={args.jobid}", "--overlap", "-N1", "-n1",
                    "--cpus-per-task=1", "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ], text=True, stderr=subprocess.STDOUT, timeout=30)
                now = datetime.datetime.now().isoformat()
                with raw.open("a") as handle:
                    for line in output.strip().splitlines():
                        gpu, memory, util = (int(x.strip()) for x in line.split(","))
                        peak[gpu] = max(peak.get(gpu, 0), memory)
                        handle.write(f"{now},{gpu},{memory},{util}\n")
                samples += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            time.sleep(args.interval)
            continue
        if started:
            break
        if state.get("state") == "sequence_exited":
            raise RuntimeError("Sequence exited before the requested experiment ran")
        time.sleep(5)

    result = {
        "experiment": args.experiment,
        "jobid": args.jobid,
        "started": start_time,
        "finished": datetime.datetime.now().isoformat(),
        "samples": samples,
        "peak_memory_MiB": peak,
        "raw": str(raw),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
