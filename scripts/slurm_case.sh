#!/usr/bin/env bash
# Run one case inside a USER-supplied existing allocation. Never cancels it.
set -euo pipefail
if [[ $# -lt 4 ]]; then echo 'Usage: bash scripts/slurm_case.sh JOBID CASE SEED NEW_OUTPUT_DIR' >&2; exit 2; fi
jobid="$1"; case_name="$2"; seed="$3"; output="$4"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exe="${D3FT_PYTHON:-python}"
srun --jobid="$jobid" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 "$python_exe" "$repo_root/scripts/run_case.py" "$case_name" --seed "$seed" --output "$output"
