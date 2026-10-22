#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo 'Usage: bash scripts/slurm_response.sh JOBID GPU_LIST STAGE [run_response.py options]'
  exit 2
fi
allocation="$1"; devices="$2"; stage="$3"; shift 3
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
srun --jobid="$allocation" --overlap --ntasks=1 \
  env CUDA_VISIBLE_DEVICES="$devices" "${D3FT_PYTHON:-python}" \
  "$repo_root/scripts/run_response.py" --stage "$stage" "$@"
