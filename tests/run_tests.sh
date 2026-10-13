#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exe="${D3FT_PYTHON:-python}"
"$python_exe" "$repo_root/scripts/run_tests.py" --suite "${1:-cpu}"
