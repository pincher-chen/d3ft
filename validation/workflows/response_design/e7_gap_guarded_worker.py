"""Opt-in, future GaP process entry. Original running workers are untouched."""
import os
import sys

sys.path.insert(0, os.environ['D3FT_ORIGINAL_PHASE3'])
import worker
from e7_gap_budget_guard import install

install(worker, os.environ['D3FT_GAP_GUARD_POLICY'], os.environ['D3FT_GAP_LOGICAL_KEY'],
        os.environ.get('D3FT_RESUME_SOURCE'))
worker.main()
