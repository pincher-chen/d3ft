"""Optional cooperative exit, strictly after a validated, durable DFT point."""
import datetime
import json
import os
from pathlib import Path


def yield_after_point(work, label, mask):
    request = os.environ.get('D3FT_YIELD_REQUEST')
    if not request or not Path(request).exists():
        return
    import numpy as np
    work = Path(work)
    # A subsequent unique run replays the deterministic root search using
    # --reuse-from. No unfinished optimizer trajectory is reused.
    with (work / 'mask.npy').open('wb') as f:
        np.save(f, mask)
        f.flush()
        os.fsync(f.fileno())
    for name in [label + '.json', 'points.json']:
        with (work / name).open('rb') as f:
            os.fsync(f.fileno())
    record = dict(time=datetime.datetime.now().astimezone().isoformat(),
                  reason='E5 eight-GPU priority; completed DFT point boundary',
                  last_completed_point=label, request=request,
                  accepted_point_saved=True, resume_via='new unique run --reuse-from',
                  classification='cooperative_pause_not_physical_failure')
    target = work / 'cooperative_pause.json'
    with target.open('w') as f:
        json.dump(record, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print('COOPERATIVE_YIELD ' + json.dumps(record), flush=True)
    raise SystemExit(75)
